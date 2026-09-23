"""End-to-end walkthrough of the R2R process-control toolchain.

Runs the whole loop on a synthetic, fault-injectable closed-loop dataset:

    identify (1) -> diagnose gap / plan (2) -> simulate (3) ->
    optimize (4) -> diagnose (5) -> regress all data (6)

then shows the assumption guards (§1.4) firing on injected faults.

Run with:  python examples/walkthrough.py     (after `pip install -e .`)
"""

import numpy as np

from r2r_control import synthetic
from r2r_control.components import (
    FBIdentifier, ExperimentPlanner, PlannerConfig,
    Simulator, ControllerConfig, PlantConfig,
    ControllerOptimizer, SearchDimension,
    DiagnosticsEngine, RegressionMachine)
from r2r_control.components.diagnostics import gain_drift_monitor
from r2r_control.metrics import compute_metrics


def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 70)
    print("Run-to-Run Process Control — end-to-end walkthrough")
    print("=" * 70)

    # ----- data ------------------------------------------------------------
    ds = synthetic.make_dataset(seed=0, doe_age_days=20, ff_leak=0.4)
    t = ds.truth
    lsl, usl = t.targets - 3.0, t.targets + 3.0
    print(f"\n[data] inline events={ds.inline.n_events}, knobs={ds.inline.n_knob}, "
          f"outputs={ds.inline.n_out}, ff={t.n_ff}; "
          f"measured y0={ds.inline.effective_measured_count('y0')}")

    # ----- Component 1: FB identification ----------------------------------
    fb = FBIdentifier(estimator="svd", seed=0).identify(ds.inline, ds.doe, doe_age_days=20)
    gain_err = np.linalg.norm(fb.M - t.M) / np.linalg.norm(t.M)
    print(f"\n[1] FB identification: relative gain error = {gain_err:.3f}")
    print(f"    effective rank {fb.identifiable.effective_rank}/"
          f"{fb.identifiable.n_directions}, relative uncertainty "
          f"{fb.relative_uncertainty():.3f}")
    print(f"    u0 source: {fb.baseline_info.get('source')}")
    print(f"    assumptions relied on: {list(fb.as_artifact().assumption_ledger())}")
    print(f"    refused (FB-08): {[a.name for a in fb.provenance.assumptions if not a.holds]}")

    # ----- Component 2: experiment planner ---------------------------------
    for loop in ("open", "closed"):
        plan = ExperimentPlanner(PlannerConfig(primary_objective="D", budget=5,
                                               loop_status=loop), seed=0).plan(
            ds.inline.regressors(), ds.inline.regressor_names, ds.inline.knob_cols_used)
        print(f"\n[2] experiment planner (loop {loop}): {plan.gap.confounding_summary}")
        print(f"    {len(plan.proposals)} proposals; top info gain "
              f"{plan.proposals[0].information_gain:.3f} (A6 discount applied if closed)")

    # ----- Component 3: simulator ------------------------------------------
    def factory(cfg):
        ctrl = ControllerConfig()
        for k, v in cfg.items():
            setattr(ctrl, k, v)
        return Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                         FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                         controller=ctrl, rel_uncertainty_M=fb.relative_uncertainty(),
                         seed=0)

    res = factory({"mpc_R": 0.05, "move_limit": 1.0}).simulate(n_steps=300)
    print(f"\n[3] simulator: std on={res.metrics_on.std} off={res.metrics_off.std}")
    print(f"    Cpk on={res.metrics_on.cpk} (MC band "
          f"{res.metrics_on.bands['cpk_p05']}–{res.metrics_on.bands['cpk_p95']})")

    # ----- Component 4: optimizer (takes the factory, never imports C3) -----
    dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
            SearchDimension("mhe_R", 0.1, 10.0, log=True),
            SearchDimension("move_limit", 0.2, 1.5)]
    best = ControllerOptimizer(factory, dims, objective_metric="std", n_steps=200,
                               n_mc=4, seed=1).optimize(algorithm="bayesian", budget=24)
    print(f"\n[4] optimizer: best robust std={best.best_objective:.3f} after "
          f"{best.n_evals} evals ({best.rejected_unstable} rejected unstable)")
    print(f"    best config: { {k: round(v, 3) for k, v in best.best_config.items()} }")
    print(f"    sensitivity: { {k: round(v, 3) for k, v in best.sensitivity.items()} }")

    # ----- Component 5: diagnostics (DIAG-07 precedence tree) --------------
    diag = DiagnosticsEngine().diagnose(
        res.y_on, res.u, fb.M, state_est=res.state_est, ff_inputs=res.ff_inputs,
        innovation=res.innovation, ff_names=ds.ff_names)
    print("\n[5] diagnostics precedence:")
    for line in diag.precedence_log:
        print(f"    {line}")
    print(f"    variance shares: "
          f"{ {k: round(v, 2) for k, v in diag.variance_decomposition['shares'].items()} }")

    # ----- Component 6: regression machine ---------------------------------
    reg = RegressionMachine(seed=0, n_stability_resamples=80).fit(
        ds.inline, "y0", M=fb.M, u0=fb.u0, adjustable_features=ds.inline.knob_cols_used,
        rel_uncertainty_M=fb.relative_uncertainty())
    print(f"\n[6] regression: estimator={reg.estimator}, forward error="
          f"{reg.forward_error:.3f}, decoupled={reg.decoupled}")
    print(f"    feedforward candidates: {reg.feedforward_candidates}; "
          f"quarantined (phantom FF): {reg.quarantined_sensors}")

    # ----- guards firing on injected faults (§1.4 / §15) --------------------
    print("\n[guards]")
    tight = factory({"mpc_R": 5.0, "move_limit": 0.03}).simulate(n_steps=300)
    d = DiagnosticsEngine().diagnose(tight.y_on, tight.u, t.M, innovation=tight.innovation)
    print(f"    E4 tight control -> routed to planner: {d.routed_to_planner}")

    drift = synthetic.make_dataset(seed=7, gain_drift=1.2, n_events=600,
                                   override_fraction=0.3, control_off_episodes=0,
                                   metrology_sparsity=0.0)
    Y = drift.inline.y_matrix()
    mon = gain_drift_monitor(Y, drift.inline.used_knobs(), window=80)
    print(f"    A1/E1 gain drift -> {mon['detail']}")

    coupled = synthetic.make_dataset(seed=8, coupled_sensor_knob=(0, 1))
    M_err = coupled.truth.M.copy()
    M_err[:, 1] *= 1.6
    r = RegressionMachine(seed=0, n_stability_resamples=40).fit(
        coupled.inline, "y0", M=M_err, u0=coupled.truth.u0, rel_uncertainty_M=0.3)
    print(f"    A3/E2 phantom feedforward -> quarantined {r.quarantined_sensors}")

    few = compute_metrics(res.y_on[:5], t.targets, lsl, usl, n_effective=np.full(t.n_out, 5),
                          min_samples=8, finite_sample_ci=True)
    print(f"    E10 five-sample Cpk -> {few.to_dict()['note']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
