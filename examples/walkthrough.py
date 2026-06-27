"""End-to-end walkthrough of the R2R process-control toolchain.

Runs the whole loop on a synthetic, fault-injectable closed-loop dataset:

    identify (1) -> diagnose gap / plan (2) -> simulate (3) ->
    optimize (4) -> diagnose (5) -> regress all data (6)

Run with:  python examples/walkthrough.py
"""

import numpy as np

from process_control import synthetic
from process_control.components import (
    FBIdentifier, ExperimentPlanner, PlannerConfig,
    Simulator, ControllerConfig, PlantConfig,
    ControllerOptimizer, SearchDimension,
    DiagnosticsEngine, RegressionMachine)


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
          f"{fb.identifiable.n_directions}, "
          f"relative uncertainty {fb.relative_uncertainty():.3f}")
    print(f"    FB-08 boundary recorded: "
          f"{[a.name for a in fb.provenance.assumptions if not a.holds]}")

    # ----- Component 2: experiment planner ---------------------------------
    pl = ExperimentPlanner(PlannerConfig(primary_objective="D", budget=5), seed=0)
    plan = pl.plan(ds.inline.regressors(), ds.inline.regressor_names,
                   ds.inline.knob_cols_used)
    print(f"\n[2] experiment planner: {plan.gap.confounding_summary}")
    print(f"    proposed {len(plan.proposals)} experiments; "
          f"top excites {plan.proposals[0].excites_directions} "
          f"(priority {plan.proposals[0].priority:.3f})")

    # ----- Component 3: simulator ------------------------------------------
    sim = Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                    FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                    controller=ControllerConfig(mpc_R=0.05, move_limit=1.0),
                    rel_uncertainty_M=fb.relative_uncertainty(), seed=0)
    res = sim.simulate(n_steps=300)
    print(f"\n[3] simulator: std on={res.metrics_on.std} off={res.metrics_off.std}")
    print(f"    Cpk on={res.metrics_on.cpk} off={res.metrics_off.cpk}")

    # ----- Component 4: optimizer ------------------------------------------
    def factory(cfg):
        return Simulator(M=t.M, targets=t.targets, lsl=lsl, usl=usl, u0=t.u0,
                         FF_gain=t.FF_gain, plant=PlantConfig(meas_noise_std=0.2),
                         controller=cfg, seed=0)

    dims = [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
            SearchDimension("mhe_R", 0.1, 10.0, log=True),
            SearchDimension("move_limit", 0.2, 1.5)]
    opt = ControllerOptimizer(factory, dims, objective_metric="std",
                              n_steps=200, n_mc=4, seed=1)
    best = opt.optimize(algorithm="bayesian", budget=24)
    print(f"\n[4] optimizer: best std={best.best_objective:.3f} "
          f"after {best.n_evals} evals ({best.rejected_unstable} rejected unstable)")
    print(f"    best config: { {k: round(v,3) for k,v in best.best_config.items()} }")
    print(f"    sensitivity: { {k: round(v,3) for k,v in best.sensitivity.items()} }")

    # ----- Component 5: diagnostics ----------------------------------------
    diag = DiagnosticsEngine().diagnose(
        res.y_on, res.u, fb.M, state_est=res.state_est, ff_inputs=res.ff_inputs,
        innovation=res.innovation, ff_names=ds.ff_names)
    print(f"\n[5] diagnostics: {diag.achievability['verdict']}")
    print(f"    variance shares: "
          f"{ {k: round(v,2) for k,v in diag.variance_decomposition['shares'].items()} }")
    print(f"    FF opportunities: {diag.ff_leakage['feedforward_opportunities']}")

    # ----- Component 6: regression machine ---------------------------------
    rm = RegressionMachine(seed=0, n_stability_resamples=80)
    reg = rm.fit(ds.inline, "y0", M=fb.M, u0=t.u0, control_present=True,
                 adjustable_features=ds.inline.knob_cols_used)
    print(f"\n[6] regression: estimator={reg.estimator}, "
          f"forward error={reg.forward_error:.3f}, decoupled={reg.decoupled}")
    print(f"    feedforward candidates: {reg.feedforward_candidates}")
    print(f"    top features: "
          f"{[(n, round(f,2)) for n,f,_ in reg.importance.ranked(top=4)]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
