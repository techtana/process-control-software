# Changelog

## Project closure — 2026-10-24 (documentation only)

No algorithm or behavior changes; the package version stays 0.2.0.

- Added `docs/project-closure.md`. It covers the assumptions (guarded and
  unguarded), applicability, methodology, limitations, nine known issues with
  evidence and suggested fixes, the heuristic thresholds, and prioritized
  future work.
- Re-verified on Python 3.11 with numpy 2.4, pandas 3.1 and scipy 1.17:
  64 tests pass with no warnings, and the walkthrough runs end to end.
- README: corrected the test count (63 → 64), added a project-status section
  and a documentation index, and corrected the layer order (`config` was
  missing).
- `docs/requirements-traceability.md`: added the requirements found to be only
  partially met, mapped to the known issues.
- `docs/architecture.md`: added a pointer to the known issues.
- `.gitignore`: ignore generated files under `artifacts/` (the placeholder
  `.gitkeep` stays).

## 0.2.0 — Re-architecture to revised specification

Aligns the codebase with the revised design requirements specification
(spec §14 corrections, §1.4 load-bearing assumptions A1–A6, §15 edge cases
E1–E14, and the §16 repository layout).

### Architectural corrections
- **De-unified the estimator (§1.3, REG-01).** Components 1 and 6 are no longer
  one class. They share a `SensitivityCore` (cross-sectional gain estimation)
  and the cross-cutting services, but Component 1 wraps it with a dynamics layer
  and Component 6 with a regularized-selection layer.
- **Structure-preserving empirical noise floor (NF-03, E6).** Replaced the plain
  per-column shuffle (which destroys autocorrelation and understates the floor)
  with block-permutation / phase-randomization surrogates that preserve marginal
  distribution, autocorrelation, and missingness — sharing the block logic with
  the block bootstrap (VAL-05).
- **Finite-sample metric uncertainty (SIM-06, E10).** Cpk/Harris on real sparse
  data carry confidence intervals, and a minimum-sample gate suppresses
  under-powered metrics ("insufficient samples").
- **Diagnostic precedence (DIAG-07, E4, E5).** Diagnostics run as an explicit
  decision tree behind an excitation-sufficiency gate; nodes can emit
  "unidentifiable from available data — excitation required".
- **Load-bearing assumptions made explicit (§1.4, A1–A6)** with named guards:
  gain-drift monitor (A1/E1), u0 validity horizon (A2/E3), phantom-FF quarantine
  (A3/E2), out-of-sequence measurement routing (A4/E7), control-off
  representativeness check (A5/E8), closed-loop excitation discount (A6/E9).
- Group-aware stability selection (E12) and integer/categorical-aware GP
  helpers (E11).

### Repository structure (§16)
- Package renamed `process_control` → `r2r_control`, moved under `src/`.
- Strict layers `contracts → core → estimation → {controller, metrics} →
  components → pipeline`, enforced by `tests/test_architecture.py`.
- Components no longer import one another; the optimizer takes a simulator
  factory, and the pipeline wires components together.

## 0.1.0 — Initial implementation

First implementation against the original specification: six components on one
shared core, 39 tests including deliberate identifiability-limit faults.
