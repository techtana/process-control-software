# Future work

**Status:** proposed on 2026-10-24 in
[project-closure.md §10](../../project-closure.md). No item is done yet.

## Before using real data (P1)

1. Back-test on historical data that has known control-off episodes.
2. Commit the revised specification (KI-9).
3. Recalibrate the §9 thresholds on real data and expose them through
   `RunConfig`.

## Capability (P2)

4. Recover the FF-selection power lost to the gold holdout (R-1).
5. Design FF-independent excitation automatically when the check reports
   `M_or_FF_model_error` (R-2).
6. Add a block-bootstrap-over-campaigns bias term for DOE-only identification
   (R-4), and a stronger feedback test (R-3).
7. Tune the ridge, elastic-net and PLS hyperparameters inside forward
   validation. Support multiple outputs in component 6 and in the pipeline.
8. Replace the equal variance-share allocation with a counterfactual per cause.
9. Model plant dynamics in the gain, if carry-over between runs matters.

## Productization (P3)

10. Load saved artifacts, add data connectors and a CLI, and speed up the slow
    paths.
11. Add continuous integration.
12. Remove `subspace_order`, or use it for order selection (R-6).

## Added on 2026-10-26

13. Make the gain-mismatch check unbiased under feedback. It still reads about
    1.5 on a correct model.
14. Handle missing values in the knob and FF columns. Today a single NaN crashes
    identification.
15. Make the planner use the gap in `plan()` itself, not only through the
    pipeline's `target_features`.
