# Code review, README rewrite and review fixes

**Status:** done 2026-10-26 on branch `review-fixes-and-readme` (commit
`9a2a055`). One fix was dropped; see Outcome.

## Goal

Review the 0.2.0 implementation, then rewrite the README for the software owner
with its summary, usage, assumptions, limits, known issues and outlook. Fix the
findings that change results, each with a test that fails before the fix.

## Approach

1. Read the whole package and its tests. Run the tests, the walkthrough and the
   README quickstart.
2. Reproduce each suspected defect with a small probe on the synthetic
   generator before recording it.
3. Rewrite the README, putting every review number behind a probe.
4. Fix the top four findings:
   - inline down-weight not applied;
   - inline-only gains reported as confident;
   - feedforward false positives;
   - a feedback-biased gain-mismatch check.
5. On 2026-10-26, PR #2 (0.3.0) was found to overlap. The owner chose to rebuild
   on top of it and keep only the fixes it does not cover.

## Outcome (2026-10-26)

- 0.3.0 had already fixed findings 1 and 2, so the review's versions were
  dropped.
- Fix 3 (feedforward screening on the FF sensors alone, plus a Bonferroni test)
  was kept and ported.
- Fix 4 (an ARX realized gain) was dropped. It failed on 0.3.0's
  receding-horizon MPC, so that finding is still open.
- Added: UTF-8 source reading in the tests, needed on Windows.
- The first version of the branch is kept as `review-fixes-and-readme-v1`.
