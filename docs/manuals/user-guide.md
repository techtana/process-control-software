# User manual

This manual is for the engineer who analyses a run-to-run loop with the
toolkit. Everything is done from Python; there is no screen or user interface.

## 1. Prepare your data

Put one processed run per row in a table (CSV or a pandas DataFrame), in time
order.

| Column | Required | What goes there |
|---|---|---|
| Run number | yes | A numeric run counter, used as `time_col`. Use numbers, not dates. |
| Recommended knob values | yes | One column per knob: what the controller asked for. |
| Used knob values | yes | One column per knob, in the same order: what was actually applied. **No empty cells.** |
| Feedforward inputs | optional | Upstream measurements, tool sensors, configuration. Numbers only. **No empty cells.** |
| Outputs | yes | One column per measured output. Leave the cell empty when a run was not measured. |
| Control off | recommended | `True` for runs made with the controller switched off. These runs set the open-loop baseline. |
| Controller prediction | optional | The controller's own prediction of each output, which enables the model-consistency check. |

Put a DOE, if you have one, in a second table with the same columns. A fresh
DOE matters a great deal: without one, the gain comes from closed-loop data and
is marked unreliable.

## 2. Load it

```python
import numpy as np
import pandas as pd
from r2r_control.contracts import EventTable
from r2r_control.components import FBIdentifier, DiagnosticsEngine, RegressionMachine

df = pd.read_csv("runs.csv")
table = EventTable(
    frame=df,
    knob_cols_recommended=["dose_rec", "focus_rec"],
    knob_cols_used=["dose_used", "focus_used"],
    ff_cols=["pre_thickness", "chamber_temp"],
    y_cols=["cd", "overlay"],
    time_col="run",
    control_off_col="control_off",
)
```

Load the DOE the same way, as `doe_table`.

## 3. Identify the gain

```python
fb = FBIdentifier().identify(table, doe_table, doe_age_days=30)
print(fb.gain_source)
print(fb.M)
print(fb.relative_uncertainty())
print(fb.low_confidence_entries)
```

Here is what normal output looks like:

- `fb.gain_source` is `DOE only` or `DOE + open-loop inline`.
- `fb.low_confidence_entries` is an empty list.
- The relative uncertainty is a few percent.

If `gain_source` says `closed-loop inline (endogenous fallback)`, no clean gain
source was available. Every entry is then low-confidence, so don't use the gain
for decisions. Run a DOE first.

`print(fb.provenance.to_json())` shows the full record: what data was used or
rejected and why, the noise floor, and the assumptions checked.

## 4. Diagnose the loop as it ran

```python
Y = table.y_matrix()
measured = np.all(np.isfinite(Y), axis=1)
diag = DiagnosticsEngine().diagnose(Y, table.used_knobs(), fb.M,
                                    ff_inputs=table.ff(), measured_mask=measured)
print("\n".join(diag.precedence_log))
```

The log has one line per check, in this order:

1. excitation
2. gain mismatch
3. FF leakage
4. QR (estimator versus controller tuning)
5. variance decomposition
6. achievability

There is an extra line, 2b, when you supply controller predictions.

- **"at floor"** means the controller is already as good as it can be.
- **"unidentifiable from available data — excitation required"** is a correct
  answer, not an error. It means the data cannot settle that question.
- Treat a "gain underestimated" result without oscillation with caution; the
  check is known to be biased under feedback.

## 5. Look for feedforward candidates

```python
reg = RegressionMachine().fit(table, "cd", M=fb.M, u0=fb.u0,
                              rel_uncertainty_M=fb.relative_uncertainty())
print(reg.feedforward_candidates)
print(reg.forward_error)
```

Candidates are sensors that explain the output beyond what the controller
already handles. An empty list is a valid answer. Ignore the candidates if the
gain was low-confidence, because gain error can create false ones.

## 6. Plan experiments

`ExperimentPlanner` proposes experiments for the knobs the data cannot pin
down. Pass those knobs as `target_features`, and the knob ranges in engineering
units through `PlannerConfig(knob_bounds=...)`. A person must approve every
proposal; nothing is run automatically. `examples/walkthrough.py` shows the
call.

## 7. Simulate and tune

`Simulator` runs the loop with and without control and reports Cpk, standard
deviation, off-target mean, %OOS and the Harris index, with uncertainty bands.
`ControllerOptimizer` searches the controller settings against it. Both need
per-output targets and spec limits. `examples/walkthrough.py` shows both.

## The whole chain in one call

`GatePipeline(RunConfig(...)).run(table, doe_table, doe_age_days=30)` runs all
six steps. Its diagnosis describes a *simulated* loop built from your gain, not
your process, so use step 4 to diagnose real data.

## Try it without data

`r2r_control.synthetic.make_dataset(seed=0)` returns a synthetic inline table,
DOE and ground truth in the same format, so you can practise every step above.
