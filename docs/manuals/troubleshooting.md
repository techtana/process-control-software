# Troubleshooting

Start with the symptom you see, then work through the cause and the fix.

## Health check

Run these with the virtual environment active. Each line should succeed.

```bash
python -c "import r2r_control; print(r2r_control.__version__)"
```

```bash
pytest
```

The first prints `0.3.0`. The second ends with `92 passed`.

## Problems

| What you see | Likely cause | Fix |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Windows, with output sent to a file or pipe | Run `python -X utf8 ...`, or set `PYTHONUTF8=1`. |
| `UnicodeDecodeError` in `test_architecture.py` | An older checkout that reads sources with the Windows code page | Update to this branch, or run `python -X utf8 -m pytest`. |
| `cannot import name '_imaging' from 'PIL'` in a figure test | A broken Pillow install | `python -m pip install --ignore-installed pillow` |
| `ModuleNotFoundError: No module named 'r2r_control'` | Package not installed, or the virtual environment is not active | Activate `.venv`, then run `pip install -e ".[test]"`. |
| `EventTable frame missing columns: [...]` | A column name in `EventTable(...)` is not in the table | Check the spelling against `df.columns`. |
| `recommended and used knob columns must align one-to-one` | The two knob lists have different lengths | List the same knobs, in the same order, in both. |
| `... has shape (a, b), expected (n_out, n_knob)` | A gain matrix with the wrong orientation | M must be outputs × knobs. |
| `LinAlgError: SVD did not converge` from `identify` | An empty cell in a knob or feedforward column | Fill the gap or drop that row. Only outputs may be empty. |
| `ValueError: no measured data usable for gain identification` | No measured outputs, or every inline run is closed-loop and there is no DOE | Check the output columns, and supply a DOE or open-loop runs. |
| `fb.gain_source` is `closed-loop inline (endogenous fallback)` | No usable DOE | Treat the gain as unreliable. Run a DOE. |
| `ValueError` from `.index` in `RegressionMachine.fit` | The output name is not in `y_cols` | Pass one of the names in `y_cols`. |
| `AttributeError: unknown RunConfig field` | A misspelled setting in `default_run_config(...)` | Check the field names in `src/r2r_control/config/schema.py`. |
| `RunConfig.targets/lsl/usl are required to simulate` | Simulation requested without spec limits | Set `targets`, `lsl` and `usl` for every output. |
| Pipeline log has no "gate 3/5" line | `targets` was not set, so simulation was skipped | Set `targets`, `lsl` and `usl`. |
| Diagnosis says "unidentifiable from available data — excitation required" | Knobs barely move under tight control | This is a correct answer. Use the planner to design excitation. |
| "gain mismatch" or "QR attribution: deferred" on a loop you trust | The gain check is biased under feedback (known issue) | Confirm with the model-consistency check (supply `y_pred`) or a DOE. |
| No feedforward candidates where you expect one | Weak effect, a conservative significance test, or the gold holdout lowering selection power | Try `RegressionMachine(gold_holdout=False)`, or collect more runs. |
| Cpk shows `insufficient samples` | Fewer measured runs than `min_samples` (8) | Collect more measured runs. |
| The tests take a minute or more | The constrained MPC is slow | This is expected on slow machines. |
