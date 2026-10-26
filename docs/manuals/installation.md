# Installation manual

This manual installs `r2r_control` on one machine. It is a Python library with
no service to run and no devices to set up.

## What you need

- Python 3.9 or newer. It was last checked on 2026-10-26 with Python 3.14,
  numpy 2.5, pandas 3.1 and scipy 1.18.
- Git, to clone the repository.
- About 300 MB of disk space for the Python environment.
- An internet connection, the first time, to download the dependencies.

## Steps

1. **Get the code.** Clone the repository and open its folder.

   ```bash
   git clone https://github.com/techtana/process-control-software.git
   ```

   ```bash
   cd process-control-software
   ```

2. **Create a virtual environment.** It keeps the toolkit's packages apart from
   the rest of your system.

   ```bash
   python -m venv .venv
   ```

3. **Activate it.** On Windows (PowerShell):

   ```bash
   .venv\Scripts\activate
   ```

   On Linux or macOS:

   ```bash
   source .venv/bin/activate
   ```

   Your prompt should now start with `(.venv)`.

4. **Install the package.** The `[test]` extra adds pytest and matplotlib.

   ```bash
   pip install -e ".[test]"
   ```

   You should see `Successfully installed ... r2r-control-0.3.0`. If you only
   need the library, `pip install -e .` is enough. Add `[viz]` if you want the
   diagnostic figures without the test tools.

5. **Check the install.** Run the test suite.

   ```bash
   pytest
   ```

   You should see `92 passed`, after about 30 seconds. If any test fails, see
   [Troubleshooting](troubleshooting.md).

6. **Run the walkthrough.**

   ```bash
   python examples/walkthrough.py
   ```

   It prints one section per component and ends with `Done.`

## On Windows

Reports contain characters such as `≈`. If you send the output to a file or a
pipe, start Python in UTF-8 mode:

```bash
python -X utf8 examples/walkthrough.py
```

Or set the environment variable `PYTHONUTF8=1` once for the session.

## Updating

Pull the new code with the virtual environment active, then reinstall.

```bash
git pull
```

```bash
pip install -e ".[test]"
```

Then run `pytest` again. Read `CHANGELOG.md` for any behaviour changes.

## Removing

Delete the `.venv` folder, and the repository folder if you no longer need it.
Nothing is installed outside them.
