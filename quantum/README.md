# Quantum experiments

Python project layout: `src/modules/` contains the importable package, `tests/` contains unit tests, `notebooks/` contains experiments, and `scripts/` contains the test runner.

From `quantum/` install with `python -m pip install -e .` and run `python -m pytest` (requires pytest). Alternatively run `bash scripts/exe_tests.sh`.

The notebooks prepend `quantum/src` to Python's module search path when started from the project root or its notebook directory.
