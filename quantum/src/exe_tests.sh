#!/usr/bin/env bash
set -euo pipefail

# Run from the Python source root regardless of the caller's working directory.
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOURCE_DIR"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
    echo "pytest is not installed for $PYTHON_BIN; installing..."
    "$PYTHON_BIN" -m pip install "pytest>=7.0"
fi

exec "$PYTHON_BIN" -m pytest tests -v "$@"
