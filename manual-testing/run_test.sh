#!/bin/bash
# Run manual tests with correct Python path

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_SRC="$PROJECT_ROOT/python"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Activate venv and set PYTHONPATH
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$PYTHON_SRC:$PYTHONPATH"

# Run the test
if [ -z "$1" ]; then
    echo "Usage: ./run_test.sh <test_file.py>"
    echo "Example: ./run_test.sh 01-python-sdk-core/initialization/test_initialization.py"
    exit 1
fi

python3 "$SCRIPT_DIR/$1"
