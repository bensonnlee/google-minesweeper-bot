#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Find Python 3
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null && python --version 2>&1 | grep -q "Python 3"; then
    PY=python
else
    echo "Error: Python 3 is required but not found."
    echo "Install it from https://www.python.org/downloads/"
    exit 1
fi

echo "Using $($PY --version)"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PY -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install/update dependencies
pip install -q -r requirements.txt

echo "Starting bot..."
echo ""
python bot.py "$@"
