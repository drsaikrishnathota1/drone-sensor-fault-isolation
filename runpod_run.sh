#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_FILE="${1:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

export PYTHONUNBUFFERED=1

python run_study.py --config "$CONFIG_FILE"

echo
echo "Study completed using: $CONFIG_FILE"
