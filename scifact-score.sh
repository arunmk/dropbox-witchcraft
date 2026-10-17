#!/usr/bin/env bash
# Score SciFact with the current default configuration.
# Use scripts/eval_suite.py directly for other configurations or datasets.
set -euo pipefail
cd "$(dirname "$0")"
python="env/bin/python"
[ -x "$python" ] || python=python3
exec "$python" scripts/eval_suite.py run --suite scifact --configs "${1:-default}" --out bench-results/eval/scifact
