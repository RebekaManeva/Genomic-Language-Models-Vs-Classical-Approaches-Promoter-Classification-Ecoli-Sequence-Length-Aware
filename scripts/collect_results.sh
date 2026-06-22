#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 optimizations/collect_results.py \
  --output-root outputs \
  --out-csv outputs/final_results.csv \
  --out-json outputs/final_results.json
