#!/usr/bin/env bash
# Reproduction driver for "Mutation Testing for Multi-Agent Systems".
#
#   bash reproduce.sh              # offline: recompute the paper's numbers from the logs (no API key)
#   bash reproduce.sh --verify     # same as above
#   bash reproduce.sh --live       # full live re-run (needs OPENAI_API_KEY; hours, real API cost)
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

case "${1:---verify}" in
  --verify)
    # Prints the main tables, then checks every reported number against the logs.
    "$PY" "$HERE/analyze_v2.py" "$HERE/results/v2"
    echo
    exec "$PY" "$HERE/verify_paper.py" --results "$HERE/results/v2" --audit "$HERE/audit"
    ;;
  --live)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY to run the live sweep}"
    OUT="$HERE/results/v2_rerun"
    echo "Live sweep: 1,076 mutants, about 52,000 agent executions with gpt-4.1-mini."
    "$PY" "$HERE/run_live_mutation_v2.py" --out "$OUT"
    "$PY" "$HERE/repair_v2.py" "$OUT"
    "$PY" "$HERE/analyze_v2.py" "$OUT"
    echo
    echo "Checking the fresh run against the paper (model output varies, so small differences are expected)..."
    "$PY" "$HERE/verify_paper.py" --results "$OUT" --audit "$HERE/audit" || true
    ;;
  *)
    echo "usage: bash reproduce.sh [--verify|--live]" >&2; exit 2 ;;
esac
