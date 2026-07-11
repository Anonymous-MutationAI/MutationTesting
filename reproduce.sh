#!/usr/bin/env bash
# Reproduction driver for "Mutation Testing for Multi-Agent Systems".
#
#   bash reproduce.sh              # offline: regenerate & verify all tables (no API key)
#   bash reproduce.sh --verify     # same as above
#   bash reproduce.sh --live       # full live re-run from manifests (needs OPENAI_API_KEY)
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHES="autopitch deep_research_clone oai_customer_service oai_financial_research \
oai_message_filter oai_research_bot octagon_vc_agents social_media_agent_system \
value_investment ydmitry_deep_research"

case "${1:---verify}" in
  --verify)
    # Recomputes Tables II-VI, the 1,873 mutant count, and 29,601 executions
    # from benchmarks/ (via src/) and results/, and checks them vs. the paper.
    exec python3 "$HERE/reproduce_tables.py"
    ;;
  --live)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY to run the live sweep}"
    echo "Live sweep: 1,084 mutants, ~30k agent executions (hours, non-trivial cost)."
    python3 "$HERE/run_live_mutation.py" --benchmarks $BENCHES --out "$HERE/results/live_rerun"
    echo "Verifying the fresh run against the paper..."
    python3 "$HERE/reproduce_tables.py" --results "$HERE/results/live_rerun"
    ;;
  *)
    echo "usage: bash reproduce.sh [--verify|--live]" >&2; exit 2 ;;
esac
