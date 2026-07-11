# Replication Package — *Mutation Testing for Multi-Agent Systems*

Self-contained artifact for the paper. It reproduces every table exactly from
the committed inputs and results, with a single command, no API key, and no
third-party dependencies.

```
python3 reproduce.py       # regenerate & verify Tables II–VI  (offline, ~seconds)
```

(`bash reproduce.sh` is an optional wrapper around the same thing, for those who
prefer it.)

## Setup

**Offline verification** (`python3 reproduce.py`) needs only **Python 3.9+** —
no virtual environment and no `pip install`. It imports only the standard
library and the bundled `src/agentcov` and `src/agentmut`.

Check your Python:

```
python3 --version        # 3.9 or newer
```

**Live re-run** (`--live`) additionally needs the OpenAI Agents SDK. Create an
isolated environment and install the pinned dependencies:

```
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # Windows: set OPENAI_API_KEY=sk-...
```

This recomputes the benchmark inventory and the full mutant set **from the
source manifests**, aggregates the committed live results, and checks every cell
against the published values — printing `PASS` only if they all match.

## What's inside

```
mutation-testing-replication/
├── reproduce.sh              # driver: --verify (default) | --live
├── reproduce_tables.py       # offline verifier (stdlib only; Tables II–VI + counts)
├── benchmarks/               # the 10 schema-v3 workflow manifests (paper inputs)
├── src/
│   ├── agentcov/             # manifest parser, coordination graph, obligation extractor
│   ├── agentmut/             # 14 mutation operators, suite builders, kill criteria, generator
│   └── agent_runtime/        # OpenAI Agents SDK execution layer (LIVE re-run only)
├── run_live_mutation.py      # the live sweep that produced the paper's numbers
├── aggregate_live_mutation.py
├── suites/                   # cached realized prompts, one per benchmark (for faithful re-run)
├── results/                  # canonical raw per-(operator,suite,criterion) results — the paper's data
├── provenance.json           # seed, model, git commit, per-benchmark manifest SHA-256
└── requirements.txt          # deps for --live only
```

## Tier 1 — Offline verification (no API key, no dependencies)

```
python3 reproduce.py              # or, with bash: bash reproduce.sh
```

| Regenerated | Paper element | Source |
|---|---|---|
| Benchmark inventory (49 / 47 / 65 / 248 / 41 / 403) | Table II | `benchmarks/` via `agentcov` |
| First-order mutant count (1,873) | §III-B | `benchmarks/` via `agentmut` |
| Executed pool + live-killable denominators (1,084 / 983 / 1,010 / 177) | Table III | `results/` |
| K-Struct MS_strict per family (T-Dev⋆ 0.717 …) | Table IV | `results/` |
| MS_strict per suite × criterion (K-Safety T-Dev⋆ 0.559 …) | Table V | `results/` |
| K-Outcome MS_strict per operator | Table VI | `results/` |
| Live agent executions (29,601) | §IV | `results/` |

`--csv out/` also dumps each table as CSV.

## Tier 2 — Full live re-run (needs `OPENAI_API_KEY`)

After completing the live-re-run setup above (venv + `requirements.txt` +
`OPENAI_API_KEY`):

```
python3 reproduce.py --live        # or: bash reproduce.sh --live
```

Configuration is fixed for parity with the paper: seed `20260605`,
`MAX_PER_OP=50` (10 for `octagon_vc_agents`), `k=3` repeats with per-test
majority vote, `gpt-4.1-mini` as the agent model. A full 10-benchmark sweep is
~1,084 mutants and ~30k executions (hours; non-trivial API cost).

**What `--live` regenerates, step by step** (all from `benchmarks/` — nothing in
`results/` is read):

1. normalize each manifest into a coordination graph and extract obligations
   (`src/agentcov`);
2. generate all first-order mutants and cap per operator (`src/agentmut` →
   1,873 generated, 1,084 executed);
3. build the four suites by obligation selection (`src/agentmut.build_suites`);
4. execute every (spec, test) pair against the original and each mutant through
   the OpenAI Agents SDK, `k=3` with per-test majority vote
   (`src/agent_runtime`), and score K-Struct / K-Outcome / K-Safety;
5. write fresh per-benchmark results to `results/live_rerun/`;
6. `reproduce.py` then aggregates and checks them against the paper.

Because agent execution is stochastic, a fresh run reproduces the paper up to
the residual flaky-kill rate, not byte-for-byte — the offline tier is the
exact-match check.

**Prompt realization (reused, not regenerated).** Each obligation is turned into
one natural-language user prompt by an LLM prompt-realizer. Those prompts are
provided in `suites/*.suite.json` (realized once with `gpt-4.1-mini`) and the
sweep reuses them, so a re-run tests the *same* prompts and stays comparable to
the paper. Regenerating the prompts is intentionally **not** part of the re-run:
new prompts would change the tests and the scores. The realized prompts, their
`objective_id`, target agent, model, and seed hash are all recorded in
`suites/`, so the realization is fully auditable. A standalone prompt-realizer
for regenerating `suites/` from the manifests may be added in a later revision.

## Notes

- The 10 manifests match the SHA-256 fingerprints in `provenance.json` — they
  are the byte-exact inputs behind the paper.
- `agentmut` and `agentcov` are pure Python (no third-party deps); the offline
  verifier imports only them and the standard library.
