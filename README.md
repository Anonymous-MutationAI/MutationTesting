# Mutation Testing for Multi-Agent Systems: Replication Package

This package contains everything needed to reproduce the results in the paper:
the ten workflow specifications, the mutation operators, every logged agent run,
the manual audit labels, and the scripts that turn them into the paper's tables
and figure.

## Quick start (no API key needed)

You need Python 3.9 or newer. Every number in the paper is computed from the
logged runs, so you can check them without calling any model.

The quickest check is one command. It prints the main tables and then compares
every reported number (listed in `paper_values.json`) with the value recomputed
from the logs, printing PASS or FAIL for each (exit status 1 if any check fails):

```
bash reproduce.sh                                 # or: python3 verify_paper.py
```

The individual scripts are:

```
python3 analyze_v2.py results/v2                  # main tables, tests, size-matched suites
python3 analyze_v2.py results/v2 --legacy         # the same under the first-submission scoring
python3 sensitivity_v2.py results/v2              # effect of each scoring rule, test-retest check
python3 analyze_control_combined.py               # all three noise controls, noise-adjusted scores (--legacy too)
python3 audit_v2.py score results/v2 --audit audit/   # equivalence audit and verdict check
```

Each command prints its results to the screen. Saved outputs: `analysis.txt`,
`analysis_legacy.txt`, `analysis_control.txt`, `analysis_control_legacy.txt`
and `sensitivity.txt`.

## How mutants are scored

`src/agentmut/rules.py` holds three rules that the live runner applies while
executing and `analyze_v2.py` applies to the logs, so both give the same result:

1. **One mutant per runtime specification.** Some redirections produce a mutant
   that behaves exactly like another one; for example, redirecting a handoff to
   a child the agent already delegates to is the same as removing the handoff.
   Only one mutant per effective specification is scored (136 of the 1,076
   logged mutants are such duplicates, all Deleg-Red or Tool-Red).
2. **Only executed tests can kill.** A test whose target agent was removed
   (Ag-Rem) cannot run; it is logged but does not count as a kill.
3. **A fault must be reached.** A Tool-Fault mutant can only be killed by a run
   that actually calls the faulted tool.

`--legacy` turns the three rules off and reproduces the numbers of the first
submission; `sensitivity_v2.py` shows the effect of each rule separately.

Every benchmark tool is bound to `_generic_backend` in
`src/agent_runtime/live_runtime_backends.py`: tools take no arguments and return
a fixed acknowledgement, so runs differ only in which agents call which tools
and which handoffs they take.

To redraw the heatmap figure (needs `matplotlib`):

```
python3 make_heatmap_v2.py "$PWD" figs/heatmap_ms_strict.pdf
```

## What is in the package

| Folder or file | Contents |
|---|---|
| `benchmarks/` | the ten workflow specifications used in the study (`SHA256SUMS` lists their hashes) |
| `src/agentcov/` | turns a specification into a coordination graph and its test obligations |
| `src/agentmut/` | the 14 mutation operators, the four test suites, and the scoring rules (`rules.py`) |
| `src/agent_runtime/` | runs a workflow through the OpenAI Agents SDK |
| `suites/` | the test prompts, one file per workflow |
| `results/v2/` | every logged run of the main experiment |
| `results/v2_control/` | noise-control replicates 2 and 3 of the unmodified workflows (2,418 runs, 2026-11-15) |
| `audit/` | both raters' labels, their disagreements, and the agreed labels |
| `figs/` | the heatmap figure |
| `paper_values.json` | every number reported in the paper, checked by `verify_paper.py` |
| `provenance_v2.json` | settings and dates of the main experiment and the extra noise controls (`provenance.json` describes the pilot) |

## Running the experiment again (needs an OpenAI API key)

This repeats the full experiment: 940 mutants and about 46,000 agent
executions with `gpt-4.1-mini`. It takes several hours and costs real API
credit. Results will differ slightly from the paper, because model output
varies from run to run.

1. Install the dependencies and set your key:

   ```
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   export OPENAI_API_KEY=your-key     # Windows: set OPENAI_API_KEY=your-key
   ```

2. Check that everything works without calling the model:

   ```
   python3 run_live_mutation_v2.py --dry-run --out /tmp/test
   ```

3. Run the experiment and the extra noise controls:

   ```
   python3 run_live_mutation_v2.py --out results/v2_rerun
   python3 run_control_extra.py --reps 2 --src results/v2_rerun --out results/v2_rerun_control
   ```

   Both commands can be stopped and started again; they continue where they
   left off.

4. If some calls failed (for example because of rate limits), find and redo them:

   ```
   python3 repair_v2.py results/v2_rerun          # list failed runs
   python3 repair_v2.py results/v2_rerun --fix    # remove them, then rerun step 3
   ```

5. Analyze the new runs:

   ```
   python3 analyze_v2.py results/v2_rerun
   ```

   `analyze_control_combined.py` reads `results/v2` and `results/v2_control`;
   to use it on a new run, copy the new folders over those two.

## Settings used in the paper

- Model: `gpt-4.1-mini`, with the SDK's default sampling settings and at most 6 turns
- Main experiment run 2026-10-05 to 2026-10-13; extra noise controls 2026-11-15 (Python 3.11)
- Seed: `20260605`
- Each test runs 3 times on the original and 3 times on the mutant; a mutant is
  killed when most of the paired runs differ
- At most 50 mutants per operator and workflow (10 for `octagon_vc_agents`), then
  one mutant per runtime specification
- Tools take no arguments and return a fixed acknowledgement
- The noise control runs every test 3 times on the unmodified workflow, three
  separate times

## Helper scripts

| Script | Purpose |
|---|---|
| `reproduce.sh` | runs the offline checks (`--verify`, default) or the full live re-run (`--live`) |
| `verify_paper.py` | checks every reported number against the logs |
| `run_live_mutation_v2.py` | runs the experiment and logs every run |
| `run_control_extra.py` | runs extra noise controls on the unmodified workflows |
| `analyze_v2.py` | main tables, paired tests, per-benchmark scores and size-matched suites (`--legacy` for the first-submission scoring) |
| `sensitivity_v2.py` | effect of each scoring rule and the test-retest check on duplicate mutants |
| `analyze_control_combined.py` | combines the noise controls and computes noise-adjusted scores |
| `repair_v2.py` | finds runs that failed for technical reasons so they can be redone |
| `dedup.py` | removes duplicate records if two runs overlapped |
| `audit_v2.py`, `merge_audit.py` | create the audit sheets, merge two raters, and score them (agreement, equivalence, verdict precision and class-weighted recall) |
| `make_heatmap_v2.py` | draws the heatmap figure |

`reproduce.py` and `results/*.json` belong to an earlier pilot run and are not
used for the numbers in the paper. `run_live_mutation.py` is the pilot runner;
the v2 scripts reuse its helpers (live manifest conversion, incident
obligations, test templates and the verdict classifier).
