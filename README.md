# Mutation Testing for Multi-Agent Systems: Replication Package

This package contains everything needed to reproduce the results in the paper:
the ten workflow specifications, the mutation operators, every logged agent run,
the manual audit labels, and the scripts that turn them into the paper's tables
and figure.

## Quick start (no API key needed)

You need Python 3.9 or newer. Every number in the paper is computed from the
logged runs, so you can check them without calling any model.

The quickest check is one command. It prints the main tables and then compares
every reported number with the value recomputed from the logs, printing PASS or
FAIL for each (exit status 1 if any check fails):

```
bash reproduce.sh                                 # or: python3 verify_paper.py
```

The individual scripts are:

```
python3 analyze_v2.py results/v2                  # main tables and size-matched suites
python3 analyze_control_combined.py               # noise control and noise-adjusted scores
python3 audit_v2.py score results/v2 --audit audit/   # equivalence audit and verdict check
```

Each command prints its results to the screen. The output of the first one is
also saved in `analysis.txt`.

To redraw the heatmap figure (needs `matplotlib`):

```
python3 make_heatmap_v2.py "$PWD" figs/heatmap_ms_strict.pdf
```

## What is in the package

| Folder or file | Contents |
|---|---|
| `benchmarks/` | the ten workflow specifications used in the study (`SHA256SUMS` lists their hashes) |
| `src/agentcov/` | turns a specification into a coordination graph and its test obligations |
| `src/agentmut/` | the 14 mutation operators, the four test suites, and the kill criteria |
| `src/agent_runtime/` | runs a workflow through the OpenAI Agents SDK |
| `suites/` | the test prompts, one file per workflow |
| `results/v2/` | every logged run of the main experiment |
| `results/v2_control/` | two extra noise-control runs of the unmodified workflows |
| `audit/` | both raters' labels, their disagreements, and the agreed labels |
| `figs/` | the heatmap figure |

## Running the experiment again (needs an OpenAI API key)

This repeats the full experiment: 1,076 mutants and about 52,000 agent
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
- Seed: `20260605`
- Each test runs 3 times on the original and 3 times on the mutant; a mutant is
  killed when most of the paired runs differ
- At most 50 mutants per operator and workflow (10 for `octagon_vc_agents`)
- The noise control runs every test 3 times on the unmodified workflow, three
  separate times

## Helper scripts

| Script | Purpose |
|---|---|
| `reproduce.sh` | runs the offline checks (`--verify`, default) or the full live re-run (`--live`) |
| `verify_paper.py` | checks every reported number against the logs |
| `run_live_mutation_v2.py` | runs the experiment and logs every run |
| `run_control_extra.py` | runs extra noise controls on the unmodified workflows |
| `analyze_v2.py` | main tables and size-matched suites |
| `analyze_control_combined.py` | combines the noise controls and computes noise-adjusted scores |
| `repair_v2.py` | finds runs that failed for technical reasons so they can be redone |
| `dedup.py` | removes duplicate records if two runs overlapped |
| `audit_v2.py`, `merge_audit.py` | create the audit sheets, merge two raters, and score them (agreement, equivalence, verdict precision and class-weighted recall) |
| `make_heatmap_v2.py` | draws the heatmap figure |

`reproduce.py`, `run_live_mutation.py` and `results/*.json` belong to an earlier
pilot run and are not used for the numbers in the paper.
