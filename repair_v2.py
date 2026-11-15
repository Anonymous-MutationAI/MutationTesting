#!/usr/bin/env python3
"""Find and remove log records that contain unscored (failed) runs, so that
run_live_mutation_v2.py re-executes exactly those on the next resume.

  python repair_v2.py results/v2            # report only
  python repair_v2.py results/v2 --fix      # back up logs, drop bad records
"""
import json, glob, shutil, sys
from pathlib import Path

d = Path(sys.argv[1]); fix = "--fix" in sys.argv
tot = {"orig": 0, "control": 0, "mutant": 0}
for p in sorted(d.glob("*.jsonl")):
    recs = [json.loads(l) for l in p.open()]
    bad_orig = {tuple(r["obl"]) for r in recs if r["type"] == "orig" and any(x is None for x in r["runs"])}
    keep, drop = [], {"orig": 0, "control": 0, "mutant": 0}
    for r in recs:
        t = r["type"]
        if t == "orig":
            bad = tuple(r["obl"]) in bad_orig
        elif t == "control":
            bad = any(x is None for x in r["runs"]) or tuple(r["obl"]) in bad_orig
        elif t == "mutant":
            bad = any(any(x is None for x in q.get("runs", [])) for q in r["per_obl"]) \
                  or any(tuple(q["obl"]) in bad_orig for q in r["per_obl"])
        else:
            bad = False
        if bad:
            drop[t] += 1
        else:
            keep.append(r)
    n_mut = sum(1 for r in recs if r["type"] == "mutant")
    print(f"{p.stem:<28} rerun: {drop['orig']:>3} orig, {drop['control']:>3} control, "
          f"{drop['mutant']:>3} of {n_mut} mutants")
    for k in tot: tot[k] += drop[k]
    if fix and any(drop.values()):
        shutil.copy(p, str(p) + ".before_repair")
        with p.open("w") as f:
            for r in keep: f.write(json.dumps(r) + "\n")
print(f"TOTAL to rerun: {tot['orig']} orig, {tot['control']} control, {tot['mutant']} mutants")
if fix:
    for s in d.glob("run_stats*.json"):
        s.rename(s.with_suffix(".json.old"))
    print("Fixed. Backups: *.jsonl.before_repair. Now resume the sweep.")
