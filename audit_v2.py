#!/usr/bin/env python3
"""Manual-audit helper for the revision (Section IV-J of the paper).

Step 1  export labeling sheets from the v2 logs
    python3 audit_v2.py export LOG_DIR --per-op 30 --verdicts 200 --out audit/
  -> audit/mutants.csv   stratified sample of obligation-preserving mutants
                         (Tool-Fault, Ag-Swap, Ag-PromptDrop) with original vs
                         mutant traces; two raters fill rater1/rater2 with
                         'eq' (behaviourally equivalent) or 'neq', and 'final'
                         after resolving disagreements.
  -> audit/verdicts.csv  sample of final responses, half predicted 'fail' and
                         half 'ok' by the keyword classifier; raters fill
                         rater1/rater2 with 'fail' (refusal/inability) or 'ok'
                         (task completion), and 'final'.

Step 2  score the filled sheets
    python3 audit_v2.py score LOG_DIR --audit audit/
  -> Cohen's kappa, % equivalent per operator, K-Outcome MS with the
     equivalent mutants excluded, verdict precision/recall/kappa.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_v2 import load, witnesses, CRITERIA, BASE_SUITES  # noqa: E402
from run_live_mutation import _refused  # noqa: E402

AUDIT_OPS = ["Tool-Fault", "Ag-Swap", "Ag-PromptDrop"]
SEED = 20260605


def _fmt_run(r):
    if r is None:
        return "UNSCORED"
    v = "FAIL" if (r.get("max_turns") or _refused(r["text"])) else "ok"
    return (f"[{v}] tools={r['tools']} deleg={r['deleg']} denied={r['denied']} "
            f"text={r['text'][:300]!r}")


def export(a):
    benches = load(Path(a.log_dir), set())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # --- mutant sheet -------------------------------------------------------
    pool = defaultdict(list)
    for b, d in benches.items():
        orig = {}
        for line in (Path(a.log_dir) / f"{b}.jsonl").open():
            import json
            r = json.loads(line)
            if r["type"] == "orig":
                orig[tuple(r["obl"])] = r["runs"]
        for m in d["mutants"]:
            if m["operator"] in AUDIT_OPS:
                pool[m["operator"]].append((b, m, orig))
    rows = []
    for op in AUDIT_OPS:
        items = pool[op]
        for b, m, orig in rng.sample(items, min(a.per_op, len(items))):
            killed = [K for K in CRITERIA[:2] if witnesses(m, K)]
            fired = [p for p in m["per_obl"] if p["fires"].get("K-Struct") or p["fires"].get("K-Outcome")]
            for p in (fired or m["per_obl"][:1])[:3]:
                o = tuple(p["obl"])
                rows.append({
                    "mutant_id": f"{b}:{m['idx']}", "operator": op, "change": m["target"],
                    "killed_under": "/".join(killed) or "none",
                    "obligation": "::".join(o),
                    "fires_S/O": f"{p['fires'].get('K-Struct')}/{p['fires'].get('K-Outcome')}",
                    "original_runs": "\n".join(_fmt_run(r) for r in orig.get(o, [])),
                    "mutant_runs": "\n".join(_fmt_run(r) for r in p.get("runs", [])),
                    "rater1": "", "rater2": "", "final": "", "notes": ""})
    with (out / "mutants.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n_mut = len({r["mutant_id"] for r in rows})
    print(f"mutants.csv: {n_mut} mutants, {len(rows)} rows (label each mutant once, "
          f"on its first row; other rows give context)")

    # --- verdict sheet ------------------------------------------------------
    texts = []
    for b, d in benches.items():
        for m in d["mutants"]:
            for p in m["per_obl"]:
                for r in p.get("runs", []):
                    if r and not r.get("max_turns"):
                        texts.append(r["text"])
    texts = sorted(set(texts))
    fail = [t for t in texts if _refused(t)]
    ok = [t for t in texts if not _refused(t)]
    half = a.verdicts // 2
    sample = [(t, "fail") for t in rng.sample(fail, min(half, len(fail)))] + \
             [(t, "ok") for t in rng.sample(ok, min(half, len(ok)))]
    rng.shuffle(sample)
    with (out / "verdicts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "response", "rater1", "rater2", "final", "_predicted_hidden"])
        for i, (t, pred) in enumerate(sample):
            w.writerow([i, t, "", "", "", pred])
    print(f"verdicts.csv: {len(sample)} responses (hide the last column from raters)")


def kappa(pairs):
    pairs = [(x, y) for x, y in pairs if x and y]
    n = len(pairs)
    if not n:
        return float("nan"), 0
    po = sum(x == y for x, y in pairs) / n
    c1, c2 = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), n


def score(a):
    benches = load(Path(a.log_dir), set())
    ad = Path(a.audit)

    # --- mutant audit -------------------------------------------------------
    first = {}
    for r in csv.DictReader((ad / "mutants.csv").open()):
        if r["mutant_id"] not in first and (r["rater1"] or r["final"]):
            first[r["mutant_id"]] = r
    norm = lambda s: s.strip().lower()  # noqa: E731
    k, n = kappa([(norm(r["rater1"]), norm(r["rater2"])) for r in first.values()])
    final = {mid: norm(r["final"] or (r["rater1"] if norm(r["rater1"]) == norm(r["rater2"]) else ""))
             for mid, r in first.items()}
    unresolved = [m for m, v in final.items() if v not in ("eq", "neq")]
    print(f"=== Equivalence audit: {len(first)} mutants labelled, Cohen's kappa = {k:.2f} (n={n})")
    if unresolved:
        print(f"  WARNING {len(unresolved)} mutants without a 'final' label: {unresolved[:5]}")
    by_op = defaultdict(Counter)
    for mid, r in first.items():
        by_op[r["operator"]][final[mid]] += 1
    tot_eq = sum(c["eq"] for c in by_op.values()); tot = sum(sum(c.values()) for c in by_op.values())
    for op, c in by_op.items():
        s = c["eq"] + c["neq"]
        print(f"  {op:<14} equivalent {c['eq']}/{s} = {c['eq']/s if s else float('nan'):.1%}")
    print(f"  overall        equivalent {tot_eq}/{tot} = {tot_eq/tot if tot else float('nan'):.1%}")

    eq_ids = {m for m, v in final.items() if v == "eq"}
    print("\n  MS with audited-equivalent mutants excluded (K-Struct / K-Outcome):")
    worst = 0.0
    for s in BASE_SUITES:
        parts = []
        for K in CRITERIA[:2]:
            vals = []
            for excl in (set(), eq_ids):
                kd = kb = 0
                for b, d in benches.items():
                    S = set(d["meta"]["suites"][s])
                    for m in d["mutants"]:
                        if f"{b}:{m['idx']}" in excl:
                            continue
                        w = witnesses(m, K)
                        if w:
                            kb += 1; kd += bool(w & S)
                vals.append(kd / kb if kb else float("nan"))
            worst = max(worst, abs(vals[1] - vals[0]))
            parts.append(f"{K} {vals[0]:.3f} -> {vals[1]:.3f}")
        print(f"    {s:<9} " + " | ".join(parts))
    print(f"  max absolute change: {worst:.3f}")

    # --- verdict validation -------------------------------------------------
    vp = ad / "verdicts.csv"
    if vp.exists():
        rows = [r for r in csv.DictReader(vp.open()) if r["rater1"] or r["final"]]
        k2, n2 = kappa([(norm(r["rater1"]), norm(r["rater2"])) for r in rows])
        tp = fp = fn = tn = 0
        for r in rows:
            gold = norm(r["final"] or (r["rater1"] if norm(r["rater1"]) == norm(r["rater2"]) else ""))
            if gold not in ("fail", "ok"):
                continue
            pred = r["_predicted_hidden"]
            tp += pred == "fail" and gold == "fail"; fp += pred == "fail" and gold == "ok"
            fn += pred == "ok" and gold == "fail"; tn += pred == "ok" and gold == "ok"
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n=== Verdict classifier: {tp+fp+fn+tn} responses, rater kappa = {k2:.2f}")
        print(f"  precision {prec:.3f}  recall {rec:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")
        print("  note: the sample is balanced on the prediction, so precision is unbiased "
              "but recall is over-estimated; report both with this caveat or weight by class size.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("log_dir"); e.add_argument("--out", default="audit")
    e.add_argument("--per-op", type=int, default=30); e.add_argument("--verdicts", type=int, default=200)
    s = sp.add_parser("score"); s.add_argument("log_dir"); s.add_argument("--audit", default="audit")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else score(a)


if __name__ == "__main__":
    main()
