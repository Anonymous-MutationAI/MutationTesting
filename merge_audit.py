#!/usr/bin/env python3
"""Combine the two raters' sheets into audit/mutants.csv and audit/verdicts.csv.

  python merge_audit.py            # merge + write audit/disagreements.csv
Fill the 'final' column of audit/disagreements.csv after discussing, then run
it again; the agreed and resolved labels go into the 'final' columns, ready for
  python audit_v2.py score results/v2 --audit audit/
"""
import csv, os
A = "audit"
def norm(s): return (s or "").strip().lower()
def labels(path, key, col):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r[key] in out:          # only the first row of each item counts
            continue
        out[r[key]] = "" if (r[col] or "").startswith("(") else norm(r[col])
    return {k: v for k, v in out.items() if v}
resolved = {}
dp = os.path.join(A, "disagreements.csv")
if os.path.exists(dp):
    for r in csv.DictReader(open(dp)):
        if norm(r["final"]): resolved[(r["sheet"], r["id"])] = norm(r["final"])
dis, missing = [], []
for sheet, master, key, col, f1, f2, ok in [
    ("mutants", "mutants.csv", "mutant_id", "LABEL (eq/neq)", "rater1_mutants.csv", "rater2_mutants.csv", ("eq", "neq")),
    ("responses", "verdicts.csv", "id", "LABEL (ok/fail)", "rater1_responses.csv", "rater2_responses.csv", ("ok", "fail"))]:
    l1, l2 = labels(os.path.join(A, f1), key, col), labels(os.path.join(A, f2), key, col)
    rows = list(csv.DictReader(open(os.path.join(A, master))))
    seen = set()
    for r in rows:
        k = r[key]
        if sheet == "mutants" and k in seen:
            continue
        seen.add(k)
        a, b = l1.get(k, ""), l2.get(k, "")
        for x, who in ((a, "rater1"), (b, "rater2")):
            if x and x not in ok: missing.append(f"{sheet} {k}: {who} wrote '{x}' (use {'/'.join(ok)})")
            if not x: missing.append(f"{sheet} {k}: {who} label missing")
        r["rater1"], r["rater2"] = a, b
        r["final"] = a if a and a == b else resolved.get((sheet, k), "")
        if a and b and a != b:
            ctx = r.get("change") or r.get("response", "")[:300]
            dis.append({"sheet": sheet, "id": k, "item": ctx, "rater1": a, "rater2": b, "final": resolved.get((sheet, k), "")})
    w = csv.DictWriter(open(os.path.join(A, master), "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
w = csv.DictWriter(open(dp, "w", newline=""), fieldnames=["sheet", "id", "item", "rater1", "rater2", "final"]); w.writeheader(); w.writerows(dis)
open_d = sum(1 for d in dis if not d["final"])
print(f"{len(dis)} disagreements ({open_d} still to resolve in {dp}); {len(missing)} missing/invalid labels")
for m in missing[:10]: print("  ", m)
if not open_d and not missing: print("Ready: python audit_v2.py score results/v2 --audit audit/")
