#!/usr/bin/env python3
"""Offline analysis of run_live_mutation_v2.py logs (no API calls).

Produces every number the revision needs:
  T1  executed pool + live-killable denominators per operator   (Table III)
  T2  K-Struct MS per family per suite                           (Table IV)
  T3  MS per suite per criterion, raw AND noise-adjusted         (Table V)
  T4  K-Outcome MS per operator                                  (Table VI)
  C   control false-kill rate per criterion                      (new Sec. flaky)
  V   per-obligation vote distribution (0..k of k pairs differ)  (new Sec. flaky)
  S   size-matched random subsets of T-Struct + coverage-greedy  (RQ1, reviewer A)
  K   k=3 vs k agreement when logs were produced with k>3        (new Sec. flaky)
  E   error / skip statistics                                    (threats)

Noise-adjusted scoring: an obligation may witness a kill under K only if the
control did NOT fire on that obligation under K (the original is stable there).

Usage: python3 analyze_v2.py LOG_DIR [--exclude-op Tool-SigMut] [--draws 100] [--csv OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from agentmut import OPERATOR_FAMILIES  # noqa: E402

CRITERIA = ["K-Struct", "K-Outcome", "K-Safety"]
BASE_SUITES = ["T-Dev*", "T-Rand*", "T-E2E*", "T-Struct"]
FAM = {op: fam.split("_")[0] for fam, ops in OPERATOR_FAMILIES.items() for op in ops}
OPS = [op for ops in OPERATOR_FAMILIES.values() for op in ops]


def load(log_dir: Path, exclude: set):
    benches = {}
    for p in sorted(log_dir.glob("*.jsonl")):
        meta, control, mutants = None, {}, []
        for line in p.open():
            r = json.loads(line)
            if r["type"] == "meta":
                meta = r
            elif r["type"] == "control":
                control[tuple(r["obl"])] = r
            elif r["type"] == "mutant" and r["operator"] not in exclude:
                mutants.append(r)
        if meta:
            meta["suites"] = {s: [tuple(o) for o in v] for s, v in meta["suites"].items()}
            benches[p.stem] = {"meta": meta, "control": control, "mutants": mutants}
    return benches


def witnesses(mut, K, control=None, adjusted=False):
    out = set()
    for p in mut["per_obl"]:
        o = tuple(p["obl"])
        if not p["fires"].get(K):
            continue
        if adjusted and "structural_fail" not in p:
            c = control.get(o)
            if c is None or c["fires"].get(K):  # unstable or unmeasured original
                continue
        out.add(o)
    return out


def score(benches, suite_of, K, adjusted=False, ops=None):
    """suite_of(bench) -> set of obligations. Returns killed, killable."""
    kd = kb = 0
    for b, d in benches.items():
        S = suite_of(b)
        for m in d["mutants"]:
            if ops and m["operator"] not in ops:
                continue
            w = witnesses(m, K, d["control"], adjusted)
            if w:
                kb += 1
                kd += bool(w & S)
    return kd, kb


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fmt(kd, kb):
    return f"{kd/kb:.3f}" if kb else " n/a "


def elements(o):
    if o[0] == "reach":
        return {("A", o[1])}
    if o[0] == "use_tool":
        return {("A", o[1]), ("T", o[2]), ("U", o[1], o[2])}
    if o[0] == "restrict_tool":
        return {("A", o[1]), ("T", o[2]), ("R", o[1], o[2])}
    if o[0] == "delegate":
        return {("A", o[1]), ("A", o[2]), ("D", o[1], o[2])}
    return set()


def coverage_greedy(obls, n):
    """A-priori coverage-driven selection: repeatedly take the obligation that
    covers the most not-yet-covered graph elements (no kill information)."""
    chosen, covered, pool = [], set(), sorted(obls)
    while pool and len(chosen) < n:
        best = max(pool, key=lambda o: (len(elements(o) - covered), o[0] == "restrict_tool"))
        chosen.append(best); covered |= elements(best); pool.remove(best)
    return set(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--exclude-op", nargs="*", default=[])
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--match", default="T-Dev*", help="baseline whose size to match")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    benches = load(Path(a.log_dir), set(a.exclude_op))
    if not benches:
        sys.exit("no logs found")
    rows = []
    suite = lambda s: (lambda b: set(benches[b]["meta"]["suites"][s]))  # noqa: E731

    # ---- T1 inventory ------------------------------------------------------
    print("=== T1: executed pool and live-killable denominators (raw | noise-adjusted) ===")
    print(f"{'Operator':<15}{'Fam':<4}{'Total':>6}" + "".join(f"{K:>18}" for K in CRITERIA))
    tot = Counter()
    for op in OPS:
        n = sum(1 for d in benches.values() for m in d["mutants"] if m["operator"] == op)
        if not n:
            continue
        cells = []
        for K in CRITERIA:
            r = sum(1 for d in benches.values() for m in d["mutants"]
                    if m["operator"] == op and witnesses(m, K))
            j = sum(1 for d in benches.values() for m in d["mutants"]
                    if m["operator"] == op and witnesses(m, K, d["control"], True))
            tot[K] += r; tot[K + "adj"] += j
            cells.append(f"{r:>9} | {j:<6}")
        tot["n"] += n
        print(f"{op:<15}{FAM[op]:<4}{n:>6}" + "".join(f"{c:>18}" for c in cells))
    print(f"{'TOTAL':<19}{tot['n']:>6}" + "".join(f"{tot[K]:>9} | {tot[K+'adj']:<6}" for K in CRITERIA))

    # ---- T2 family table ---------------------------------------------------
    print("\n=== T2: K-Struct MS per family ===")
    fams = ["F1", "F2", "F3", "F4"]
    print(f"{'Suite':<10}" + "".join(f"{f:>8}" for f in fams) + f"{'Overall':>9}")
    for s in BASE_SUITES:
        cells = [fmt(*score(benches, suite(s), "K-Struct",
                            ops={o for o in OPS if FAM[o] == f})) for f in fams]
        print(f"{s:<10}" + "".join(f"{c:>8}" for c in cells)
              + f"{fmt(*score(benches, suite(s), 'K-Struct')):>9}")

    # ---- T3 criteria table, raw and adjusted --------------------------------
    print("\n=== T3: MS per suite per criterion (raw / noise-adjusted), 95% Wilson CI on raw ===")
    for s in BASE_SUITES:
        parts = []
        for K in CRITERIA:
            kd, kb = score(benches, suite(s), K)
            jd, jb = score(benches, suite(s), K, adjusted=True)
            lo, hi = wilson(kd, kb)
            parts.append(f"{K}: {fmt(kd,kb)} [{lo:.2f},{hi:.2f}] / {fmt(jd,jb)}")
            rows.append({"table": "T3", "suite": s, "criterion": K, "killed": kd,
                         "killable": kb, "ms": kd / kb if kb else None,
                         "killed_adj": jd, "killable_adj": jb, "ms_adj": jd / jb if jb else None})
        print(f"  {s:<9} " + " | ".join(parts))

    # ---- T4 per-operator K-Outcome ------------------------------------------
    print("\n=== T4: K-Outcome MS per operator (raw / noise-adjusted) ===")
    print(f"{'Operator':<15}" + "".join(f"{s:>16}" for s in BASE_SUITES))
    for op in OPS:
        if any(m["operator"] == op for d in benches.values() for m in d["mutants"]):
            print(f"{op:<15}" + "".join(
                f"{fmt(*score(benches, suite(s), 'K-Outcome', ops={op})) + ' / ' + fmt(*score(benches, suite(s), 'K-Outcome', adjusted=True, ops={op})):>16}"
                for s in BASE_SUITES))

    # ---- C control false-kill rate ------------------------------------------
    print("\n=== C: original-vs-original control (false-kill rate per obligation) ===")
    for K in CRITERIA:
        f = n = 0
        for d in benches.values():
            for c in d["control"].values():
                v = c["fires"].get(K)
                if v is not None:
                    n += 1; f += bool(v)
        lo, hi = wilson(f, n)
        print(f"  {K:<10} {f}/{n} = {f/n if n else float('nan'):.3f}  [{lo:.3f},{hi:.3f}]")
        rows.append({"table": "C", "criterion": K, "fires": f, "n": n})
    by_kind = defaultdict(lambda: [0, 0])
    for d in benches.values():
        for o, c in d["control"].items():
            v = c["fires"].get("K-Struct")
            if v is not None:
                by_kind[o[0]][0] += bool(v); by_kind[o[0]][1] += 1
    print("  K-Struct by obligation kind: " + ", ".join(
        f"{k} {f}/{n}" for k, (f, n) in sorted(by_kind.items())))

    # ---- V vote distribution ------------------------------------------------
    print("\n=== V: distribution of differing pairs per (mutant, obligation) ===")
    for K in CRITERIA:
        dist = Counter()
        for d in benches.values():
            for m in d["mutants"]:
                for p in m["per_obl"]:
                    if "votes" in p and p["votes"][K]:
                        dist[(sum(p["votes"][K]), len(p["votes"][K]))] += 1
        print(f"  {K:<10} " + "  ".join(f"{x}/{n}:{c}" for (x, n), c in sorted(dist.items())))

    # ---- S size-matched suites ----------------------------------------------
    print(f"\n=== S: size-matched subsets of T-Struct (size = |{a.match}| per benchmark) ===")
    sizes = {b: len(d["meta"]["suites"][a.match]) for b, d in benches.items()}
    rng = random.Random(20260605)
    for K in CRITERIA:
        base = score(benches, suite(a.match), K)
        draws = []
        for _ in range(a.draws):
            pick = {b: set(rng.sample(sorted(d["meta"]["suites"]["T-Struct"]),
                                      min(sizes[b], len(d["meta"]["suites"]["T-Struct"]))))
                    for b, d in benches.items()}
            kd, kb = score(benches, lambda b: pick[b], K)
            if kb:
                draws.append(kd / kb)
        draws.sort()
        greedy = {b: coverage_greedy(d["meta"]["suites"]["T-Struct"], sizes[b])
                  for b, d in benches.items()}
        g = score(benches, lambda b: greedy[b], K)
        if draws:
            lo, hi = draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]
            mean = sum(draws) / len(draws)
            print(f"  {K:<10} {a.match} {fmt(*base)} | random mean {mean:.3f} [{lo:.3f},{hi:.3f}] "
                  f"| coverage-greedy {fmt(*g)}")
            rows.append({"table": "S", "criterion": K, "baseline": base[0] / base[1] if base[1] else None,
                         "rand_mean": mean, "rand_lo": lo, "rand_hi": hi,
                         "greedy": g[0] / g[1] if g[1] else None})
    print("  sizes: " + ", ".join(f"{b}={n}" for b, n in sizes.items()))

    # ---- K k=3 vs full-k agreement -----------------------------------------
    ks = {d["meta"]["k"] for d in benches.values()}
    if max(ks) > 3:
        print("\n=== K: majority over first 3 runs vs over all k runs ===")
        for K in CRITERIA:
            agree = n = 0
            for d in benches.values():
                for m in d["mutants"]:
                    for p in m["per_obl"]:
                        v = p.get("votes", {}).get(K, [])
                        if len(v) >= 5:
                            n += 1
                            agree += ((sum(v[:3]) * 2 > 3) == (sum(v) * 2 > len(v)))
            print(f"  {K:<10} agreement {agree}/{n} = {agree/n if n else float('nan'):.3f}")

    # ---- E stats -----------------------------------------------------------
    sp = Path(a.log_dir) / "run_stats.json"
    if sp.exists():
        print("\n=== E: run statistics ===")
        print("  " + json.dumps(json.load(sp.open())))
    mt = sum(1 for d in benches.values() for m in d["mutants"] for p in m["per_obl"]
             for r in p.get("runs", []) if r and r.get("max_turns"))
    print(f"  mutant runs hitting max_turns: {mt}")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\n  -> {a.csv}")


if __name__ == "__main__":
    main()
