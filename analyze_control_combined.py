#!/usr/bin/env python3
"""Noise analysis using all three control replicates (no API calls, stdlib only).

    python3 analyze_control_combined.py [--legacy] [--results results/v2] [--control results/v2_control]

Replicate 1 is the original-versus-original control logged with the main sweep
(results/v2); replicates 2 and 3 come from run_control_extra.py
(results/v2_control). An obligation counts as noisy under a criterion if ANY
replicate fires on it; noise-adjusted scores let only non-noisy obligations
witness a kill (structural failures are never kills under the revised rules).

Mutants are loaded with the pool and kill rules of src/agentmut/rules.py;
--legacy reproduces the numbers of the first submission.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analyze_v2 import coverage_greedy, load, wilson, witnesses  # noqa: E402

CR = ["K-Struct", "K-Outcome", "K-Safety"]
SU = ["T-Dev*", "T-Rand*", "T-E2E*", "T-Struct"]
SEED = 20260605


def load_replicates(B, control_dir):
    """bench -> obligation -> [fires of replicate 1, 2, 3]."""
    reps = defaultdict(lambda: defaultdict(list))
    for b, d in B.items():
        for o, c in d["control"].items():
            reps[b][o].append(c["fires"])
        for line in open(control_dir / f"{b}.jsonl"):
            r = json.loads(line)
            reps[b][tuple(r["obl"])].append(r["fires"])
    return reps


def union_control(reps, upto=3):
    return {b: {o: {"fires": {K: any(f.get(K) for f in fl[:upto]) for K in CR}} for o, fl in ob.items()}
            for b, ob in reps.items()}


def single_control(reps, i):
    return {b: {o: {"fires": {K: bool(fl[i].get(K)) for K in CR}} for o, fl in ob.items() if len(fl) > i}
            for b, ob in reps.items()}


def score(B, control, sof, K, adjusted, ops=None):
    kd = kb = 0
    for b, d in B.items():
        S = sof(b)
        for m in d["mutants"]:
            if ops and m["operator"] not in ops:
                continue
            w = witnesses(m, K, control[b], adjusted)
            if w:
                kb += 1
                kd += bool(w & S)
    return kd, kb


def exposure(B, control, s, K):
    """Share of scored mutants with a noisy incident obligation in suite s."""
    n = f = 0
    for b, d in B.items():
        S = set(d["meta"]["suites"][s])
        for m in d["mutants"]:
            inc = {tuple(p["obl"]) for p in m["per_obl"] if "structural_fail" not in p}
            n += 1
            f += any(control[b].get(o, {}).get("fires", {}).get(K) for o in inc & S)
    return f / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--control", default=str(ROOT / "results" / "v2_control"))
    ap.add_argument("--legacy", action="store_true", help="first-submission scoring rules")
    a = ap.parse_args()
    B = load(Path(a.results), set(), legacy=a.legacy)
    reps = load_replicates(B, Path(a.control))
    comb = union_control(reps)
    suite = lambda s: (lambda b: set(B[b]["meta"]["suites"][s]))  # noqa: E731
    print(f"rules: {'legacy (first submission)' if a.legacy else 'revised (agentmut.rules)'}; "
          f"scored mutants: {sum(len(d['mutants']) for d in B.values())}")

    print("\n=== per-obligation false-kill rate: each replicate, and union of 3")
    for K in CR:
        per = []
        for i in range(3):
            f = n = 0
            for b in reps:
                for fl in reps[b].values():
                    if len(fl) > i and fl[i].get(K) is not None:
                        n += 1
                        f += bool(fl[i][K])
            per.append(f"rep{i + 1} {f}/{n}={f / n:.3f}")
        u = sum(1 for b in comb for c in comb[b].values() if c["fires"][K])
        N = sum(len(comb[b]) for b in comb)
        lo, hi = wilson(u, N)
        print(f"{K:9}", " | ".join(per), f"| union {u}/{N}={u / N:.3f} [{lo:.3f},{hi:.3f}]")

    print("\n=== mutant-level noise exposure (control fires on an incident obligation in the suite), K-Struct")
    for label, ctl in [(f"rep{i + 1}", single_control(reps, i)) for i in range(3)] + [("union", comb)]:
        print(f"  {label:<6}" + " | ".join(f"{s} {exposure(B, ctl, s, 'K-Struct'):.3f}" for s in SU))
    for K in ("K-Outcome", "K-Safety"):
        print(f"  union {K}: " + " | ".join(f"{s} {exposure(B, comb, s, K):.3f}" for s in SU))

    print("\n=== MS raw / adjusted (union control)")
    for s in SU:
        parts = []
        for K in CR:
            r_ = score(B, comb, suite(s), K, False)
            j = score(B, comb, suite(s), K, True)
            parts.append(f"{K} {r_[0] / r_[1]:.3f}/{j[0] / j[1]:.3f} (n={j[1]})")
        print(f"{s:9}", " | ".join(parts))

    print("\n=== Res-Rem MS adjusted (union control), K-Struct / K-Outcome")
    for s in SU[:3]:
        print(f"  {s:<8} " + " | ".join(
            f"{K} {x[0] / x[1]:.3f} (n={x[1]})"
            for K, x in ((K, score(B, comb, suite(s), K, True, {"Res-Rem"})) for K in CR[:2])))

    print("\n=== size-matched (adjusted, union control)")
    rng = random.Random(SEED)
    for K in CR:
        dev = score(B, comb, suite("T-Dev*"), K, True)
        dr = []
        for _ in range(100):
            pick = {b: set(rng.sample(sorted(d["meta"]["suites"]["T-Struct"]), len(d["meta"]["suites"]["T-Dev*"])))
                    for b, d in B.items()}
            x = score(B, comb, lambda b: pick[b], K, True)
            dr.append(x[0] / x[1])
        dr.sort()
        g = {b: coverage_greedy(d["meta"]["suites"]["T-Struct"], len(d["meta"]["suites"]["T-Dev*"]))
             for b, d in B.items()}
        gr = score(B, comb, lambda b: g[b], K, True)
        print(f"{K:9} T-Dev* {dev[0] / dev[1]:.3f} | random {sum(dr) / 100:.3f} [{dr[2]:.3f},{dr[97]:.3f}] "
              f"| greedy {gr[0] / gr[1]:.3f}")


if __name__ == "__main__":
    main()
