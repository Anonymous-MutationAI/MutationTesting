#!/usr/bin/env python3
"""How much each scoring rule changes the results (no API calls, stdlib only).

    python3 sensitivity_v2.py [results/v2]

Starting from the first-submission (legacy) scoring, applies each rule of
src/agentmut/rules.py on its own and then all together (the revised scoring
used by analyze_v2.py), and reports:

  R  test-retest reliability: runtime-duplicate mutants are the same mutant
     executed twice with independent runs, so their agreement measures how
     reproducible a single kill decision is.
  V  MS of the three baselines under every criterion for each rule set,
     with the first control replicate's noise adjustment as an extra row.
  O  per-operator K-Struct MS, legacy versus revised.
  S  size-matched random subsets of T-Struct, legacy versus revised.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analyze_v2 import BASE_SUITES, CRITERIA, OPS, load  # noqa: E402
from agentmut import generate_mutants  # noqa: E402
from agentmut.rules import drop_runtime_duplicates, majority, rescored_votes, runtime_signature  # noqa: E402

SEED = 20260605
BASELINES = BASE_SUITES[:3]


def split_duplicates(B):
    """bench -> (kept mutants, {sig-representative idx: [duplicate mutants]})."""
    kept, groups = {}, {}
    for b, d in B.items():
        by_key = {(mu.operator, str(mu.target)): mu
                  for mu in generate_mutants(json.load((ROOT / "benchmarks" / f"{b}.json").open()), b)}
        items = [(m["idx"], by_key[(m["operator"], m["target"])]) for m in d["mutants"]]
        k, _ = drop_runtime_duplicates(items)
        keep_idx = {i for i, _ in k}
        kept[b] = [m for m in d["mutants"] if m["idx"] in keep_idx]
        sig = {i: runtime_signature(mu.manifest) for i, mu in items}
        g = {}
        for m in d["mutants"]:
            g.setdefault(sig[m["idx"]], []).append(m)
        groups[b] = [v for v in g.values() if len(v) > 1]
    return kept, groups


def witness_fn(executed_only=False, reach=False, adjusted=False):
    def w(m, K, control):
        out = set()
        for p in m["per_obl"]:
            o = tuple(p["obl"])
            if "structural_fail" in p:
                if not executed_only and p["fires"].get(K):
                    out.add(o)
                continue
            v = rescored_votes(m, p)[K] if reach else p["votes"][K]
            if not (v and majority(v)):
                continue
            if adjusted:
                c = control.get(o)
                if c is None or c["fires"].get(K):
                    continue
            out.add(o)
        return out
    return w


def rows_for(B, pool, wfn, K):
    rows = []
    for b, d in B.items():
        S = {s: set(d["meta"]["suites"][s]) for s in BASE_SUITES}
        for m in pool[b]:
            w = wfn(m, K, d["control"])
            if w:
                rows.append((b, m["operator"], {s: bool(w & S[s]) for s in BASE_SUITES}))
    return rows


def ms(rows, s):
    return sum(r[s] for *_, r in rows) / len(rows) if rows else float("nan")


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "v2"
    B = load(log_dir, set(), legacy=True)
    full = {b: d["mutants"] for b, d in B.items()}
    kept, groups = split_duplicates(B)

    # ---- R reliability ---------------------------------------------------------
    print("=== R: runtime-duplicate mutants as a test-retest check (legacy logs) ===")
    n_dup = sum(len(g) - 1 for gs in groups.values() for g in gs)
    ops = Counter(m["operator"] for gs in groups.values() for g in gs for m in g)
    print(f"  {n_dup} redundant mutants in {sum(len(gs) for gs in groups.values())} groups; "
          f"operators involved: {dict(ops)}")
    w = witness_fn()
    for K in ("K-Struct", "K-Outcome"):
        agree = n = oa = on = 0
        for b, gs in groups.items():
            for g in gs:
                for x, y in combinations(g, 2):
                    wx, wy = w(x, K, None), w(y, K, None)
                    for s in BASELINES:
                        S = set(B[b]["meta"]["suites"][s])
                        agree += bool(wx & S) == bool(wy & S)
                        n += 1
                    fx = {tuple(p["obl"]): majority(p["votes"][K]) for p in x["per_obl"] if "votes" in p}
                    fy = {tuple(p["obl"]): majority(p["votes"][K]) for p in y["per_obl"] if "votes" in p}
                    for o in fx.keys() & fy.keys():
                        oa += fx[o] == fy[o]
                        on += 1
        print(f"  {K:<10} same baseline kill decision {agree}/{n} = {agree / n:.3f}; "
              f"same decision per shared obligation {oa}/{on} = {oa / on:.3f}")

    # ---- V variants -----------------------------------------------------------
    variants = [
        ("legacy (first submission)", full, witness_fn()),
        ("+ duplicates removed", kept, witness_fn()),
        ("+ executed witnesses only", full, witness_fn(executed_only=True)),
        ("+ Tool-Fault reachability", full, witness_fn(reach=True)),
        ("revised (all three rules)", kept, witness_fn(True, True)),
        ("revised + first-replicate adj.", kept, witness_fn(True, True, True)),
    ]
    print("\n=== V: MS of the baselines (killable mutants per criterion in n) ===")
    print(f"{'rules':<32}{'n':>14}" + "".join(f"{K[2:] + ' ' + s.rstrip('*')[2:]:>14}"
                                             for K in CRITERIA for s in BASELINES))
    cache = {}
    for name, pool, wfn in variants:
        cells, ns = [], []
        for K in CRITERIA:
            rows = rows_for(B, pool, wfn, K)
            cache[(name, K)] = rows
            ns.append(len(rows))
            cells += [f"{ms(rows, s):>14.3f}" for s in BASELINES]
        print(f"{name:<32}{'/'.join(map(str, ns)):>14}" + "".join(cells))

    # ---- O per operator -------------------------------------------------------
    print("\n=== O: per-operator K-Struct MS, legacy -> revised (T-Dev* / T-Rand* / T-E2E*) ===")
    for op in OPS:
        cell = []
        for name in ("legacy (first submission)", "revised (all three rules)"):
            rows = [r for r in cache[(name, "K-Struct")] if r[1] == op]
            cell.append(" / ".join(f"{ms(rows, s):.2f}" for s in BASELINES) + f" (n={len(rows)})")
        print(f"  {op:<14} {cell[0]:<28} -> {cell[1]}")

    # ---- S size-matched ------------------------------------------------------
    print("\n=== S: size-matched random subsets of T-Struct (|T-Dev*| per benchmark, 100 draws each) ===")
    sizes = {b: len(d["meta"]["suites"]["T-Dev*"]) for b, d in B.items()}
    for name, pool, wfn in variants:
        if name not in ("legacy (first submission)", "revised (all three rules)",
                        "revised + first-replicate adj."):
            continue
        for K in CRITERIA:
            rng = random.Random(SEED)
            wit = [(b, x) for b in B for x in (wfn(m, K, B[b]["control"]) for m in pool[b]) if x]
            draws = []
            for _ in range(100):
                pick = {b: set(rng.sample(sorted(d["meta"]["suites"]["T-Struct"]), sizes[b]))
                        for b, d in B.items()}
                draws.append(sum(bool(x & pick[b]) for b, x in wit) / len(wit))
            draws.sort()
            dev = ms(cache[(name, K)], "T-Dev*")
            print(f"  [{name}] {K:<10} T-Dev* {dev:.3f} | random mean {sum(draws) / 100:.3f} "
                  f"[{draws[2]:.3f}, {draws[97]:.3f}] | draws below T-Dev*: {sum(x < dev for x in draws)}/100")


if __name__ == "__main__":
    main()
