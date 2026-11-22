#!/usr/bin/env python3
"""Check the numbers reported in the paper against the logged runs.

    python3 verify_paper.py                 # results/v2, audit/, paper_values.json
    python3 verify_paper.py --results DIR   # e.g. a fresh live re-run

paper_values.json holds every number the paper reports, grouped by table or
section. This script recomputes each one from benchmarks/, the logs and the
audit sheets (using the pool and kill rules in src/agentmut/rules.py), prints
PASS or FAIL per value, and exits with status 1 if any check fails. It needs
only the Python standard library and makes no API calls. Values that depend on
the two extra noise-control replicates are skipped while results/v2_control/
is absent.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analyze_v2 import (BASE_SUITES, CRITERIA, FAM, OPS, cluster_boot,  # noqa: E402
                        cochran_q, coverage_greedy, kill_rows, load, ms_rows,
                        witnesses, wilson)
from agentmut.rules import rescored_votes  # noqa: E402

SEED = 20260605
BASELINES = BASE_SUITES[:3]
PAIRS = [("T-Dev*", "T-Rand*"), ("T-Dev*", "T-E2E*"), ("T-E2E*", "T-Rand*")]


def r(x, nd=3):
    return None if x is None or x != x else round(x, nd)


def ms(B, s, K, ops=None):
    kd = kb = 0
    for d in B.values():
        S = set(d["meta"]["suites"][s])
        for m in d["mutants"]:
            if ops and m["operator"] not in ops:
                continue
            w = witnesses(m, K)
            if w:
                kb += 1
                kd += bool(w & S)
    return kd / kb if kb else None


# ----------------------------------------------------------------------------
# Computations (one per group in paper_values.json)
# ----------------------------------------------------------------------------
def compute_inventory():
    from agentcov import extract_objectives, graph_from_dict
    from agentmut import build_suites, cap_per_operator, generate_mutants
    from agentmut.rules import drop_runtime_duplicates, well_formedness_violations
    spec = importlib.util.spec_from_file_location("v2", ROOT / "run_live_mutation_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    out = {"table2": {}, "generated": 0, "after_cap": 0, "placeholder_skipped": 0,
           "ill_formed": 0, "runtime_duplicates": 0, "scored": 0, "table3": {}}
    dup_ops = Counter()
    for p in sorted((ROOT / "benchmarks").glob("*.json")):
        b, m = p.stem, json.load(p.open())
        obl = [o.witness_key() for o in extract_objectives(graph_from_dict(m, validate_bindings=False))]
        out["table2"][b] = [sum(1 for a in m["agents"] if a.get("reachable", True)), len(m["tools"]),
                            len(m["permissions"].get("allow", [])), len(m["permissions"].get("restrict", [])),
                            len(m["delegations"]), len(obl)]
        suites = build_suites(m, random.Random(SEED))
        out["table3"][b] = [len(suites[s]) for s in BASE_SUITES]
        allm = generate_mutants(m, b)
        capped = cap_per_operator(allm, 10 if b == "octagon_vc_agents" else 50, random.Random(SEED))
        items = []
        for i, mu in enumerate(capped):
            if v2.runtime_identical(m, mu):
                out["placeholder_skipped"] += 1
            elif well_formedness_violations(mu.manifest):
                out["ill_formed"] += 1
            else:
                items.append((i, mu))
        kept, dropped = drop_runtime_duplicates(items)
        dup_ops.update(mu.operator for _, mu in dropped)
        out["generated"] += len(allm)
        out["after_cap"] += len(capped)
        out["runtime_duplicates"] += len(dropped)
        out["scored"] += len(kept)
        if b == "oai_customer_service":
            out["running_example_generated"] = len(allm)
            out["running_example_scored"] = dict(sorted(Counter(mu.operator for _, mu in kept).items()))
    out["table2_total"] = [sum(v[i] for v in out["table2"].values()) for i in range(6)]
    out["table3_total"] = [sum(v[i] for v in out["table3"].values()) for i in range(4)]
    out["runtime_duplicates_by_operator"] = dict(sorted(dup_ops.items()))
    return out


def compute_tables(B):
    t4 = {}
    for op in OPS:
        mm = [m for d in B.values() for m in d["mutants"] if m["operator"] == op]
        t4[op] = [len(mm)] + [sum(1 for m in mm if witnesses(m, K)) for K in CRITERIA]
    return {
        "t_rand_restrict_probes": sum(1 for d in B.values() for o in d["meta"]["suites"]["T-Rand*"]
                                      if o[0] == "restrict_tool"),
        "table4": t4,
        "table4_total": [sum(v[i] for v in t4.values()) for i in range(4)],
        "table5": {s: [r(ms(B, s, "K-Struct", {o for o in OPS if FAM[o] == f})) for f in ("F1", "F2", "F3", "F4")]
                   + [r(ms(B, s, "K-Struct"))] for s in BASE_SUITES},
        "table6": {s: [r(ms(B, s, K)) for K in CRITERIA] for s in BASE_SUITES},
        "table8": {op: [r(ms(B, s, "K-Outcome", {op})) for s in BASE_SUITES] for op in OPS},
        "figure3": {K: {op: [r(ms(B, s, K, {op}), 2) for s in BASE_SUITES] for op in OPS} for K in CRITERIA},
    }


def compute_table7(B):
    out = {}
    for s in BASE_SUITES:
        cnt = Counter()
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                ps = [p for p in m["per_obl"] if tuple(p["obl"]) in S]
                trace = any(p["fires"].get("K-Struct") for p in ps)
                outcome = any(p["fires"].get("K-Outcome") for p in ps)
                verdict = False
                for p in ps:
                    if "votes" in p:
                        v = rescored_votes(m, p)
                        only = [o and not st for o, st in zip(v["K-Outcome"], v["K-Struct"])]
                        verdict |= sum(only) * 2 > len(only)
                cnt["both" if trace and verdict else "trace" if trace else
                    "verdict" if verdict else "mixed" if outcome else "neither"] += 1
        out[s] = [cnt[k] for k in ("trace", "verdict", "both", "mixed", "neither")]
    return out


def compute_stats(B):
    out = {"wilson_ci_kstruct": {}, "cochran_q": {}, "discordant_pairs": {},
           "bootstrap_ci": {}, "bootstrap_ci_difference": {}}
    for K in CRITERIA:
        rows = kill_rows(B, K)
        q, _ = cochran_q(rows, BASELINES)
        out["cochran_q"][K] = r(q, 1)
        out["discordant_pairs"][K] = {f"{x} vs {y}": [sum(1 for _, z in rows if z[x] and not z[y]),
                                                      sum(1 for _, z in rows if z[y] and not z[x])]
                                      for x, y in PAIRS}
        out["bootstrap_ci"][K] = {s: [r(v) for v in cluster_boot(rows, lambda smp, s=s: ms_rows(smp, s))]
                                  for s in BASELINES}
        out["bootstrap_ci_difference"][K] = {
            f"{x} - {y}": [r(v) for v in cluster_boot(rows, lambda smp, x=x, y=y: ms_rows(smp, x) - ms_rows(smp, y))]
            for x, y in PAIRS}
        if K == "K-Struct":
            n = len(rows)
            for s in BASE_SUITES:
                out["wilson_ci_kstruct"][s] = [r(v) for v in wilson(sum(z[s] for _, z in rows), n)]
    by = {}
    for b, z in kill_rows(B, "K-Struct"):
        by.setdefault(b, []).append((b, z))
    out["per_benchmark_kstruct"] = {b: [len(v)] + [r(ms_rows(v, s)) for s in BASELINES] for b, v in sorted(by.items())}
    out["macro_average_kstruct"] = [r(sum(ms_rows(v, s) for v in by.values()) / len(by)) for s in BASELINES]
    return out


def compute_size_matched(B):
    """Same draw sequence as analyze_v2.py (one generator across the criteria)."""
    sizes = {b: len(d["meta"]["suites"]["T-Dev*"]) for b, d in B.items()}
    rng = random.Random(SEED)
    out = {}
    for K in CRITERIA:
        wit = [(b, w) for b, d in B.items() for w in (witnesses(m, K) for m in d["mutants"]) if w]
        draws = []
        for _ in range(100):
            pick = {b: set(rng.sample(sorted(d["meta"]["suites"]["T-Struct"]), sizes[b]))
                    for b, d in B.items()}
            draws.append(sum(bool(w & pick[b]) for b, w in wit) / len(wit))
        draws.sort()
        greedy = {b: coverage_greedy(d["meta"]["suites"]["T-Struct"], sizes[b]) for b, d in B.items()}
        out[K] = {"t_dev": r(ms(B, "T-Dev*", K)), "random_mean": r(sum(draws) / len(draws)),
                  "random_range": [r(draws[2]), r(draws[97])],
                  "greedy": r(sum(bool(w & greedy[b]) for b, w in wit) / len(wit))}
    return out


def compute_ksafety_sources(B):
    out = {}
    for s in BASELINES:
        tot = probe = other = 0
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                hit = witnesses(m, "K-Safety") & S
                if hit:
                    tot += 1
                    probe += any(o[0] == "restrict_tool" for o in hit)
                    other += any(o[0] != "restrict_tool" for o in hit)
        out[s] = [tot, probe, other]
    return out


def compute_noise(B, results):
    n_exec = 0
    for p in results.glob("*.jsonl"):
        for line in p.open():
            rec = json.loads(line)
            if rec["type"] in ("orig", "control"):
                n_exec += sum(1 for x in rec["runs"] if x is not None)
            elif rec["type"] == "mutant":
                n_exec += sum(1 for q in rec["per_obl"] for x in q.get("runs", []) if x is not None)
    n_dup = sum(1 for d in B.values() for m in d["duplicates"] for q in m["per_obl"]
                for x in q.get("runs", []) if x is not None)
    dist = Counter()
    for d in B.values():
        for m in d["mutants"]:
            for p in m["per_obl"]:
                if "votes" in p:
                    dist[sum(rescored_votes(m, p)["K-Struct"])] += 1
    ctrl = {}
    for K in CRITERIA:
        v = [c["fires"].get(K) for d in B.values() for c in d["control"].values()]
        v = [x for x in v if x is not None]
        ctrl[K] = r(100 * sum(map(bool, v)) / len(v), 1)
    n_scored = sum(len(d["mutants"]) for d in B.values())
    exposure = {}
    for s in BASE_SUITES:
        hit = 0
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                hit += any(d["control"].get(tuple(o), {}).get("fires", {}).get("K-Struct")
                           for o in m["incident"] if tuple(o) in S)
        exposure[s] = r(100 * hit / n_scored, 1)
    return {
        "executions_logged": n_exec,
        "executions_of_duplicates": n_dup,
        "executions_scored": n_exec - n_dup,
        "max_turns_runs": sum(1 for d in B.values() for m in d["mutants"] for p in m["per_obl"]
                              for x in p.get("runs", []) if x and x.get("max_turns")),
        "not_executable_obligations": sum(1 for d in B.values() for m in d["mutants"]
                                          for p in m["per_obl"] if "structural_fail" in p),
        "kstruct_pairs_differing_pct": [r(100 * dist[i] / sum(dist.values()), 1) for i in range(4)],
        "control_rep1_fire_pct": ctrl,
        "control_rep1_noisy_incident_obligation_pct": exposure,
    }


def compute_audit(B, audit):
    from audit_v2 import kappa
    from run_live_mutation import _refused
    norm = lambda s: (s or "").strip().lower()  # noqa: E731
    first = {}
    for row in csv.DictReader((audit / "mutants.csv").open()):
        first.setdefault(row["mutant_id"], row)
    k, n = kappa([(norm(x["rater1"]), norm(x["rater2"])) for x in first.values()])
    eq = Counter(x["operator"] for x in first.values() if norm(x["final"]) == "eq")
    unex = 0
    for d in B.values():
        for m in d["mutants"]:
            if m["operator"] == "Tool-Fault":
                tool = m["target"].split("@")[0]
                unex += not any(tool in (x.get("tools") or []) for p in m["per_obl"] for x in p.get("runs", []) if x)
    sampled_unex = 0
    ids = {f"{b}:{m['idx']}": m for b, d in B.items() for m in d["mutants"]}
    for mid, x in first.items():
        if x["operator"] == "Tool-Fault" and norm(x["final"]) == "eq":
            m = ids[mid]
            tool = m["target"].split("@")[0]
            sampled_unex += not any(tool in (y.get("tools") or []) for p in m["per_obl"] for y in p.get("runs", []) if y)
    rows = list(csv.DictReader((audit / "verdicts.csv").open()))
    k2, n2 = kappa([(norm(x["rater1"]), norm(x["rater2"])) for x in rows])
    cell = Counter((x["_predicted_hidden"], norm(x["final"])) for x in rows)
    tp, fp, fn, tn = cell[("fail", "fail")], cell[("fail", "ok")], cell[("ok", "fail")], cell[("ok", "ok")]
    texts = {x["text"] for d in B.values() for m in d["mutants"] + d["duplicates"] for p in m["per_obl"]
             for x in p.get("runs", []) if x and not x.get("max_turns")}
    wf = sum(1 for t in texts if _refused(t)) / len(texts)
    ptp, pfn, ptn = wf * tp / (tp + fp), (1 - wf) * fn / (fn + tn), (1 - wf) * tn / (fn + tn)
    return {
        "mutants_labelled": n, "raters_agree": sum(norm(x["rater1"]) == norm(x["rater2"]) for x in first.values()),
        "kappa_mutants": r(k, 2), "labelled_eq_by_operator": dict(sorted(eq.items())),
        "sampled_eq_tool_fault_never_calling_tool": sampled_unex,
        "tool_fault_never_calling_tool": unex,
        "responses_labelled_by_both": n2, "kappa_responses": r(k2, 2),
        "distinct_responses": len(texts), "verdict_precision": r(tp / (tp + fp), 2),
        "verdict_recall_weighted": r(ptp / (ptp + pfn), 2), "verdict_accuracy_weighted": r(ptp + ptn, 2),
    }


# ----------------------------------------------------------------------------
def compare(expected, got, path, counts):
    if isinstance(expected, dict):
        for k, v in expected.items():
            if k.startswith("_"):
                continue
            if not isinstance(got, dict) or k not in got:
                counts["fail"] += 1
                print(f"  FAIL  {path}{k}: not computed")
                continue
            compare(v, got[k], f"{path}{k} / ", counts)
        return
    ok = expected == got
    counts["pass" if ok else "fail"] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {path.rstrip(' /')}: got {got!r}" + ("" if ok else f", paper {expected!r}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--audit", default=str(ROOT / "audit"))
    ap.add_argument("--values", default=str(ROOT / "paper_values.json"))
    a = ap.parse_args()
    results = Path(a.results)
    B = load(results, set())
    expected = json.load(open(a.values))
    computed = {
        "inventory": compute_inventory(),
        "tables": compute_tables(B),
        "table7": compute_table7(B),
        "statistics": compute_stats(B),
        "size_matched": compute_size_matched(B),
        "ksafety_sources": compute_ksafety_sources(B),
        "noise_and_runs": compute_noise(B, results),
    }
    if (Path(a.audit) / "mutants.csv").exists():
        computed["audit"] = compute_audit(B, Path(a.audit))

    counts = Counter()
    for group, exp in expected.items():
        if group.startswith("_"):
            continue
        print(f"\n--- {group}")
        if group == "noise_three_replicates":
            ctrl = results.parent / f"{results.name}_control"
            if not ctrl.is_dir() or not (ROOT / "analyze_control_combined.py").exists():
                for k in exp:
                    if not k.startswith("_"):
                        counts["skip"] += 1
                        print(f"  SKIP  {k}: needs {ctrl.name}/ and analyze_control_combined.py")
                continue
        if group not in computed:
            counts["skip"] += 1
            print(f"  SKIP  {group}: inputs not found")
            continue
        compare(exp, computed[group], "", counts)
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped")
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
