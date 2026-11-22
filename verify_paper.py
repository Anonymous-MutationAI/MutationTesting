#!/usr/bin/env python3
"""Check the numbers reported in the paper against the logged runs.

    python3 verify_paper.py                 # uses results/v2 and audit/
    python3 verify_paper.py --results DIR   # e.g. a fresh live re-run

Needs only the Python standard library and makes no API calls. Each check
prints PASS or FAIL with the recomputed and the reported value; the script
exits with status 1 if any check fails. Numbers that depend on the two extra
noise-control replicates are checked only when results/v2_control/ is present.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analyze_v2 import (BASE_SUITES, CRITERIA, FAM, OPS, coverage_greedy,  # noqa: E402
                        load, witnesses, wilson)

BENCHES = ["autopitch", "deep_research_clone", "oai_customer_service",
           "oai_financial_research", "oai_message_filter", "oai_research_bot",
           "octagon_vc_agents", "social_media_agent_system", "value_investment",
           "ydmitry_deep_research"]
SEED = 20260605

# ----------------------------------------------------------------------------
# Values as reported in the paper
# ----------------------------------------------------------------------------
TABLE2 = {  # Ag Tl Al Re De Obl
    "autopitch": (7, 6, 6, 36, 6, 55),
    "deep_research_clone": (3, 1, 1, 2, 2, 8),
    "oai_customer_service": (3, 2, 2, 4, 4, 13),
    "oai_financial_research": (7, 7, 7, 42, 6, 62),
    "oai_message_filter": (3, 1, 1, 2, 2, 8),
    "oai_research_bot": (4, 4, 4, 12, 3, 23),
    "octagon_vc_agents": (12, 12, 12, 132, 11, 167),
    "social_media_agent_system": (2, 3, 3, 3, 1, 9),
    "value_investment": (4, 9, 24, 12, 3, 43),
    "ydmitry_deep_research": (4, 2, 5, 3, 3, 15),
}
TABLE3 = {  # T-Dev* T-Rand* T-E2E* T-Struct
    "autopitch": (13, 6, 9, 55),
    "deep_research_clone": (4, 1, 4, 8),
    "oai_customer_service": (5, 1, 5, 13),
    "oai_financial_research": (14, 6, 10, 62),
    "oai_message_filter": (4, 1, 4, 8),
    "oai_research_bot": (8, 2, 6, 23),
    "octagon_vc_agents": (24, 17, 14, 167),
    "social_media_agent_system": (5, 1, 4, 9),
    "value_investment": (28, 4, 8, 43),
    "ydmitry_deep_research": (9, 2, 7, 15),
}
TABLE4 = {  # Total K-Struct K-Outcome K-Safety
    "Del-Rem": (40, 37, 38, 0), "Del-Red": (96, 94, 94, 0),
    "Del-Inv": (36, 35, 36, 0), "Del-Add": (120, 113, 114, 0),
    "Tool-Rem": (63, 63, 63, 0), "Tool-Red": (157, 157, 157, 63),
    "Tool-Fault": (80, 40, 58, 0), "Tool-SigMut": (45, 45, 45, 0),
    "Res-Relax": (126, 124, 124, 115), "Res-Tighten": (63, 63, 63, 0),
    "Res-Rem": (126, 92, 102, 0), "Ag-Rem": (38, 38, 38, 0),
    "Ag-Swap": (47, 42, 45, 0), "Ag-PromptDrop": (39, 29, 30, 0),
}
TABLE5 = {  # F1 F2 F3 F4 Overall (K-Struct)
    "T-Dev*": (.731, .928, .609, .670, .751),
    "T-Rand*": (.452, .610, .409, .532, .498),
    "T-E2E*": (.552, .600, .473, .642, .555),
    "T-Struct": (1, 1, 1, 1, 1),
}
TABLE6 = {  # K-Struct K-Outcome K-Safety
    "T-Dev*": (.751, .742, .573), "T-Rand*": (.498, .524, .376),
    "T-E2E*": (.555, .548, .500), "T-Struct": (1, 1, 1),
}
TABLE7 = {  # trace-only verdict-only both mixed neither
    "T-Dev*": (713, 13, 17, 4, 329), "T-Rand*": (439, 35, 45, 9, 548),
    "T-E2E*": (532, 8, 7, 5, 524), "T-Struct": (741, 25, 231, 10, 69),
}
TABLE8 = {  # K-Outcome per operator: T-Dev* T-Rand* T-E2E* T-Struct
    "Del-Rem": (.842, .553, .579, 1), "Del-Red": (.787, .543, .532, 1),
    "Del-Inv": (.889, .667, .750, 1), "Del-Add": (.579, .526, .500, 1),
    "Tool-Rem": (.937, .603, .619, 1), "Tool-Red": (.962, .669, .618, 1),
    "Tool-Fault": (.707, .121, .207, 1), "Tool-SigMut": (1, .867, .911, 1),
    "Res-Relax": (.621, .484, .581, 1), "Res-Tighten": (.968, .619, .667, 1),
    "Res-Rem": (.333, .216, .206, 1), "Ag-Rem": (1, .763, 1, 1),
    "Ag-Swap": (.511, .444, .489, 1), "Ag-PromptDrop": (.467, .433, .400, 1),
}
FIG3_KSTRUCT = {  # T-Dev* T-Rand* T-E2E* T-Struct (two decimals)
    "Del-Rem": (.86, .38, .57, 1), "Del-Red": (.79, .39, .52, 1),
    "Del-Inv": (.91, .69, .77, 1), "Del-Add": (.58, .45, .50, 1),
    "Tool-Rem": (.94, .59, .62, 1), "Tool-Red": (.96, .67, .62, 1),
    "Tool-Fault": (.70, .12, .17, 1), "Tool-SigMut": (1, .87, .89, 1),
    "Res-Relax": (.62, .46, .58, 1), "Res-Tighten": (.97, .62, .65, 1),
    "Res-Rem": (.35, .20, .21, 1), "Ag-Rem": (1, .74, 1, 1),
    "Ag-Swap": (.52, .40, .50, 1), "Ag-PromptDrop": (.45, .45, .38, 1),
}
FIG3_KSAFETY = {"Tool-Red": (.62, .25, .41, 1), "Res-Relax": (.55, .44, .55, 1)}
RUNNING_EXAMPLE_OPS = {  # oai_customer_service executed mutants (Section 3.2)
    "Del-Rem": 4, "Del-Red": 4, "Del-Inv": 0, "Del-Add": 2, "Tool-Rem": 2,
    "Tool-Red": 2, "Tool-Fault": 4, "Tool-SigMut": 2, "Res-Relax": 4,
    "Res-Tighten": 2, "Res-Rem": 4, "Ag-Rem": 2, "Ag-Swap": 0, "Ag-PromptDrop": 0,
}


# ----------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.passed = self.failed = self.skipped = 0

    def section(self, title):
        print(f"\n--- {title}")

    def eq(self, label, got, want, tol=0.0):
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = abs(got - want) <= tol + 1e-9
        else:
            ok = got == want
        self.passed += ok
        self.failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, paper {want!r}")

    def skip(self, label, why):
        self.skipped += 1
        print(f"  SKIP  {label}: {why}")


def ms(benches, suite, K, ops=None, adjusted=False):
    kd = kb = 0
    for d in benches.values():
        S = set(d["meta"]["suites"][suite])
        for m in d["mutants"]:
            if ops and m["operator"] not in ops:
                continue
            w = witnesses(m, K, d["control"], adjusted)
            if w:
                kb += 1
                kd += bool(w & S)
    return kd, kb


def r3(kd, kb):
    return round(kd / kb, 3) if kb else None


def r2(kd, kb):
    return round(kd / kb, 2) if kb else None


def check_inventory(c):
    """Tables 2 and 3, mutant generation and capping (from benchmarks/ via src/)."""
    from agentcov import extract_objectives, graph_from_dict
    from agentmut import build_suites, cap_per_operator, generate_mutants
    spec = importlib.util.spec_from_file_location("v2", ROOT / "run_live_mutation_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)

    c.section("Table 2: benchmark inventory")
    tot = [0] * 6
    gen = capped_n = skipped = 0
    cs_ops = None
    for b in BENCHES:
        m = json.load((ROOT / "benchmarks" / f"{b}.json").open())
        g = graph_from_dict(m, validate_bindings=False)
        obl = [o.witness_key() for o in extract_objectives(g)]
        row = (sum(1 for a in m["agents"] if a.get("reachable", True)), len(m["tools"]),
               len(m["permissions"].get("allow", [])), len(m["permissions"].get("restrict", [])),
               len(m["delegations"]), len(obl))
        c.eq(f"{b} (Ag, Tl, Al, Re, De, Obl)", row, TABLE2[b])
        tot = [x + y for x, y in zip(tot, row)]

        allm = generate_mutants(m, b)
        cap = 10 if b == "octagon_vc_agents" else 50
        capped = cap_per_operator(allm, cap, random.Random(SEED))
        kept = [mu for mu in capped if not v2.runtime_identical(m, mu)]
        gen += len(allm)
        capped_n += len(capped)
        skipped += len(capped) - len(kept)
        if b == "oai_customer_service":
            cs_ops = Counter(mu.operator for mu in kept)
    c.eq("Total (Ag, Tl, Al, Re, De, Obl)", tuple(tot), (49, 47, 65, 248, 41, 403))

    c.section("Section 4.4: mutant generation")
    c.eq("first-order mutants generated", gen, 1873)
    c.eq("runtime-identical placeholder mutants skipped", skipped, 8)
    c.eq("executed mutants (capped minus skipped)", capped_n - skipped, 1076)
    c.eq("oai_customer_service executed mutants", sum(cs_ops.values()), 32)
    c.eq("oai_customer_service per-operator counts",
         {op: cs_ops.get(op, 0) for op in RUNNING_EXAMPLE_OPS}, RUNNING_EXAMPLE_OPS)


def check_tables(c, B):
    c.section("Table 3: suite sizes")
    for b in BENCHES:
        got = tuple(len(B[b]["meta"]["suites"][s]) for s in BASE_SUITES)
        c.eq(b, got, TABLE3[b])
    c.eq("Total", tuple(sum(len(B[b]["meta"]["suites"][s]) for b in BENCHES) for s in BASE_SUITES),
         (114, 41, 71, 403))
    c.eq("restrict probes in T-Rand*",
         sum(1 for b in BENCHES for o in B[b]["meta"]["suites"]["T-Rand*"] if o[0] == "restrict_tool"), 24)

    c.section("Table 4: executed pool and live-killable denominators")
    tot = [0, 0, 0, 0]
    for op, want in TABLE4.items():
        ms_ = [m for d in B.values() for m in d["mutants"] if m["operator"] == op]
        got = (len(ms_),) + tuple(sum(1 for m in ms_ if witnesses(m, K)) for K in CRITERIA)
        c.eq(op, got, want)
        tot = [x + y for x, y in zip(tot, got)]
    c.eq("Total", tuple(tot), (1076, 972, 1007, 178))

    c.section("Table 5: K-Struct MS by family")
    for s, want in TABLE5.items():
        got = tuple(r3(*ms(B, s, "K-Struct", {o for o in OPS if FAM[o] == f}))
                    for f in ("F1", "F2", "F3", "F4")) + (r3(*ms(B, s, "K-Struct")),)
        c.eq(s, got, tuple(float(x) for x in want), tol=0.0005)

    c.section("Table 6: MS under each criterion")
    for s, want in TABLE6.items():
        got = tuple(r3(*ms(B, s, K)) for K in CRITERIA)
        c.eq(s, got, tuple(float(x) for x in want), tol=0.0005)

    c.section("Table 8: K-Outcome MS per operator")
    for op, want in TABLE8.items():
        got = tuple(r3(*ms(B, s, "K-Outcome", {op})) for s in BASE_SUITES)
        c.eq(op, got, tuple(float(x) for x in want), tol=0.0005)

    c.section("Figure 3: K-Struct and K-Safety MS per operator (two decimals)")
    for op, want in FIG3_KSTRUCT.items():
        got = tuple(r2(*ms(B, s, "K-Struct", {op})) for s in BASE_SUITES)
        c.eq(f"K-Struct {op}", got, tuple(float(x) for x in want), tol=0.005)
    for op in OPS:
        kb = ms(B, "T-Struct", "K-Safety", {op})[1]
        if op in FIG3_KSAFETY:
            got = tuple(r2(*ms(B, s, "K-Safety", {op})) for s in BASE_SUITES)
            c.eq(f"K-Safety {op}", got, tuple(float(x) for x in FIG3_KSAFETY[op]), tol=0.005)
        else:
            c.eq(f"K-Safety {op} is n/a", kb, 0)


def check_table7(c, B):
    c.section("Table 7: trace versus verdict kills (all 1,076 mutants)")
    for s, want in TABLE7.items():
        cnt = Counter()
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                ps = [p for p in m["per_obl"] if tuple(p["obl"]) in S]
                trace = any(p["fires"].get("K-Struct") for p in ps)
                outcome = any(p["fires"].get("K-Outcome") for p in ps)
                verdict = False
                for p in ps:
                    v = p.get("votes")
                    if v:
                        only = [o and not st for o, st in zip(v["K-Outcome"], v["K-Struct"])]
                        verdict |= sum(only) * 2 > len(only)
                key = ("both" if trace and verdict else "trace" if trace else
                       "verdict" if verdict else "mixed" if outcome else "neither")
                cnt[key] += 1
        got = tuple(cnt[k] for k in ("trace", "verdict", "both", "mixed", "neither"))
        c.eq(s, got, want)


def check_rq1_stats(c, B):
    c.section("Section 4.5: confidence intervals and tests (K-Struct, raw)")
    rows = []
    for d in B.values():
        for m in d["mutants"]:
            w = witnesses(m, "K-Struct")
            if w:
                rows.append({s: bool(w & set(d["meta"]["suites"][s])) for s in BASE_SUITES})
    n = len(rows)
    kills = {s: sum(r[s] for r in rows) for s in BASE_SUITES}
    for s, want in (("T-Dev*", (0.72, 0.78)), ("T-E2E*", (0.52, 0.59)), ("T-Rand*", (0.47, 0.53))):
        lo, hi = wilson(kills[s], n)
        c.eq(f"95% Wilson CI {s}", (round(lo, 2), round(hi, 2)), want)
    lo, hi = wilson(kills["T-Struct"], n)
    c.eq("95% Wilson CI T-Struct", (round(lo, 3), round(hi, 3)), (0.996, 1.0))
    tot = sum(kills[s] for s in BASE_SUITES[:3])
    chi = sum((o - e) ** 2 / e for s in BASE_SUITES[:3]
              for o, e in ((kills[s], n * tot / (3 * n)), (n - kills[s], n * (3 * n - tot) / (3 * n))))
    c.eq("chi^2(2) across the three baselines", round(chi, 1), 143.1)
    only = (sum(1 for r in rows if r["T-Dev*"] and not r["T-E2E*"]),
            sum(1 for r in rows if r["T-E2E*"] and not r["T-Dev*"]))
    c.eq("discordant pairs (T-Dev* only, T-E2E* only)", only, (191, 0))

    c.section("Section 4.5: size-matched subsets of T-Struct (raw)")
    sizes = {b: len(d["meta"]["suites"]["T-Dev*"]) for b, d in B.items()}
    rng = random.Random(SEED)  # same draw sequence as analyze_v2.py
    want = {"K-Struct": (0.757, None, 0.784), "K-Safety": (0.664, (0.573, 0.775), 0.725)}
    for K in CRITERIA:
        draws = []
        for _ in range(100):
            pick = {b: set(rng.sample(sorted(d["meta"]["suites"]["T-Struct"]),
                                      min(sizes[b], len(d["meta"]["suites"]["T-Struct"]))))
                    for b, d in B.items()}
            kd = kb = 0
            for b, d in B.items():
                for m in d["mutants"]:
                    w = witnesses(m, K)
                    if w:
                        kb += 1
                        kd += bool(w & pick[b])
            draws.append(kd / kb)
        draws.sort()
        greedy = {b: coverage_greedy(d["meta"]["suites"]["T-Struct"], sizes[b]) for b, d in B.items()}
        gkd = gkb = 0
        for b, d in B.items():
            for m in d["mutants"]:
                w = witnesses(m, K)
                if w:
                    gkb += 1
                    gkd += bool(w & greedy[b])
        if K in want:
            mean, rng_want, g = want[K]
            c.eq(f"{K} random-subset mean", round(sum(draws) / len(draws), 3), mean)
            if rng_want:
                c.eq(f"{K} random-subset 95% range", (round(draws[2], 3), round(draws[97], 3)), rng_want)
            c.eq(f"{K} coverage-greedy subset", round(gkd / gkb, 3), g)


def check_ksafety_sources(c, B):
    c.section("Section 4.6: where K-Safety kills come from")
    want = {"T-Dev*": (102, 0, 102), "T-E2E*": (89, 0, 89), "T-Rand*": (67, 51, 17)}
    for s, w_ in want.items():
        tot = probe = other = 0
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                hit = witnesses(m, "K-Safety") & S
                if hit:
                    tot += 1
                    probe += any(o[0] == "restrict_tool" for o in hit)
                    other += any(o[0] != "restrict_tool" for o in hit)
        c.eq(f"{s} (kills, via restrict probes, via other tests)", (tot, probe, other), w_)


def check_noise(c, B, results):
    c.section("Sections 4.3 and 4.9: executions, votes and the first control replicate")
    n_exec = 0
    for p in results.glob("*.jsonl"):
        for line in p.open():
            r = json.loads(line)
            if r["type"] in ("orig", "control"):
                n_exec += sum(1 for x in r["runs"] if x is not None)
            elif r["type"] == "mutant":
                n_exec += sum(1 for q in r["per_obl"] for x in q.get("runs", []) if x is not None)
    c.eq("agent executions (main sweep + first control)", n_exec, 52251)
    mt = sum(1 for d in B.values() for m in d["mutants"] for p in m["per_obl"]
             for r in p.get("runs", []) if r and r.get("max_turns"))
    c.eq("mutant runs reaching max_turns", mt, 573)
    dist = Counter()
    for d in B.values():
        for m in d["mutants"]:
            for p in m["per_obl"]:
                if "votes" in p:
                    dist[sum(p["votes"]["K-Struct"])] += 1
    tot = sum(dist.values())
    c.eq("K-Struct pairs differing 0/1/2/3 of 3 (%)",
         tuple(round(100 * dist[i] / tot, 1) for i in range(4)), (57.0, 12.0, 8.1, 22.9))
    for K, want in (("K-Struct", 7.4), ("K-Outcome", 12.7), ("K-Safety", 0.0)):
        f = n = 0
        for d in B.values():
            for cr in d["control"].values():
                v = cr["fires"].get(K)
                if v is not None:
                    n += 1
                    f += bool(v)
        c.eq(f"first control replicate fires under {K} (% of obligations)", round(100 * f / n, 1), want)
    fk = {"T-Dev*": 39.2, "T-E2E*": 21.4, "T-Rand*": 19.4, "T-Struct": 64.4}
    for s, want in fk.items():
        hit = 0
        for d in B.values():
            S = set(d["meta"]["suites"][s])
            for m in d["mutants"]:
                hit += any(d["control"].get(tuple(o), {}).get("fires", {}).get("K-Struct")
                           for o in m["incident"] if tuple(o) in S)
        c.eq(f"first replicate: % of mutants with a noisy incident obligation in {s}",
             round(100 * hit / 1076, 1), want, tol=0.2)


def check_three_replicates(c, results):
    c.section("Noise-adjusted scores (all three control replicates)")
    ctrl = results.parent / f"{results.name}_control"
    if not ctrl.is_dir() or not (ROOT / "analyze_control_combined.py").exists():
        for label in ("noise-adjusted MS (K-Struct 0.653 / 0.392 / 0.352)",
                      "size-matched subsets, noise-adjusted (0.661, greedy 0.562)",
                      "noisy obligations over three replicates (21.6% / 30.3%)",
                      "853 mutants killable after adjustment"):
            c.skip(label, f"needs {ctrl.name}/ and analyze_control_combined.py")
        return
    c.skip("noise-adjusted scores", "run analyze_control_combined.py and compare with Section 4.9")


def check_audit(c, B, audit):
    c.section("Section 4.10: equivalence audit and verdict validation")
    if not (audit / "mutants.csv").exists():
        c.skip("audit", f"{audit}/mutants.csv not found")
        return
    from audit_v2 import kappa
    from run_live_mutation import _refused
    norm = lambda s: (s or "").strip().lower()  # noqa: E731

    first = {}
    for r in csv.DictReader((audit / "mutants.csv").open()):
        first.setdefault(r["mutant_id"], r)
    k, n = kappa([(norm(r["rater1"]), norm(r["rater2"])) for r in first.values()])
    agree = sum(norm(r["rater1"]) == norm(r["rater2"]) for r in first.values())
    c.eq("mutants labelled", n, 90)
    c.eq("raters agree", agree, 89)
    c.eq("Cohen's kappa (mutants)", round(k, 2), 0.95)
    eq_ag = Counter(r["operator"] for r in first.values()
                    if norm(r["final"]) == "eq" and r["operator"] != "Tool-Fault")
    c.eq("equivalent Ag-Swap / Ag-PromptDrop mutants", (eq_ag["Ag-Swap"], eq_ag["Ag-PromptDrop"]), (1, 1))

    unexercised = 0
    for d in B.values():
        for m in d["mutants"]:
            if m["operator"] == "Tool-Fault":
                tool = m["target"].split("@")[0]
                unexercised += not any(tool in (r.get("tools") or [])
                                       for p in m["per_obl"] for r in p.get("runs", []) if r)
    c.eq("Tool-Fault mutants that never call the faulted tool", unexercised, 15)

    rows = list(csv.DictReader((audit / "verdicts.csv").open()))
    k2, n2 = kappa([(norm(r["rater1"]), norm(r["rater2"])) for r in rows])
    c.eq("responses labelled by both raters", n2, 199)
    c.eq("Cohen's kappa (responses)", round(k2, 2), 0.85)
    cell = Counter((r["_predicted_hidden"], norm(r["final"])) for r in rows)
    tp, fp = cell[("fail", "fail")], cell[("fail", "ok")]
    fn, tn = cell[("ok", "fail")], cell[("ok", "ok")]
    texts = {r["text"] for d in B.values() for m in d["mutants"] for p in m["per_obl"]
             for r in p.get("runs", []) if r and not r.get("max_turns")}
    n_fail = sum(1 for t in texts if _refused(t))
    c.eq("distinct mutant responses", len(texts), 47404)
    wf, wo = n_fail / len(texts), 1 - n_fail / len(texts)
    ptp, pfp = wf * tp / (tp + fp), wf * fp / (tp + fp)
    pfn, ptn = wo * fn / (fn + tn), wo * tn / (fn + tn)
    c.eq("verdict precision", round(tp / (tp + fp), 2), 0.68)
    c.eq("verdict recall (weighted by class share)", round(ptp / (ptp + pfn), 2), 0.71)
    c.eq("verdict accuracy (weighted by class share)", round(ptp + ptn, 2), 0.88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--audit", default=str(ROOT / "audit"))
    a = ap.parse_args()
    results = Path(a.results)
    B = load(results, set())
    missing = [b for b in BENCHES if b not in B]
    if missing:
        sys.exit(f"no logs for {missing} in {results}")

    c = Checker()
    check_inventory(c)
    check_tables(c, B)
    check_table7(c, B)
    check_rq1_stats(c, B)
    check_ksafety_sources(c, B)
    check_noise(c, B, results)
    check_three_replicates(c, results)
    check_audit(c, B, Path(a.audit))
    print(f"\n{c.passed} passed, {c.failed} failed, {c.skipped} skipped")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
