#!/usr/bin/env python3
"""
reproduce.py — one-command reproduction of
"Mutation Testing for Multi-Agent Systems". No bash required.

    python3 reproduce.py            # offline: regenerate & verify all tables (no API key, no deps)
    python3 reproduce.py --verify   # same as above
    python3 reproduce.py --csv out/ # also dump the tables as CSV
    python3 reproduce.py --live      # full live re-run from manifests (needs OPENAI_API_KEY)

Offline mode recomputes, entirely from the bundled inputs and the Python
standard library:
  * Table II  benchmark inventory            (benchmarks/ via src/agentcov)
  * §III-B    first-order mutant count 1,873  (benchmarks/ via src/agentmut)
  * Table III executed pool + denominators    (results/)
  * Table IV  K-Struct MS per family          (results/)
  * Table V   MS per suite x criterion        (results/)
  * Table VI  K-Outcome MS per operator       (results/)
  * §IV       29,601 agent executions         (results/)
and checks every value against the published paper. Exit 0 (PASS) iff all match.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

BENCHMARKS = [
    "autopitch", "deep_research_clone", "oai_customer_service",
    "oai_financial_research", "oai_message_filter", "oai_research_bot",
    "octagon_vc_agents", "social_media_agent_system", "value_investment",
    "ydmitry_deep_research",
]
SUITES = ["T-Dev*", "T-Rand*", "T-E2E*", "T-Struct"]
CRITERIA = ["K-Struct", "K-Outcome", "K-Safety"]
OPS = ["Del-Rem", "Del-Red", "Del-Inv", "Del-Add", "Tool-Rem", "Tool-Red",
       "Tool-Fault", "Tool-SigMut", "Res-Relax", "Res-Tighten", "Res-Rem",
       "Ag-Rem", "Ag-Swap", "Ag-PromptDrop"]
FAMILY = {op: ("F1" if op.startswith("Del") else "F2" if op.startswith("Tool")
               else "F3" if op.startswith("Res") else "F4") for op in OPS}
TOL = 0.0015

# ---------- published ground truth -------------------------------------------
PAPER_T2_TOTAL = dict(Ag=49, Tl=47, Al=65, Re=248, De=41, Obl=403)
PAPER_GEN_TOTAL = 1873
PAPER_GEN_PEROP = {
    "Del-Rem": 41, "Del-Red": 196, "Del-Inv": 37, "Del-Add": 231,
    "Tool-Rem": 65, "Tool-Red": 421, "Tool-Fault": 94, "Tool-SigMut": 47,
    "Res-Relax": 248, "Res-Tighten": 65, "Res-Rem": 248,
    "Ag-Rem": 39, "Ag-Swap": 92, "Ag-PromptDrop": 49,
}
PAPER_T3 = {
    "Del-Rem": (40, 39, 39, 0), "Del-Red": (96, 95, 95, 0),
    "Del-Inv": (36, 35, 35, 0), "Del-Add": (120, 112, 112, 0),
    "Tool-Rem": (63, 62, 62, 0), "Tool-Red": (157, 154, 154, 61),
    "Tool-Fault": (80, 45, 62, 0), "Tool-SigMut": (45, 45, 45, 0),
    "Res-Relax": (126, 125, 125, 116), "Res-Tighten": (63, 63, 63, 0),
    "Res-Rem": (126, 91, 100, 0), "Ag-Rem": (38, 38, 38, 0),
    "Ag-Swap": (47, 43, 43, 0), "Ag-PromptDrop": (47, 36, 37, 0),
}
PAPER_T3_TOTAL = (1084, 983, 1010, 177)
PAPER_T4 = {
    "T-Dev*": (0.665, 0.925, 0.584, 0.615, 0.717),
    "T-Rand*": (0.420, 0.627, 0.412, 0.470, 0.488),
    "T-E2E*": (0.527, 0.667, 0.462, 0.590, 0.560),
    "T-Struct": (1.0, 1.0, 1.0, 1.0, 1.0),
}
PAPER_T5 = {"T-Dev*": (0.717, 0.725, 0.559), "T-Rand*": (0.488, 0.506, 0.401),
            "T-E2E*": (0.560, 0.563, 0.503), "T-Struct": (1.0, 1.0, 1.0)}
PAPER_T6 = {
    "Del-Rem": (0.769, 0.564, 0.564, 1.0), "Del-Red": (0.684, 0.389, 0.474, 1.0),
    "Del-Inv": (0.857, 0.657, 0.771, 1.0), "Del-Add": (0.554, 0.500, 0.491, 1.0),
    "Tool-Rem": (1.000, 0.710, 0.855, 1.0), "Tool-Red": (0.961, 0.682, 0.649, 1.0),
    "Tool-Fault": (0.726, 0.113, 0.258, 1.0), "Tool-SigMut": (1.000, 0.867, 0.956, 1.0),
    "Res-Relax": (0.616, 0.488, 0.568, 1.0), "Res-Tighten": (0.968, 0.619, 0.651, 1.0),
    "Res-Rem": (0.280, 0.190, 0.200, 1.0), "Ag-Rem": (1.000, 0.789, 1.000, 1.0),
    "Ag-Swap": (0.628, 0.442, 0.605, 1.0), "Ag-PromptDrop": (0.378, 0.270, 0.324, 1.0),
}
PAPER_EXECUTIONS = 29601


def fmt(x): return "n/a" if x != x else f"{x:.3f}"


def check(label, got, exp, fails, tol=TOL):
    if not (abs(got - exp) <= tol):
        fails.append(f"{label}: got {fmt(got)} expected {fmt(exp)}")


def table2_and_generation(bench_dir: Path, fails, csv):
    try:
        from agentcov import extract_objectives, graph_from_json
        from agentmut import generate_mutants
    except Exception as e:  # pragma: no cover
        sys.exit(f"ERROR importing bundled src/ libraries: {e}\n"
                 f"Run this script from inside the package (it adds ./src to the path).")
    print("\n=== Table II: benchmark inventory (recomputed from benchmarks/) ===")
    print(f"{'benchmark':28s} Ag Tl Al  Re  De  Obl")
    tot = dict(Ag=0, Tl=0, Al=0, Re=0, De=0, Obl=0)
    rows = []; per_op = Counter(); gen_total = 0
    for b in BENCHMARKS:
        p = bench_dir / f"{b}.json"
        g = graph_from_json(p, validate_bindings=False)
        r = dict(Ag=len(g.reachable_agents), Tl=len(g.tools),
                 Al=len(g.allowed_obligations), Re=len(g.restricted_obligations),
                 De=len(g.delegation_obligations), Obl=len(extract_objectives(g)))
        print(f"{b:28s} {r['Ag']:2d} {r['Tl']:2d} {r['Al']:2d} {r['Re']:3d} {r['De']:2d} {r['Obl']:3d}")
        rows.append([b] + [r[k] for k in ("Ag", "Tl", "Al", "Re", "De", "Obl")])
        for k in tot:
            tot[k] += r[k]
        for mu in generate_mutants(json.loads(p.read_text()), b):
            op = mu["operator"] if isinstance(mu, dict) else getattr(mu, "operator")
            per_op[str(op)] += 1; gen_total += 1
    print(f"{'TOTAL':28s} {tot['Ag']:2d} {tot['Tl']:2d} {tot['Al']:2d} {tot['Re']:3d} {tot['De']:2d} {tot['Obl']:3d}")
    for k, v in PAPER_T2_TOTAL.items():
        if tot[k] != v:
            fails.append(f"T2 total {k}: got {tot[k]} expected {v}")
    print(f"\n=== First-order generation: {gen_total} mutants (paper 1,873) ===")
    if gen_total != PAPER_GEN_TOTAL:
        fails.append(f"generation total: got {gen_total} expected {PAPER_GEN_TOTAL}")
    for op, exp in PAPER_GEN_PEROP.items():
        if per_op[op] != exp:
            fails.append(f"generation {op}: got {per_op[op]} expected {exp}")
    csv["table2"] = ("benchmark,Ag,Tl,Al,Re,De,Obl", [",".join(map(str, r)) for r in rows])


def load_results(results_dir: Path):
    agg = defaultdict(lambda: {"killable": 0, "killed": 0})
    per_op_total = Counter(); seen = set(); execs = 0
    for b in BENCHMARKS:
        fp = results_dir / f"{b}.json"
        if not fp.exists():
            sys.exit(f"ERROR: missing results file {fp}")
        d = json.loads(fp.read_text())
        execs += (d.get("live_calls") or {}).get("n", 0)
        for r in d["rows"]:
            key = (r["operator"], r["suite"], r["criterion"])
            agg[key]["killable"] += r["killable"]
            agg[key]["killed"] += r["killed"]
            if (r["operator"], b) not in seen:
                seen.add((r["operator"], b)); per_op_total[r["operator"]] += r["total"]
    return agg, per_op_total, execs


def score(agg, ops, s, c):
    kd = sum(agg[(op, s, c)]["killed"] for op in ops)
    kb = sum(agg[(op, s, c)]["killable"] for op in ops)
    return kd / kb if kb else float("nan")


def verify(bench_dir: Path, results_dir: Path, csv_dir):
    fails, csv = [], {}
    table2_and_generation(bench_dir, fails, csv)
    agg, per_op_total, execs = load_results(results_dir)

    print("\n=== Table III: executed pool + live-killable denominators ===")
    print(f"{'Operator':14s}{'Total':>7}{'K-Struct':>10}{'K-Outcome':>11}{'K-Safety':>10}")
    tot = [0, 0, 0, 0]; rows3 = []
    for op in OPS:
        row = (per_op_total[op],
               agg[(op, "T-Struct", "K-Struct")]["killable"],
               agg[(op, "T-Struct", "K-Outcome")]["killable"],
               agg[(op, "T-Struct", "K-Safety")]["killable"])
        print(f"{op:14s}{row[0]:7d}{row[1]:10d}{row[2]:11d}{row[3]:10d}")
        rows3.append([op, FAMILY[op], *row])
        for i in range(4):
            tot[i] += row[i]
        if row != PAPER_T3[op]:
            fails.append(f"T3 {op}: got {row} expected {PAPER_T3[op]}")
    print(f"{'TOTAL':14s}{tot[0]:7d}{tot[1]:10d}{tot[2]:11d}{tot[3]:10d}")
    if tuple(tot) != PAPER_T3_TOTAL:
        fails.append(f"T3 TOTAL: got {tuple(tot)} expected {PAPER_T3_TOTAL}")
    csv["table3"] = ("operator,family,total,kstruct,koutcome,ksafety",
                     [",".join(map(str, r)) for r in rows3])

    print("\n=== Table IV: K-Struct MS_strict per family per suite ===")
    print(f"{'Suite':10s}{'F1':>8}{'F2':>8}{'F3':>8}{'F4':>8}{'MS':>8}")
    rows4 = []
    for s in SUITES:
        fam = [score(agg, [o for o in OPS if FAMILY[o] == f], s, "K-Struct")
               for f in ("F1", "F2", "F3", "F4")]
        ov = score(agg, OPS, s, "K-Struct")
        print(f"{s:10s}" + "".join(f"{fmt(v):>8}" for v in fam) + f"{fmt(ov):>8}")
        rows4.append([s] + [f"{v:.3f}" for v in fam] + [f"{ov:.3f}"])
        for i, f in enumerate(("F1", "F2", "F3", "F4")):
            check(f"T4 {s} {f}", fam[i], PAPER_T4[s][i], fails)
        check(f"T4 {s} overall", ov, PAPER_T4[s][4], fails)
    csv["table4"] = ("suite,F1,F2,F3,F4,overall", [",".join(r) for r in rows4])

    print("\n=== Table V: MS_strict per suite per criterion ===")
    print(f"{'Suite':10s}{'K-Struct':>10}{'K-Outcome':>11}{'K-Safety':>10}")
    rows5 = []
    for s in SUITES:
        vals = [score(agg, OPS, s, c) for c in CRITERIA]
        print(f"{s:10s}" + "".join(f"{fmt(v):>10}" for v in vals))
        rows5.append([s] + [f"{v:.3f}" for v in vals])
        for i, c in enumerate(CRITERIA):
            check(f"T5 {s} {c}", vals[i], PAPER_T5[s][i], fails)
    csv["table5"] = ("suite,kstruct,koutcome,ksafety", [",".join(r) for r in rows5])

    print("\n=== Table VI: K-Outcome MS_strict per operator per suite ===")
    print(f"{'Operator':14s}{'T-Dev*':>9}{'T-Rand*':>9}{'T-E2E*':>9}{'T-Struct':>9}")
    order = ("T-Dev*", "T-Rand*", "T-E2E*", "T-Struct")
    rows6 = []
    for op in OPS:
        vals = [score(agg, [op], s, "K-Outcome") for s in order]
        print(f"{op:14s}" + "".join(f"{fmt(v):>9}" for v in vals))
        rows6.append([op] + [f"{v:.3f}" for v in vals])
        for i, s in enumerate(order):
            check(f"T6 {op} {s}", vals[i], PAPER_T6[op][i], fails)
    csv["table6"] = ("operator,tdev,trand,te2e,tstruct", [",".join(r) for r in rows6])

    print(f"\nLive agent executions: {execs}  (paper: {PAPER_EXECUTIONS})")
    if execs != PAPER_EXECUTIONS:
        fails.append(f"executions: got {execs} expected {PAPER_EXECUTIONS}")

    if csv_dir:
        out = Path(csv_dir); out.mkdir(parents=True, exist_ok=True)
        for name, (h, lines) in csv.items():
            (out / f"{name}.csv").write_text(h + "\n" + "\n".join(lines) + "\n")
        print(f"\nCSVs written to {out}")

    print("\n" + "=" * 62)
    if fails:
        print(f"FAIL: {len(fails)} mismatch(es) vs. the published paper:")
        for f in fails[:50]:
            print("  -", f)
        return 1
    print("PASS: Tables II-VI, the 1,873 mutant count, and 29,601 executions")
    print("      all match the published paper, recomputed from source.")
    return 0


def run_live(bench_dir: Path):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY to run the live sweep (--live).")
    out = HERE / "results" / "live_rerun"
    print("Live sweep: 1,084 mutants, ~30k agent executions (hours, non-trivial cost).")
    subprocess.run(
        [sys.executable, str(HERE / "run_live_mutation.py"),
         "--benchmarks", *BENCHMARKS, "--out", str(out)],
        check=True,
    )
    print("\nVerifying the fresh live run against the paper...")
    return verify(bench_dir, out, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="offline verification (default)")
    ap.add_argument("--live", action="store_true", help="full live re-run, then verify (needs OPENAI_API_KEY)")
    ap.add_argument("--benchmarks", default=str(HERE / "benchmarks"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--csv", default=None, help="also write tables as CSV to this dir")
    args = ap.parse_args()
    if args.live:
        sys.exit(run_live(Path(args.benchmarks)))
    sys.exit(verify(Path(args.benchmarks), Path(args.results), args.csv))


if __name__ == "__main__":
    main()
