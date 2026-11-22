#!/usr/bin/env python3
"""Extra original-vs-original control repeats for the v2 sweep.

Re-runs the unmutated workflow on every obligation `--reps` more times
(k runs each), compares each run with the cached original runs from
results/v2/<bench>.jsonl, and logs one record per (obligation, rep) to
results/v2_control/<bench>.jsonl. Resumable. Existing v2 logs are only read.

  python run_control_extra.py --reps 2                 # all benchmarks
  python run_control_extra.py --reps 2 --benchmarks oai_customer_service
  python run_control_extra.py --dry-run --reps 2       # no API calls
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from run_live_mutation import LiveTest, template_for, to_live_manifest  # noqa: E402
import run_live_mutation_v2 as v2  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--src", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v2_control"))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    benches = a.benchmarks or sorted(p.stem for p in Path(a.src).glob("*.jsonl"))
    if a.dry_run:
        raw = v2.make_fake_executor(7, 0.15)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("ERROR: set OPENAI_API_KEY first (or use --dry-run).")
        raw = v2.make_live_executor(a.model); v2._install_live_backends()
    stats = defaultdict(int)
    def execute(man, t):
        for att in range(a.max_retries):
            try:
                stats["calls"] += 1; return raw(man, t)
            except Exception as e:  # noqa: BLE001
                stats["err:" + type(e).__name__] += 1
                if "insufficient_quota" in str(e) or type(e).__name__ == "AuthenticationError":
                    sys.exit(f"STOP: {type(e).__name__}: {e}")
                time.sleep(min(30, 2 ** att))
        stats["unscored"] += 1; return None
    for bench in benches:
        m3 = json.load((ROOT / "benchmarks" / f"{bench}.json").open())
        live = to_live_manifest(m3)
        if not a.dry_run: v2._register_tools(live)
        restrict = {(e["agent"], e["tool"]) for e in m3["permissions"].get("restrict", [])}
        orig, k = {}, 3
        for line in open(Path(a.src) / f"{bench}.jsonl"):
            r = json.loads(line)
            if r["type"] == "meta": k = r["k"]
            if r["type"] == "orig": orig[tuple(r["obl"])] = r["runs"]
        realized = {}
        sp = ROOT / "suites" / f"{bench}.suite.json"
        if sp.exists():
            for t in json.load(sp.open()).get("tests", []):
                realized[tuple(t["objective_id"].split("::"))] = (t["target_agent"], t["user_prompt"])
        lp = out / f"{bench}.jsonl"; done = set()
        if lp.exists():
            for line in lp.open():
                r = json.loads(line)
                if all(x is not None for x in r["runs"]): done.add((tuple(r["obl"]), r["rep"]))
        log = lp.open("a")
        print(f"=== {bench}: {len(orig)} obligations x {a.reps} reps", flush=True)
        for o, oruns in orig.items():
            t = LiveTest(*realized[o], o[0]) if o in realized else template_for(o)
            if t is None: continue
            t.obligation = o
            for rep in range(1, a.reps + 1):
                if (o, rep) in done: continue
                runs = [execute(live, t) for _ in range(k)]
                votes = {K: [] for K in v2.CRITERIA}
                for i, mr in enumerate(runs):
                    orr = oruns[i % len(oruns)]
                    if mr is None or orr is None: continue
                    kk = v2.kills(orr, mr, restrict)
                    for K in v2.CRITERIA: votes[K].append(kk[K])
                log.write(json.dumps({"type": "control", "rep": rep, "obl": list(o), "runs": runs, "votes": votes,
                    "fires": {K: (v2.majority(v) if v else None) for K, v in votes.items()}}) + "\n"); log.flush()
        log.close()
        print(f"    done {bench}; calls so far {stats['calls']}", flush=True)
    print("stats:", dict(stats))

if __name__ == "__main__":
    main()
