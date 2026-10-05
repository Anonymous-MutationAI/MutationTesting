#!/usr/bin/env python3
"""Revised live mutation sweep (v2) addressing the ICSE 2027 reviews.

Changes relative to run_live_mutation.py (v1, the submitted paper):

  1. Original-vs-original CONTROL (A1, B): every obligation is executed k extra
     times on the UNMUTATED spec and scored exactly like a mutant, giving the
     false-kill (noise) rate per criterion and per obligation.
  2. Infrastructure errors are RETRIED, never counted as kills (v1 counted any
     mutant-side exception as a K-Struct + K-Outcome kill).
     MaxTurnsExceeded is treated as a behavioural outcome (verdict = fail),
     with the partial trace kept.
  3. Runtime-identical mutants are SKIPPED: Ag-PromptDrop / Ag-Swap mutants
     whose effective instructions equal the original (placeholder hint
     '# System context' is ignored by the adapter). --drop-sigmut removes
     Tool-SigMut entirely.
  4. Per-obligation, per-run LOGGING to JSONL, so flaky-kill distributions,
     noise-adjusted scores, size-matched suites and audits can all be computed
     offline by analyze_v2.py without re-running agents.
  5. --subsample N for the k=10 stability check; --dry-run uses a stochastic
     fake executor to test the pipeline without an API key.

Kill rule (unchanged from v1, now documented): mutant run i is compared with
cached original run i; a criterion fires on an obligation when a strict
majority of the k pairs differ.

Usage:
  export OPENAI_API_KEY=sk-...
  python3 run_live_mutation_v2.py --benchmarks oai_customer_service --out logs/
  python3 analyze_v2.py logs/
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_live_mutation import (  # noqa: E402  (reuse v1 helpers unchanged)
    LiveTest, incident_obligations, template_for, to_live_manifest, _refused,
)
from agentmut import build_suites, generate_mutants, cap_per_operator  # noqa: E402

CRITERIA = ["K-Struct", "K-Outcome", "K-Safety"]
PLACEHOLDER_HINTS = {"", "# System context"}
INFRA_MARKERS = ("Timeout", "RateLimit", "Connection", "InternalServer",
                 "APIError", "APIStatus", "ServiceUnavailable", "Overloaded")


# ---------------------------------------------------------------------------
# Runtime-identical mutant filter
# ---------------------------------------------------------------------------
def _effective_hints(m: dict) -> dict:
    return {a["name"]: ("" if (a.get("system_prompt_hint") or "").strip() in PLACEHOLDER_HINTS
                        else (a.get("system_prompt_hint") or "").strip())
            for a in m["agents"]}


def runtime_identical(orig: dict, mut) -> bool:
    if mut.operator not in ("Ag-PromptDrop", "Ag-Swap"):
        return False
    return _effective_hints(orig) == _effective_hints(mut.manifest)


# ---------------------------------------------------------------------------
# Execution summaries (JSON-serialisable, so runs can be logged and reloaded)
# ---------------------------------------------------------------------------
def summarize(execution, max_turns_hit=False) -> dict:
    obs = execution.observation
    return {
        "tools": sorted(obs.tools_used),
        "denied": sorted(obs.denied_tools),
        "deleg": sorted([list(e) for e in set(map(tuple, execution.delegated_edges))]),
        "pairs": sorted([list(p) for p in (getattr(obs, "tool_calls", None) or set())]),
        "text": str(execution.final_output)[:2000],
        "max_turns": bool(max_turns_hit),
    }


def verdict_fail(s: dict) -> bool:
    return s["max_turns"] or _refused(s["text"])


def kills(o_sum: dict, m_sum: dict, restrict_pairs: set) -> dict:
    """Compare one original run with one mutant run (both summaries)."""
    struct = (set(o_sum["tools"]) != set(m_sum["tools"])
              or set(map(tuple, o_sum["deleg"])) != set(map(tuple, m_sum["deleg"]))
              or set(o_sum["denied"]) != set(m_sum["denied"]))
    outcome = struct or (verdict_fail(m_sum) and not verdict_fail(o_sum))
    safety = any(tuple(p) in restrict_pairs for p in m_sum["pairs"])
    return {"K-Struct": struct, "K-Outcome": outcome, "K-Safety": safety}


def majority(votes):
    return sum(1 for v in votes if v) * 2 > len(votes)


# ---------------------------------------------------------------------------
# Live executor (keeps the partial trace on MaxTurnsExceeded)
# ---------------------------------------------------------------------------
def make_live_executor(model):
    import agent_runtime.openai_agents_runtime as oar
    from agents import Runner
    from agent_runtime.graph_model import CoverageObservation
    from agent_runtime.live_runtime_backends import LocalRuntimeState

    def run(manifest, test):
        agents = oar.build_openai_agent_graph(manifest, model=model)
        if test.target_agent not in agents:
            raise LookupError(f"target agent '{test.target_agent}' not in graph")
        ctx = oar.FrameworkTrace(runtime_state=LocalRuntimeState())
        max_hit, items, final = False, [], ""
        try:
            result = Runner.run_sync(agents[test.target_agent], test.scenario,
                                     context=ctx, max_turns=6)
            items, final = result.new_items, str(result.final_output)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ != "MaxTurnsExceeded":
                raise
            max_hit = True
            rd = getattr(e, "run_data", None)
            items = list(getattr(rd, "new_items", []) or [])
            final = "MAX_TURNS_EXCEEDED"
        deleg = []
        for it in items:
            if getattr(it, "type", None) == "handoff_output_item":
                s = getattr(getattr(it, "source_agent", None), "name", None)
                t = getattr(getattr(it, "target_agent", None), "name", None)
                if s and t:
                    deleg.append((s, t))
        obs = CoverageObservation(
            target_agent=test.target_agent, mode=test.mode,
            tools_used=set(ctx.used_tools), denied_tools=set(ctx.denied_tools),
            skills_used=set(ctx.used_skills), delegated_edges=set(deleg),
            tool_calls=set(ctx.used_pairs))
        return summarize(SimpleNamespace(observation=obs, final_output=final,
                                         delegated_edges=deleg), max_hit)
    return run


def make_fake_executor(seed, noise):
    """Stochastic stand-in for --dry-run: follows the manifest, adds noise."""
    rng = random.Random(seed)

    def run(manifest, test):
        allow = {(e["agent"], e["tool"]) for e in manifest["permissions"].get("allow", [])}
        deny = {(e["agent"], e["tool"]) for e in manifest["permissions"].get("deny", [])}
        deleg = {(e["from"], e["to"]) for e in manifest.get("delegations", [])}
        a = test.target_agent
        o = test.obligation
        tools, denied, dl, pairs, text = [], [], [], [], "done"
        if o[0] == "use_tool" and (a, o[2]) in allow:
            tools, pairs = [o[2]], [[a, o[2]]]
        elif o[0] == "use_tool":
            text = "I cannot access that tool"
        if o[0] == "restrict_tool":
            if (a, o[2]) in allow:
                tools, pairs = [o[2]], [[a, o[2]]]
            elif (a, o[2]) in deny:
                denied = [o[2]]
        if o[0] == "delegate" and (a, o[2]) in deleg:
            dl = [[a, o[2]]]
        if rng.random() < noise:  # spurious extra handoff / tool
            own = sorted(t for (x, t) in allow if x == a)
            if own:
                t = rng.choice(own); tools = sorted(set(tools) | {t}); pairs.append([a, t])
        return {"tools": sorted(tools), "denied": denied, "deleg": dl,
                "pairs": pairs, "text": text, "max_turns": False}
    return run


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="default: all manifests in benchmarks/")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-per-op", type=int, default=50)
    ap.add_argument("--octagon-cap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--drop-sigmut", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--subsample", type=int, default=0,
                    help="score only N randomly chosen mutants per benchmark (k=10 check)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "v2"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-noise", type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    benches = args.benchmarks or sorted(p.stem for p in (ROOT / "benchmarks").glob("*.json"))

    if args.dry_run:
        raw_exec = make_fake_executor(args.seed, args.dry_noise)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY required (or use --dry-run).", file=sys.stderr)
            return 2
        raw_exec = make_live_executor(args.model)
        _install_live_backends()

    stats = defaultdict(int)

    def execute(manifest, test):
        """Retry infrastructure errors; return summary or None (unscored)."""
        for attempt in range(args.max_retries):
            try:
                stats["calls"] += 1
                return raw_exec(manifest, test)
            except LookupError:
                raise
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                infra = any(m in name for m in INFRA_MARKERS)
                stats["infra_errors" if infra else "other_errors"] += 1
                stats[f"err:{name}"] += 1
                time.sleep(min(30, 2 ** attempt))
        stats["unscored_runs"] += 1
        return None

    for bench in benches:
        m3 = json.load((ROOT / "benchmarks" / f"{bench}.json").open())
        live_orig = to_live_manifest(m3)
        if not args.dry_run:
            _register_tools(live_orig)
        restrict_pairs = {(e["agent"], e["tool"]) for e in m3["permissions"].get("restrict", [])}
        suites = build_suites(m3, random.Random(args.seed))
        full = suites["T-Struct"]
        cap = args.octagon_cap if bench == "octagon_vc_agents" else args.max_per_op
        mutants = cap_per_operator(generate_mutants(m3, bench), cap, random.Random(args.seed))
        # filters are applied AFTER capping so the sampled pool matches v1
        keep = []
        for i, mu in enumerate(mutants):
            if runtime_identical(m3, mu):
                stats["skipped_runtime_identical"] += 1; continue
            if args.drop_sigmut and mu.operator == "Tool-SigMut":
                stats["skipped_sigmut"] += 1; continue
            keep.append((i, mu))
        if args.subsample and len(keep) > args.subsample:
            keep = sorted(random.Random(args.seed + 1).sample(keep, args.subsample))

        log_path = out_dir / f"{bench}.jsonl"
        done_mut, orig_cache, control_done = set(), {}, set()
        if log_path.exists():  # resume
            for line in log_path.open():
                r = json.loads(line)
                if r["type"] == "orig":
                    orig_cache[tuple(r["obl"])] = r["runs"]
                elif r["type"] == "control":
                    control_done.add(tuple(r["obl"]))
                elif r["type"] == "mutant":
                    done_mut.add(r["idx"])
        log = log_path.open("a")

        def emit(rec):
            log.write(json.dumps(rec) + "\n"); log.flush()

        if not log_path.stat().st_size:
            emit({"type": "meta", "bench": bench, "k": args.k, "seed": args.seed,
                  "model": "fake" if args.dry_run else args.model,
                  "suites": {s: [list(o) for o in obs] for s, obs in suites.items()},
                  "restrict_pairs": sorted(map(list, restrict_pairs)),
                  "n_generated_capped": len(mutants), "n_scored": len(keep),
                  "drop_sigmut": args.drop_sigmut})

        realized = {}
        sp = ROOT / "suites" / f"{bench}.suite.json"
        if sp.exists():
            for t in json.load(sp.open()).get("tests", []):
                realized[tuple(t["objective_id"].split("::"))] = (t["target_agent"], t["user_prompt"])

        def get_test(o):
            t = LiveTest(*realized[o], o[0]) if o in realized else template_for(o)
            if t is not None:
                t.obligation = o
            return t

        def orig_runs(o):
            if o not in orig_cache:
                t = get_test(o)
                runs = [execute(live_orig, t) for _ in range(args.k)]
                orig_cache[o] = runs
                emit({"type": "orig", "obl": list(o), "runs": runs})
            return orig_cache[o]

        def score_against_orig(o, runs):
            o_runs = orig_runs(o)
            votes = {K: [] for K in CRITERIA}
            for i, mr in enumerate(runs):
                orr = o_runs[i % len(o_runs)]
                if mr is None or orr is None:
                    continue
                kk = kills(orr, mr, restrict_pairs)
                for K in CRITERIA:
                    votes[K].append(kk[K])
            return votes

        print(f"\n=== {bench}: {len(keep)} mutants scored "
              f"(of {len(mutants)} capped), {len(full)} obligations ===", flush=True)

        # ---- original-vs-original control -------------------------------
        if not args.no_control:
            for o in full:
                if o in control_done or get_test(o) is None:
                    continue
                t = get_test(o)
                ctrl = [execute(live_orig, t) for _ in range(args.k)]
                votes = score_against_orig(o, ctrl)
                emit({"type": "control", "obl": list(o), "runs": ctrl, "votes": votes,
                      "fires": {K: (majority(v) if v else None) for K, v in votes.items()}})

        # ---- mutants ------------------------------------------------------
        for idx, mu in keep:
            if idx in done_mut:
                continue
            incident = incident_obligations(m3, mu.manifest, full)
            live_mut = to_live_manifest(mu.manifest)
            if not args.dry_run:
                _register_tools(live_mut)
            faults = {f["tool"]: f.get("mode", "fail:error")
                      for f in mu.manifest.get("_fault_injection", [])}
            mut_agents = {a["name"] for a in mu.manifest["agents"]}
            per_obl = []
            for o in incident:
                t = get_test(o)
                if t is None:
                    continue
                if t.target_agent not in mut_agents:
                    per_obl.append({"obl": list(o), "structural_fail": "target_removed",
                                    "fires": {"K-Struct": True, "K-Outcome": True, "K-Safety": False}})
                    continue
                if not args.dry_run:
                    _set_faults(faults)
                try:
                    mruns = [execute(live_mut, t) for _ in range(args.k)]
                except LookupError:
                    per_obl.append({"obl": list(o), "structural_fail": "target_unreachable",
                                    "fires": {"K-Struct": True, "K-Outcome": True, "K-Safety": False}})
                    continue
                finally:
                    if not args.dry_run:
                        _set_faults({})
                votes = score_against_orig(o, mruns)
                per_obl.append({"obl": list(o), "runs": mruns, "votes": votes,
                                "fires": {K: (majority(v) if v else None) for K, v in votes.items()}})
            emit({"type": "mutant", "idx": idx, "operator": mu.operator,
                  "target": str(mu.target), "incident": [list(o) for o in incident],
                  "per_obl": per_obl})
            w = {K: sum(1 for p in per_obl if p["fires"].get(K)) for K in CRITERIA}
            print(f"  {mu.operator:<13} {str(mu.target)[:32]:<32} incident={len(incident):>3} "
                  f"witness S/O/Sf={w['K-Struct']}/{w['K-Outcome']}/{w['K-Safety']} "
                  f"calls={stats['calls']}", flush=True)
        log.close()

    stats_path = out_dir / "run_stats.json"
    prev = json.load(stats_path.open()) if stats_path.exists() else {}
    for k, v in stats.items():
        prev[k] = prev.get(k, 0) + v
    json.dump(prev, stats_path.open("w"), indent=2)
    print(f"\nrun stats: {dict(stats)}\nlogs in {out_dir}/  ->  python3 analyze_v2.py {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Live tool backends (same behaviour as v1: real local binding or injected fault)
# ---------------------------------------------------------------------------
_FAULTS: dict = {}
_BINDINGS: dict = {}


def _backend(state, tool_name, agent_name):
    mode = _FAULTS.get(tool_name)
    if mode == "fail:error":
        return f"ERROR: tool '{tool_name}' failed with an injected runtime error."
    if mode == "fail:malformed":
        return "<<<malformed>>> not-valid-json payload from tool"
    real = _BINDINGS.get(tool_name)
    if real is not None:
        try:
            return real(state, tool_name, agent_name)
        except Exception as e:  # noqa: BLE001
            return f"ok: tool '{tool_name}' (real binding error: {type(e).__name__})"
    return f"ok: tool '{tool_name}' executed by agent '{agent_name}'."


def _register_tools(manifest):
    from agent_runtime.live_runtime_backends import TOOL_BACKENDS
    for t in manifest.get("tools", []):
        b = t.get("binding") or {}
        if t["name"] not in _BINDINGS and b.get("module") and b.get("callable"):
            try:
                _BINDINGS[t["name"]] = getattr(importlib.import_module(b["module"]), b["callable"])
            except Exception:  # noqa: BLE001
                _BINDINGS[t["name"]] = None
        TOOL_BACKENDS[t["name"]] = _backend


def _set_faults(f):
    _FAULTS.clear(); _FAULTS.update(f)


def _install_live_backends():
    """Same instruction wrapper as v1 (role + description + non-placeholder hint)."""
    import agent_runtime.openai_agents_runtime as oar
    orig = oar._instructions_for_agent

    def wrapped(manifest, agent_name):
        base = orig(manifest, agent_name)
        meta = next((a for a in manifest.get("agents", []) if a["name"] == agent_name), {})
        pre = f"Your role: {meta.get('role','')}. {meta.get('description','')}".strip()
        hint = (meta.get("system_prompt_hint") or "").strip()
        if hint and hint not in PLACEHOLDER_HINTS:
            pre += "\n" + hint
        return (pre + "\n\n" + base) if pre else base
    oar._instructions_for_agent = wrapped


if __name__ == "__main__":
    raise SystemExit(main())
