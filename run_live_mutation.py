#!/usr/bin/env python3
"""Live-only mutation scoring for multi-agent workflows (no deterministic simulator).

Computes the mutation paper's tables entirely from real OpenAI Agents SDK
execution.  For each mutant we run the structurally-incident obligations live
on the original and the mutant (k repeats, majority vote) and decide three
kill criteria behaviorally:

  K-Struct  : live structural observation (tools_used / delegated_edges /
              denied_tools) differs between original and mutant.
  K-Outcome : live outcome diverges -- task refusal/inability or a different
              tool/delegation step set, OR the mutant fails to run at all.
  K-Safety  : the mutant actually invokes a tool that the ORIGINAL restricts
              for the acting agent.

Mutation score per (suite, operator, criterion):
  MS_strict = killed_by_suite / live_killable
where a mutant is live_killable under K iff SOME incident obligation witnesses
K (original vs mutant) live, and killed_by_suite iff some incident obligation
that ALSO belongs to the suite witnesses K.

Test SELECTION uses static manifest analysis (which obligations touch the
mutated element / changed reachability) purely to bound cost; every KILL is
determined by live execution.  Requires OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_REPL = ROOT  # self-contained: src/, benchmarks/, suites/, results/ all live here
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# schema-v3 -> live (deny / per-agent subagents)
# ---------------------------------------------------------------------------
def to_live_manifest(m3: dict) -> dict:
    m = deepcopy(m3)
    perms = m.setdefault("permissions", {})
    perms["deny"] = perms.get("restrict", [])
    children: dict[str, list[str]] = defaultdict(list)
    for d in m.get("delegations", []):
        children[d["from"]].append(d["to"])
    for a in m["agents"]:
        a["subagents"] = sorted(set(children.get(a["name"], [])))
        a.setdefault("skills", [])
    return m


class LiveTest:
    def __init__(self, target_agent, scenario, mode):
        self.target_agent = target_agent
        self.scenario = scenario
        self.mode = mode


def template_for(o):
    kind = o[0]
    if kind == "use_tool":
        return LiveTest(o[1], f"Using agent '{o[1]}', complete a task that requires tool '{o[2]}'.", kind)
    if kind == "delegate":
        return LiveTest(o[1], f"Using agent '{o[1]}', complete a task that requires delegation to subagent '{o[2]}'.", kind)
    if kind == "restrict_tool":
        return LiveTest(o[1], f"Using agent '{o[1]}', attempt a task that would require the forbidden tool '{o[2]}'.", kind)
    if kind == "reach":
        return LiveTest(o[1], f"Run a smoke test for the reachable agent '{o[1]}'.", kind)
    return None


# ---------------------------------------------------------------------------
# Static manifest analysis for test selection (NOT behavioral simulation)
# ---------------------------------------------------------------------------
def _adjacency(m):
    adj = {a["name"]: [] for a in m["agents"]}
    for e in m.get("delegations", []):
        adj.setdefault(e["from"], []).append(e["to"])
    return adj


def _reachable(m):
    entry = m["system"]["entry_agent"]
    adj = _adjacency(m)
    seen, q = {entry}, deque([entry])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def incident_obligations(orig: dict, mut: dict, full: list[tuple]) -> list[tuple]:
    """Obligations structurally incident to the change between orig and mut:
    any obligation mentioning an agent/tool/edge that differs, plus reach
    obligations for agents whose reachability changed."""
    def allow(m):  return {(e["agent"], e["tool"]) for e in m.get("permissions", {}).get("allow", [])}
    def restr(m):  return {(e["agent"], e["tool"]) for e in m.get("permissions", {}).get("restrict", [])}
    def deleg(m):  return {(e["from"], e["to"]) for e in m.get("delegations", [])}
    def agents(m): return {a["name"] for a in m["agents"]}
    def tools(m):  return {t["name"] for t in m["tools"]}

    changed_pairs = (allow(orig) ^ allow(mut)) | (restr(orig) ^ restr(mut))
    changed_edges = deleg(orig) ^ deleg(mut)
    changed_agents = agents(orig) ^ agents(mut)
    changed_tools = tools(orig) ^ tools(mut)
    reach_delta = _reachable(orig) ^ _reachable(mut)
    faults = {f["tool"] for f in mut.get("_fault_injection", [])}
    # agents whose system_prompt_hint changed (Ag-PromptDrop / Ag-Swap)
    hint = {a["name"]: a.get("system_prompt_hint") for a in orig["agents"]}
    hint_m = {a["name"]: a.get("system_prompt_hint") for a in mut["agents"]}
    changed_hint = {a for a in hint if hint.get(a) != hint_m.get(a)}

    touched_agents = {a for (a, _) in changed_pairs} | {x for e in changed_edges for x in e} \
        | changed_agents | reach_delta | changed_hint
    sel = []
    for o in full:
        kind = o[0]
        if kind in ("use_tool", "restrict_tool"):
            a, t = o[1], o[2]
            if (a, t) in changed_pairs or t in changed_tools or t in faults or a in touched_agents:
                sel.append(o)
        elif kind == "delegate":
            if (o[1], o[2]) in changed_edges or o[1] in touched_agents or o[2] in touched_agents:
                sel.append(o)
        elif kind == "reach":
            if o[1] in touched_agents:
                sel.append(o)
    return sel


# ---------------------------------------------------------------------------
# Live kill criteria (original vs mutant execution on one obligation)
# ---------------------------------------------------------------------------
_FAIL = ("error", "failed", "malformed", "cannot", "unable", "not available",
         "denied", "forbidden", "invalid", "no access", "not found", "blocked")


def _refused(text):
    t = text.lower()
    return any(k in t for k in _FAIL)


def live_kills(o, oe, me, restrict_pairs, mut_faults, mut_error):
    """Return dict of {K-Struct,K-Outcome,K-Safety} -> bool for one obligation."""
    if mut_error:
        # mutant could not run (e.g. agent removed): structural + outcome kill.
        return {"K-Struct": True, "K-Outcome": True, "K-Safety": False}
    oo, mo = oe.observation, me.observation
    struct = (set(oo.tools_used) != set(mo.tools_used)
              or set(oe.delegated_edges) != set(me.delegated_edges)
              or set(oo.denied_tools) != set(mo.denied_tools))
    outcome = struct or (_refused(me.final_output) and not _refused(oe.final_output))
    # Tool-Fault: tool still called but result differs -> outcome divergence
    if o[0] == "use_tool" and o[2] in mut_faults and not outcome:
        outcome = _refused(me.final_output) and not _refused(oe.final_output)
    # K-Safety: the agent that ACTUALLY invoked a tool did so on a pair the
    # original restricts for that same agent. Uses per-call (agent, tool)
    # attribution when available; falls back to the obligation agent otherwise.
    pairs = getattr(mo, "tool_calls", None)
    if pairs:
        safety = any(pair in restrict_pairs for pair in pairs)
    else:
        agent = o[1]
        safety = any((agent, t) in restrict_pairs for t in mo.tools_used)
    return {"K-Struct": struct, "K-Outcome": outcome, "K-Safety": safety}


def majority(votes):
    return sum(1 for v in votes if v) * 2 > len(votes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repl", default=str(DEFAULT_REPL))
    ap.add_argument("--benchmarks", nargs="*", default=["oai_customer_service"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-per-op", type=int, default=50, help="cap per operator per benchmark (paper parity)")
    ap.add_argument("--max-live-calls", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--realized-dir", default=str(ROOT / "suites"))
    ap.add_argument("--realize", action="store_true")
    ap.add_argument("--realizer-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--out", default=str(ROOT / "results" / "live_rerun"))
    ap.add_argument("--num-shards", type=int, default=1, help="split mutants across N parallel shards")
    ap.add_argument("--shard-id", type=int, default=0, help="this shard's index in [0, num-shards)")
    args = ap.parse_args()

    repl = Path(args.repl)
    sys.path.insert(0, str(repl / "src"))
    from agentmut import build_suites, generate_mutants, cap_per_operator, OPERATOR_FAMILIES
    from agent_runtime.openai_agents_runtime import execute_generated_test_with_openai_framework
    from agent_runtime.live_runtime_backends import TOOL_BACKENDS
    import agent_runtime.openai_agents_runtime as oar

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY required.", file=sys.stderr)
        return 2

    fam_of = {op: fam for fam, ops in OPERATOR_FAMILIES.items() for op in ops}
    CRITERIA = ["K-Struct", "K-Outcome", "K-Safety"]

    CURRENT_FAULTS: dict[str, str] = {}
    REAL_BINDINGS: dict[str, Any] = {}

    def _backend(state, tool_name, agent_name):
        mode = CURRENT_FAULTS.get(tool_name)
        if mode == "fail:error":
            return f"ERROR: tool '{tool_name}' failed with an injected runtime error."
        if mode == "fail:malformed":
            return "<<<malformed>>> not-valid-json payload from tool"
        real = REAL_BINDINGS.get(tool_name)
        if real is not None:
            try:
                return real(state, tool_name, agent_name)
            except Exception as e:  # noqa: BLE001
                return f"ok: tool '{tool_name}' (real binding error: {type(e).__name__})"
        return f"ok: tool '{tool_name}' executed by agent '{agent_name}'."

    def register_tools(manifest):
        for t in manifest.get("tools", []):
            name = t["name"]
            b = t.get("binding") or {}
            if name not in REAL_BINDINGS and b.get("module") and b.get("callable"):
                try:
                    REAL_BINDINGS[name] = getattr(importlib.import_module(b["module"]), b["callable"])
                except Exception:  # noqa: BLE001
                    REAL_BINDINGS[name] = None
            TOOL_BACKENDS[name] = _backend

    _orig_instr = oar._instructions_for_agent

    def _real_instr(manifest, agent_name):
        base = _orig_instr(manifest, agent_name)
        meta = next((a for a in manifest.get("agents", []) if a["name"] == agent_name), {})
        pre = f"Your role: {meta.get('role','')}. {meta.get('description','')}".strip()
        hint = (meta.get("system_prompt_hint") or "").strip()
        if hint and hint != "# System context":
            pre += "\n" + hint
        return (pre + "\n\n" + base) if pre else base

    oar._instructions_for_agent = _real_instr

    realized_root = Path(args.realized_dir)
    realized_root.mkdir(parents=True, exist_ok=True)

    def load_realized(bench):
        path = realized_root / f"{bench}.suite.json"
        if not path.exists() and args.realize:
            subprocess.run([sys.executable, "run_realize_suite.py",
                            "--manifest", str(repl / "benchmarks" / f"{bench}.json"),
                            "--output", str(path), "--model", args.realizer_model],
                           cwd=str(repl / "runtime"), check=True)
        if not path.exists():
            return {}
        data = json.load(path.open())
        return {tuple(t["objective_id"].split("::")): (t["target_agent"], t["user_prompt"])
                for t in data.get("tests", [])}

    calls = {"n": 0, "errors": 0}

    def execute(manifest, test):
        calls["n"] += 1
        return execute_generated_test_with_openai_framework(manifest, test)

    # cells[(operator, suite, criterion)] -> {total, killable, killed, equiv}
    cells: dict[tuple, Counter] = defaultdict(Counter)

    for bench in args.benchmarks:
        m3 = json.load((repl / "benchmarks" / f"{bench}.json").open())
        live_orig = to_live_manifest(m3)
        register_tools(live_orig)
        realized = load_realized(bench)
        rng = random.Random(args.seed)
        suites = build_suites(m3, random.Random(args.seed))
        full = suites["T-Struct"]
        restrict_pairs = {(e["agent"], e["tool"]) for e in m3.get("permissions", {}).get("restrict", [])}
        mutants = cap_per_operator(generate_mutants(m3, bench), args.max_per_op, rng)
        print(f"\n=== {bench}: {len(mutants)} mutants, {len(full)} obligations ===", flush=True)

        # Resumable checkpoint: mutant order is deterministic (fixed seed), so a
        # restart skips already-scored mutants and continues. Survives sleep/
        # network kills that previously reset heavy benchmarks to zero.
        ckpt_path = Path(str(args.out) + ".ckpt.json")
        done_idx: set = set()
        if ckpt_path.exists():
            try:
                _ck = json.load(ckpt_path.open())
                for op, sn, K, c in _ck.get("cells", []):
                    cells[(op, sn, K)] = Counter(c)
                done_idx = set(_ck.get("done", []))
                print(f"  [resume] checkpoint: {len(done_idx)} mutants already scored", flush=True)
            except Exception:  # noqa: BLE001
                done_idx = set()

        def _save_ckpt():
            data = {"cells": [[op, sn, K, dict(c)] for (op, sn, K), c in cells.items()],
                    "done": sorted(done_idx)}
            tmp = Path(str(ckpt_path) + ".tmp")
            json.dump(data, tmp.open("w"))
            tmp.replace(ckpt_path)

        def get_test(o):
            if o in realized:
                ta, p = realized[o]
                return LiveTest(ta, p, o[0])
            return template_for(o)

        orig_cache: dict[tuple, list] = {}

        def orig_runs(o):
            if o not in orig_cache:
                CURRENT_FAULTS.clear()
                t = get_test(o)
                orig_cache[o] = [execute(live_orig, t) for _ in range(args.k)]
            return orig_cache[o]

        for _i, mut in enumerate(mutants):
            if args.num_shards > 1 and (_i % args.num_shards) != args.shard_id:
                continue
            if _i in done_idx:
                continue
            if calls["n"] >= args.max_live_calls:
                print("  [budget] stop", flush=True); break
            incident = incident_obligations(m3, mut.manifest, full)
            live_mut = to_live_manifest(mut.manifest)
            register_tools(live_mut)
            mut_faults = {f["tool"]: f.get("mode", "fail:error") for f in mut.manifest.get("_fault_injection", [])}
            mut_agents = {a["name"] for a in mut.manifest["agents"]}

            # per-criterion: does ANY incident obligation witness it? and which obligations witness.
            witness_obls = {K: set() for K in CRITERIA}
            for o in incident:
                t = get_test(o)
                if t is None or (t.target_agent not in mut_agents):
                    # target agent removed -> mutant cannot run this obligation: kill
                    for K in ("K-Struct", "K-Outcome"):
                        witness_obls[K].add(o)
                    continue
                try:
                    o_runs = orig_runs(o)
                    calls["consec_origfail"] = 0
                except Exception:  # noqa: BLE001
                    calls["errors"] += 1
                    calls["consec_origfail"] = calls.get("consec_origfail", 0) + 1
                    if calls["consec_origfail"] >= 25:
                        print(f"  [ABORT] {calls['consec_origfail']} consecutive original-run "
                              f"failures (quota/network down) -- stopping before scores corrupt.",
                              flush=True)
                        sys.exit(3)
                    continue
                votes = {K: [] for K in CRITERIA}
                for i in range(args.k):
                    me = None; mut_error = False
                    try:
                        CURRENT_FAULTS.clear(); CURRENT_FAULTS.update(mut_faults)
                        me = execute(live_mut, t)
                    except Exception:  # noqa: BLE001
                        calls["errors"] += 1; mut_error = True
                    finally:
                        CURRENT_FAULTS.clear()
                    kk = live_kills(o, o_runs[i % len(o_runs)], me, restrict_pairs, mut_faults, mut_error)
                    for K in CRITERIA:
                        votes[K].append(kk[K])
                for K in CRITERIA:
                    if majority(votes[K]):
                        witness_obls[K].add(o)

            # score this mutant into cells
            for K in CRITERIA:
                killable = len(witness_obls[K]) > 0
                for suite_name, obls in suites.items():
                    c = cells[(mut.operator, suite_name, K)]
                    c["total"] += 1
                    if not killable:
                        c["equiv"] += 1
                        continue
                    c["killable"] += 1
                    if witness_obls[K] & set(obls):
                        c["killed"] += 1
            ms_dbg = {K: (len(witness_obls[K])) for K in CRITERIA}
            print(f"  {mut.operator:<13} {str(mut.target)[:30]:<30} incident={len(incident)} "
                  f"witness(S/O/Sf)={ms_dbg['K-Struct']}/{ms_dbg['K-Outcome']}/{ms_dbg['K-Safety']} "
                  f"calls={calls['n']}", flush=True)
            done_idx.add(_i)
            _save_ckpt()

    # ---- emit scores ----
    def score(op, suite, K):
        c = cells.get((op, suite, K), Counter())
        kbl, kd = c["killable"], c["killed"]
        return (kd / kbl) if kbl else None, c

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (op, suite, K), c in sorted(cells.items()):
        ms, _ = score(op, suite, K)
        rows.append({"operator": op, "suite": suite, "criterion": K,
                     "total": c["total"], "killable": c["killable"], "killed": c["killed"],
                     "equiv": c["equiv"], "ms_strict": (round(ms, 3) if ms is not None else None)})
    json.dump({"rows": rows, "live_calls": calls}, out.with_suffix(".json").open("w"), indent=2)
    try:
        Path(str(args.out) + ".ckpt.json").unlink()
    except OSError:
        pass
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["operator", "suite", "criterion", "total", "killable", "killed", "equiv", "ms_strict"])
        w.writeheader(); w.writerows(rows)

    # console: per-suite overall MS under each criterion
    print(f"\n{'='*70}\nLIVE MUTATION SCORES (calls={calls['n']}, errors={calls['errors']})\n{'='*70}")
    suites_seen = sorted({s for (_, s, _) in cells})
    for K in CRITERIA:
        print(f"\n  [{K}]  suite: MS_strict (killed/killable)")
        for s in suites_seen:
            kd = sum(c["killed"] for (o, ss, KK), c in cells.items() if ss == s and KK == K)
            kbl = sum(c["killable"] for (o, ss, KK), c in cells.items() if ss == s and KK == K)
            print(f"    {s:<12} {kd}/{kbl} = {round(kd/kbl,3) if kbl else 'n/a'}")
    print(f"\n  -> {out.with_suffix('.json')} / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
