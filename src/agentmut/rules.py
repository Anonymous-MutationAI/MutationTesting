"""Pool and kill rules shared by the live runner and the offline analysis.

Three rules decide which mutants are scored and which runs may witness a kill.
The live runner (run_live_mutation_v2.py) applies them while executing, and
analyze_v2.load() applies them to existing logs, so both paths agree.

1. Runtime duplicates. Two mutants are duplicates when their effective runtime
   specification is identical: the same agents and (non-placeholder)
   instruction hints, tools, allow and restrict pairs, delegation edges,
   injected faults and entry agent. The adapter builds each agent's tools and
   handoffs from sets, so, for example, redirecting a handoff to a child the
   parent already delegates to (Deleg-Red) behaves exactly like removing that
   handoff (Deleg-Rem). Only one mutant per specification is scored; the
   representative is chosen by OPERATOR_PRIORITY, then by generation order.

2. Executed witnesses only. A test whose target agent no longer exists in the
   mutant (Ag-Rem) cannot be executed. It is logged as a structural failure
   but does not count as a kill: in a real suite the test would enter at the
   workflow's entry agent, so the failure is an artifact of starting each test
   at its target agent.

3. Fault reachability (Tool-Fault). A Tool-Fault mutant behaves exactly like
   the original on any run that does not call the faulted tool. A run pair can
   therefore vote for a kill only when the mutant run calls the faulted tool
   (the reachability condition of the RIP model of mutation analysis).

The legacy rules (no deduplication, structural failures counted as kills, no
reachability condition) reproduce the numbers of the first submission.
"""

from __future__ import annotations

import json

PLACEHOLDER_HINTS = {"", "# System context"}

# Representative for a group of runtime duplicates: the most direct operator.
OPERATOR_PRIORITY = {op: i for i, op in enumerate([
    "Del-Rem", "Tool-Rem", "Res-Tighten", "Res-Rem", "Ag-Rem", "Del-Inv", "Del-Add",
    "Del-Red", "Tool-Red", "Res-Relax", "Tool-SigMut", "Tool-Fault", "Ag-Swap",
    "Ag-PromptDrop"])}


def _hint(agent: dict) -> str:
    h = (agent.get("system_prompt_hint") or "").strip()
    return "" if h in PLACEHOLDER_HINTS else h


def runtime_signature(m: dict) -> str:
    """Canonical description of what the runtime adapter actually executes."""
    perms = m.get("permissions", {})
    return json.dumps([
        sorted((a["name"], _hint(a)) for a in m["agents"]),
        sorted(t["name"] for t in m["tools"]),
        sorted({(e["agent"], e["tool"]) for e in perms.get("allow", [])}),
        sorted({(e["agent"], e["tool"]) for e in perms.get("restrict", [])}),
        sorted({(e["from"], e["to"]) for e in m.get("delegations", [])}),
        sorted((f["tool"], f.get("mode", "")) for f in m.get("_fault_injection", [])),
        m["system"]["entry_agent"],
    ])


def runtime_identical_to_original(orig: dict, mut) -> bool:
    """Mutant whose runtime specification equals the original's (never executed)."""
    return runtime_signature(orig) == runtime_signature(mut.manifest)


def drop_runtime_duplicates(items):
    """items: list of (idx, mutant). Returns (kept, dropped), both lists of (idx, mutant).

    One representative per runtime signature is kept, chosen by
    OPERATOR_PRIORITY and then by idx; the kept list preserves input order.
    """
    best = {}
    for idx, mu in items:
        sig = runtime_signature(mu.manifest)
        key = (OPERATOR_PRIORITY.get(mu.operator, 99), idx)
        if sig not in best or key < best[sig][0]:
            best[sig] = (key, idx)
    keep_idx = {idx for _, idx in best.values()}
    kept = [(i, mu) for i, mu in items if i in keep_idx]
    dropped = [(i, mu) for i, mu in items if i not in keep_idx]
    return kept, dropped


def well_formedness_violations(m: dict) -> list:
    """Invariants (i)-(iv) of Section 3.2; returns the violated ones (empty if well formed)."""
    agents = {a["name"] for a in m["agents"]}
    tools = [t["name"] for t in m["tools"]]
    perms = m.get("permissions", {})
    allow = {(e["agent"], e["tool"]) for e in perms.get("allow", [])}
    restrict = {(e["agent"], e["tool"]) for e in perms.get("restrict", [])}
    deleg = {(e["from"], e["to"]) for e in m.get("delegations", [])}
    bad = []
    if not (all(a in agents and t in tools for a, t in allow | restrict)
            and all(x in agents and y in agents for x, y in deleg)):
        bad.append("i: edge endpoint not declared")
    if allow & restrict:
        bad.append("ii: pair both allowed and restricted")
    if m["system"]["entry_agent"] not in agents:
        bad.append("iii: entry agent missing")
    if len(set(tools)) != len(tools):
        bad.append("iv: tool names not unique")
    return bad


def faulted_tool(operator: str, target: str):
    """Name of the faulted tool for a Tool-Fault mutant, else None."""
    return str(target).split("@")[0] if operator == "Tool-Fault" else None


def fault_reached(tool, mutant_run) -> bool:
    """Reachability: True unless this is a Tool-Fault mutant run that never calls the tool."""
    return tool is None or (mutant_run is not None and tool in (mutant_run.get("tools") or []))


def majority(votes) -> bool:
    return sum(1 for v in votes if v) * 2 > len(votes)


def rescored_votes(mut: dict, p: dict) -> dict:
    """Per-pair votes of one logged obligation with the reachability condition applied."""
    votes = p.get("votes") or {}
    tool = faulted_tool(mut["operator"], mut["target"])
    if tool is None:
        return {K: list(v) for K, v in votes.items()}
    runs = p.get("runs") or []
    out = {}
    for K, v in votes.items():
        # votes are logged only for pairs where both runs exist; the logs
        # contain no missing runs, so votes and mutant runs align by index.
        if len(v) != len(runs):
            raise ValueError(f"vote/run misalignment for {mut['operator']} {mut['target']}")
        out[K] = [x and fault_reached(tool, r) for x, r in zip(v, runs)]
    return out


def rescore_obligation(mut: dict, p: dict) -> dict:
    """Recompute one logged obligation's per-criterion firing under the rules."""
    if "structural_fail" in p:
        return {"K-Struct": False, "K-Outcome": False, "K-Safety": False}
    return {K: (majority(v) if v else None) for K, v in rescored_votes(mut, p).items()}
