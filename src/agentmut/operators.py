"""Mutation operators for schema-v3 workflow manifests.

Every operator is a pure function ``dict -> list[dict_mutated]``. The input
manifest is never mutated in place; each variant is produced via a deep
copy. Operators are deterministic and total: given the same manifest, the
same list is returned in the same order.

Operator catalogue (12):

  Family F1 — Delegation faults
    Del-Rem   : drop one delegation edge
    Del-Red   : redirect (parent, *) to a different existing reachable child
    Del-Inv   : invert direction of one edge (parent <-> child swap)
    Del-Add   : add a delegation edge that did not previously exist

  Family F2 — Tool wiring faults
    Tool-Rem  : drop one allow edge
    Tool-Red  : redirect an allow edge to a different existing tool
    Tool-Fault: do not change the graph; mark a tool as fault-injected
                (consumed by the runtime adapter; K-Struct treats Tool-Fault
                as non-structural and the equivalence filter will mark it
                equivalent under K-Struct)
    Tool-SigMut: rename one tool symbol to a near-synonym
                (graph-equivalent but exercises the realizer/runtime
                binding; K-Struct will not kill — equivalent under K-Struct)

  Family F3 — Restriction faults
    Res-Relax  : move one restrict edge into allow
    Res-Tighten: move one allow edge into restrict
    Res-Rem    : drop one restrict edge (silently revoked restriction)

  Family F4 — Agent / instruction faults
    Ag-Rem        : drop one non-entry reachable agent
    Ag-Swap       : swap the names of two non-entry reachable agents
    Ag-PromptDrop : clear the system_prompt_hint of one reachable agent
                    (graph-equivalent under K-Struct)

Operators that produce graph-equivalent mutants (Tool-Fault, Tool-SigMut,
Ag-PromptDrop) are still emitted: they are intended to be killed by
K-Outcome and K-Safety, not by K-Struct. The equivalence filter applied
under K-Struct will mark them equivalent and the runner will record the
distinction.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Manifest = dict


@dataclass(frozen=True)
class MutationOperator:
    """A named mutation operator over a manifest."""

    name: str
    family: str   # "F1_delegation" / "F2_tool" / "F3_restrict" / "F4_agent"
    apply: Callable[[Manifest], list[tuple[Manifest, str]]]
    """Returns a list of (mutated_manifest, target_description) pairs."""


@dataclass(frozen=True)
class Mutant:
    """A single mutant produced by an operator."""

    mutant_id: str
    operator: str
    family: str
    target: str
    manifest: Manifest = field(repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reachable_agents(m: Manifest) -> list[str]:
    """Return reachable agent names by BFS from system.entry_agent."""
    agents = {a["name"]: a for a in m["agents"]}
    entry = m["system"]["entry_agent"]
    adjacency: dict[str, list[str]] = {n: [] for n in agents}
    for e in m["delegations"]:
        adjacency.setdefault(e["from"], []).append(e["to"])
    seen: set[str] = set()
    stack = [entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adjacency.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)
    return sorted(n for n in seen if agents[n].get("reachable", True))


def _copy(m: Manifest) -> Manifest:
    return copy.deepcopy(m)


def _allow(m: Manifest) -> list[dict]:
    return m.get("permissions", {}).get("allow", [])


def _restrict(m: Manifest) -> list[dict]:
    return m.get("permissions", {}).get("restrict", [])


# ---------------------------------------------------------------------------
# F1 — Delegation operators
# ---------------------------------------------------------------------------

def _op_del_rem(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    for i, edge in enumerate(m["delegations"]):
        mm = _copy(m)
        mm["delegations"] = [e for j, e in enumerate(mm["delegations"]) if j != i]
        out.append((mm, f"{edge['from']}->{edge['to']}"))
    return out


def _op_del_red(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    reach = set(_reachable_agents(m))
    for i, edge in enumerate(m["delegations"]):
        parent = edge["from"]
        old_child = edge["to"]
        for new_child in sorted(reach):
            if new_child == parent or new_child == old_child:
                continue
            mm = _copy(m)
            mm["delegations"][i] = {**edge, "to": new_child}
            out.append((mm, f"{parent}->{old_child}=>{new_child}"))
    return out


def _op_del_inv(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    for i, edge in enumerate(m["delegations"]):
        # Skip self-loops (no change) and skip if the inverted edge already exists.
        if edge["from"] == edge["to"]:
            continue
        inverted = {**edge, "from": edge["to"], "to": edge["from"]}
        if any(e["from"] == inverted["from"] and e["to"] == inverted["to"]
               for e in m["delegations"]):
            continue
        mm = _copy(m)
        mm["delegations"][i] = inverted
        out.append((mm, f"{edge['from']}<->{edge['to']}"))
    return out


def _op_del_add(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    reach = sorted(_reachable_agents(m))
    existing = {(e["from"], e["to"]) for e in m["delegations"]}
    for p, c in itertools.product(reach, reach):
        if p == c or (p, c) in existing:
            continue
        mm = _copy(m)
        mm["delegations"].append({"from": p, "to": c, "trigger": "delegate"})
        out.append((mm, f"+{p}->{c}"))
    return out


# ---------------------------------------------------------------------------
# F2 — Tool wiring operators
# ---------------------------------------------------------------------------

def _op_tool_rem(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    allow = _allow(m)
    for i, e in enumerate(allow):
        mm = _copy(m)
        mm["permissions"]["allow"] = [x for j, x in enumerate(allow) if j != i]
        out.append((mm, f"{e['agent']}|{e['tool']}"))
    return out


def _op_tool_red(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    tools = sorted(t["name"] for t in m["tools"])
    allow = _allow(m)
    for i, e in enumerate(allow):
        for new_tool in tools:
            if new_tool == e["tool"]:
                continue
            mm = _copy(m)
            mm["permissions"]["allow"][i] = {**e, "tool": new_tool}
            # If the redirected target was previously restricted for this
            # agent, drop the restrict entry so the manifest stays consistent
            # with the parser's allow/restrict-disjointness invariant.
            mm["permissions"]["restrict"] = [
                r for r in mm["permissions"].get("restrict", [])
                if not (r["agent"] == e["agent"] and r["tool"] == new_tool)
            ]
            out.append((mm, f"{e['agent']}|{e['tool']}=>{new_tool}"))
    return out


def _op_tool_fault(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    for t in m["tools"]:
        for mode in ("fail:error", "fail:malformed"):
            mm = _copy(m)
            mm.setdefault("_fault_injection", []).append(
                {"tool": t["name"], "mode": mode}
            )
            out.append((mm, f"{t['name']}@{mode}"))
    return out


def _op_tool_sigmut(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    for i, t in enumerate(m["tools"]):
        new_name = f"{t['name']}_v2"
        if any(o["name"] == new_name for o in m["tools"]):
            continue
        mm = _copy(m)
        old = t["name"]
        mm["tools"][i] = {**t, "name": new_name}
        for slot in ("allow", "restrict"):
            for entry in mm["permissions"].get(slot, []):
                if entry["tool"] == old:
                    entry["tool"] = new_name
        out.append((mm, f"{old}=>{new_name}"))
    return out


# ---------------------------------------------------------------------------
# F3 — Restriction operators
# ---------------------------------------------------------------------------

def _op_res_relax(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    restrict = _restrict(m)
    for i, e in enumerate(restrict):
        # Cannot relax if the (agent, tool) pair is already in allow.
        if any(a["agent"] == e["agent"] and a["tool"] == e["tool"] for a in _allow(m)):
            continue
        mm = _copy(m)
        mm["permissions"]["restrict"] = [x for j, x in enumerate(restrict) if j != i]
        mm["permissions"].setdefault("allow", []).append({**e})
        out.append((mm, f"relax({e['agent']}|{e['tool']})"))
    return out


def _op_res_tighten(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    allow = _allow(m)
    for i, e in enumerate(allow):
        if any(r["agent"] == e["agent"] and r["tool"] == e["tool"] for r in _restrict(m)):
            continue
        mm = _copy(m)
        mm["permissions"]["allow"] = [x for j, x in enumerate(allow) if j != i]
        mm["permissions"].setdefault("restrict", []).append({**e})
        out.append((mm, f"tighten({e['agent']}|{e['tool']})"))
    return out


def _op_res_rem(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    restrict = _restrict(m)
    for i, e in enumerate(restrict):
        mm = _copy(m)
        mm["permissions"]["restrict"] = [x for j, x in enumerate(restrict) if j != i]
        out.append((mm, f"{e['agent']}|{e['tool']}"))
    return out


# ---------------------------------------------------------------------------
# F4 — Agent / instruction operators
# ---------------------------------------------------------------------------

def _op_ag_rem(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    entry = m["system"]["entry_agent"]
    reach = _reachable_agents(m)
    for name in reach:
        if name == entry:
            continue
        mm = _copy(m)
        mm["agents"] = [a for a in mm["agents"] if a["name"] != name]
        mm["delegations"] = [
            e for e in mm["delegations"]
            if e["from"] != name and e["to"] != name
        ]
        for slot in ("allow", "restrict"):
            mm["permissions"][slot] = [
                p for p in mm["permissions"].get(slot, []) if p["agent"] != name
            ]
        out.append((mm, name))
    return out


def _op_ag_swap(m: Manifest) -> list[tuple[Manifest, str]]:
    """Swap the ``system_prompt_hint`` of two reachable agents.

    This is a *graph-equivalent* mutation by construction: it changes
    only the instruction text of two agents and leaves the manifest's
    obligation set untouched. The deterministic trace simulator
    (Section IV.B) cannot witness this perturbation because the
    simulator does not model prompt-mediated routing; under
    K-Struct/K-Outcome/K-Safety the operator therefore reports as
    ``n/a`` (no killable mutants) on every suite. It is included in the
    catalogue so that a future live-LLM runtime adapter can detect it.

    To keep the mutant non-identity, pairs whose ``system_prompt_hint``
    fields are equal (including both empty) are skipped.
    """
    out = []
    entry = m["system"]["entry_agent"]
    candidates = [n for n in _reachable_agents(m) if n != entry]
    name_to_idx = {a["name"]: i for i, a in enumerate(m["agents"])}
    for a, b in itertools.combinations(candidates, 2):
        ia, ib = name_to_idx[a], name_to_idx[b]
        pa = m["agents"][ia].get("system_prompt_hint", "")
        pb = m["agents"][ib].get("system_prompt_hint", "")
        if pa == pb:
            continue   # identity swap; skip
        mm = _copy(m)
        mm["agents"][ia] = {**mm["agents"][ia], "system_prompt_hint": pb}
        mm["agents"][ib] = {**mm["agents"][ib], "system_prompt_hint": pa}
        out.append((mm, f"{a}<->{b}"))
    return out


def _op_ag_promptdrop(m: Manifest) -> list[tuple[Manifest, str]]:
    out = []
    for i, ag in enumerate(m["agents"]):
        if not ag.get("reachable", True):
            continue
        if not ag.get("system_prompt_hint"):
            continue
        mm = _copy(m)
        mm["agents"][i] = {**ag, "system_prompt_hint": ""}
        out.append((mm, ag["name"]))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATORS: list[MutationOperator] = [
    MutationOperator("Del-Rem",       "F1_delegation", _op_del_rem),
    MutationOperator("Del-Red",       "F1_delegation", _op_del_red),
    MutationOperator("Del-Inv",       "F1_delegation", _op_del_inv),
    MutationOperator("Del-Add",       "F1_delegation", _op_del_add),
    MutationOperator("Tool-Rem",      "F2_tool",       _op_tool_rem),
    MutationOperator("Tool-Red",      "F2_tool",       _op_tool_red),
    MutationOperator("Tool-Fault",    "F2_tool",       _op_tool_fault),
    MutationOperator("Tool-SigMut",   "F2_tool",       _op_tool_sigmut),
    MutationOperator("Res-Relax",     "F3_restrict",   _op_res_relax),
    MutationOperator("Res-Tighten",   "F3_restrict",   _op_res_tighten),
    MutationOperator("Res-Rem",       "F3_restrict",   _op_res_rem),
    MutationOperator("Ag-Rem",        "F4_agent",      _op_ag_rem),
    MutationOperator("Ag-Swap",       "F4_agent",      _op_ag_swap),
    MutationOperator("Ag-PromptDrop", "F4_agent",      _op_ag_promptdrop),
]

OPERATOR_FAMILIES = {
    "F1_delegation": ["Del-Rem", "Del-Red", "Del-Inv", "Del-Add"],
    "F2_tool":       ["Tool-Rem", "Tool-Red", "Tool-Fault", "Tool-SigMut"],
    "F3_restrict":   ["Res-Relax", "Res-Tighten", "Res-Rem"],
    "F4_agent":      ["Ag-Rem", "Ag-Swap", "Ag-PromptDrop"],
}


def apply_operator(op: MutationOperator, m: Manifest, benchmark: str) -> list[Mutant]:
    """Apply one operator to one manifest and return the resulting mutants."""
    out: list[Mutant] = []
    for i, (mm, target) in enumerate(op.apply(m)):
        out.append(Mutant(
            mutant_id=f"{benchmark}::{op.name}::{i:04d}::{target}",
            operator=op.name,
            family=op.family,
            target=target,
            manifest=mm,
        ))
    return out


def generate_mutants(m: Manifest, benchmark: str) -> list[Mutant]:
    """Apply every operator in OPERATORS to ``m``."""
    out: list[Mutant] = []
    for op in OPERATORS:
        out.extend(apply_operator(op, m, benchmark))
    return out
