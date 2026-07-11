"""Deterministic, LLM-free trace simulator for schema-v3 manifests.

The simulator interprets a manifest as a state machine. A `Scenario`
asks the system to satisfy one structural obligation; the simulator
walks the shortest delegation path from the entry agent and emits a
`Trace` (sequence of agent steps + tool calls + delegations + the
verdict of the obligation).

The simulator is intentionally **conservative**: it represents the
*intended* behaviour of a well-formed workflow. It assumes that:

  * every well-formed agent follows the delegation path that leads to
    the target;
  * an agent calls a tool when (and only when) the tool is in its
    `allow` set and the scenario targets that agent-tool pair;
  * an agent never calls a tool in its `restrict` set
    (this is what K-Safety checks for — a mutant that *removes* a
    restriction allows the simulator to fire a forbidden call);
  * fault-injected tools (the `_fault_injection` side-channel produced
    by the Tool-Fault operator) return a canned error/malformed payload
    that the agent reports as a tool-call FAILURE.

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Scenario + Trace data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One test case asking the workflow to satisfy a structural obligation."""

    scenario_id: str
    obligation: tuple  # ("reach", a) | ("use_tool", a, t) | ("restrict_tool", a, t) | ("delegate", p, c)


@dataclass(frozen=True)
class TraceStep:
    agent: str
    action: str   # "enter" | "delegate" | "tool_ok" | "tool_fail" | "tool_forbidden" | "halt"
    target: Optional[str] = None  # tool name, child agent name, etc.


@dataclass(frozen=True)
class Trace:
    scenario_id: str
    verdict: str          # "PASS" | "FAIL" | "UNREACHABLE" | "ABORT_FORBIDDEN" | "ABORT_FAULT"
    steps: tuple[TraceStep, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Manifest views
# ---------------------------------------------------------------------------

def _adjacency(manifest: dict) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {a["name"]: [] for a in manifest["agents"]}
    for e in manifest["delegations"]:
        adj.setdefault(e["from"], []).append(e["to"])
    return adj


def _allow_set(manifest: dict) -> set[tuple[str, str]]:
    return {(e["agent"], e["tool"])
            for e in manifest.get("permissions", {}).get("allow", [])}


def _restrict_set(manifest: dict) -> set[tuple[str, str]]:
    return {(e["agent"], e["tool"])
            for e in manifest.get("permissions", {}).get("restrict", [])}


def _fault_set(manifest: dict) -> set[str]:
    return {f["tool"] for f in manifest.get("_fault_injection", [])}


def _shortest_path(adj: dict[str, list[str]], src: str, dst: str) -> Optional[list[str]]:
    """BFS shortest path src->dst, inclusive. None if unreachable."""
    if src == dst:
        return [src]
    seen = {src}
    parent: dict[str, str] = {}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = cur
            if nxt == dst:
                # reconstruct
                path = [nxt]
                while path[-1] != src:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            q.append(nxt)
    return None


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(manifest: dict, scenario: Scenario) -> Trace:
    """Execute one scenario against the (mutant or original) manifest.

    Verdict semantics:
      PASS              — obligation satisfied
      FAIL              — workflow walked but obligation not satisfied
                          (e.g. arrived at wrong agent, tool not called)
      UNREACHABLE       — no path from entry to the target agent
      ABORT_FORBIDDEN   — the would-be-called tool is in restrict for the
                          current agent (K-Safety abort)
      ABORT_FAULT       — fault-injected tool returned error/malformed
    """
    entry = manifest["system"]["entry_agent"]
    adj = _adjacency(manifest)
    allow = _allow_set(manifest)
    restrict = _restrict_set(manifest)
    faults = _fault_set(manifest)

    steps: list[TraceStep] = [TraceStep(entry, "enter")]
    kind = scenario.obligation[0]

    if kind == "reach":
        target_agent = scenario.obligation[1]
        path = _shortest_path(adj, entry, target_agent)
        if path is None:
            return Trace(scenario.scenario_id, "UNREACHABLE", tuple(steps))
        for child in path[1:]:
            parent = steps[-1].agent
            steps.append(TraceStep(parent, "delegate", child))
            steps.append(TraceStep(child, "enter"))
        steps.append(TraceStep(target_agent, "halt"))
        return Trace(scenario.scenario_id, "PASS", tuple(steps))

    if kind == "delegate":
        parent, child = scenario.obligation[1], scenario.obligation[2]
        path_to_parent = _shortest_path(adj, entry, parent)
        if path_to_parent is None:
            return Trace(scenario.scenario_id, "UNREACHABLE", tuple(steps))
        for nxt in path_to_parent[1:]:
            prev = steps[-1].agent
            steps.append(TraceStep(prev, "delegate", nxt))
            steps.append(TraceStep(nxt, "enter"))
        # Is the (parent -> child) edge present?
        if child not in adj.get(parent, []):
            steps.append(TraceStep(parent, "halt"))
            return Trace(scenario.scenario_id, "FAIL", tuple(steps))
        steps.append(TraceStep(parent, "delegate", child))
        steps.append(TraceStep(child, "enter"))
        steps.append(TraceStep(child, "halt"))
        return Trace(scenario.scenario_id, "PASS", tuple(steps))

    if kind == "use_tool":
        agent, tool = scenario.obligation[1], scenario.obligation[2]
        path = _shortest_path(adj, entry, agent)
        if path is None:
            return Trace(scenario.scenario_id, "UNREACHABLE", tuple(steps))
        for child in path[1:]:
            prev = steps[-1].agent
            steps.append(TraceStep(prev, "delegate", child))
            steps.append(TraceStep(child, "enter"))
        # K-Safety: if the tool is in restrict for this agent, the
        # simulator refuses to call it.
        if (agent, tool) in restrict:
            steps.append(TraceStep(agent, "tool_forbidden", tool))
            return Trace(scenario.scenario_id, "ABORT_FORBIDDEN", tuple(steps))
        # Must be allowed to invoke.
        if (agent, tool) not in allow:
            steps.append(TraceStep(agent, "halt"))
            return Trace(scenario.scenario_id, "FAIL", tuple(steps))
        if tool in faults:
            steps.append(TraceStep(agent, "tool_fail", tool))
            return Trace(scenario.scenario_id, "ABORT_FAULT", tuple(steps))
        steps.append(TraceStep(agent, "tool_ok", tool))
        steps.append(TraceStep(agent, "halt"))
        return Trace(scenario.scenario_id, "PASS", tuple(steps))

    if kind == "restrict_tool":
        # Negative-polarity witness (PASS = violation elicited).
        # Under the *intended* spec, an agent never calls a restricted
        # tool, so the simulator always reports a refusal. A mutant that
        # silently moves a restricted edge to allow will cause the
        # simulator to actually fire the call -> ABORT_FORBIDDEN -> the
        # trace diverges from the original -> K-Outcome kills it.
        # K-Safety: any simulator path that fires a restricted tool
        # counts as a safety violation.
        agent, tool = scenario.obligation[1], scenario.obligation[2]
        path = _shortest_path(adj, entry, agent)
        if path is None:
            return Trace(scenario.scenario_id, "UNREACHABLE", tuple(steps))
        for child in path[1:]:
            prev = steps[-1].agent
            steps.append(TraceStep(prev, "delegate", child))
            steps.append(TraceStep(child, "enter"))
        # In the original spec the restriction is in place: refuse.
        if (agent, tool) in restrict:
            steps.append(TraceStep(agent, "tool_forbidden", tool))
            steps.append(TraceStep(agent, "halt"))
            return Trace(scenario.scenario_id, "FAIL", tuple(steps))  # FAIL = refused as expected
        # The mutant has relaxed the restriction. The simulator now
        # fires the call (because nothing prevents it). This is the
        # K-Safety violation.
        if (agent, tool) in allow:
            steps.append(TraceStep(agent, "tool_ok", tool))
            steps.append(TraceStep(agent, "halt"))
            return Trace(scenario.scenario_id, "PASS", tuple(steps))  # PASS = violation elicited
        # Neither in allow nor restrict — agent has no reason to fire.
        steps.append(TraceStep(agent, "halt"))
        return Trace(scenario.scenario_id, "FAIL", tuple(steps))

    raise ValueError(f"unknown obligation kind: {kind}")
