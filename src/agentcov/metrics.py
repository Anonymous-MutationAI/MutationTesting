"""Schema v3 coverage metrics (4 paper-aligned criteria).

C1 Agent coverage           — every reachable agent observed.
C2 Allowed-tool coverage    — every reachable allow edge exercised.
C4 Restricted-tool coverage — every reachable restrict edge witnessed by an
                              explicit negative observation.
C5 Delegation coverage      — every reachable delegation edge observed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .graph_model import CoverageObservation, WorkflowGraph


Witness = Callable[[CoverageObservation, object], bool]


def _coverage(
    observations: Iterable[CoverageObservation],
    obligations: set[object],
    witness: Witness,
) -> float:
    if not obligations:
        return 1.0
    obs = list(observations)
    covered = {item for item in obligations if any(witness(o, item) for o in obs)}
    return len(covered) / len(obligations)


def c1_agent(graph: WorkflowGraph, observations: Iterable[CoverageObservation]) -> float:
    return _coverage(
        observations,
        set(graph.agent_obligations),
        lambda o, agent: o.target_agent == agent,
    )


def c2_allowed_tool(graph: WorkflowGraph, observations: Iterable[CoverageObservation]) -> float:
    return _coverage(
        observations,
        set(graph.allowed_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge[1] in o.tools_used,
    )


def c4_restricted_tool(graph: WorkflowGraph, observations: Iterable[CoverageObservation]) -> float:
    return _coverage(
        observations,
        set(graph.restricted_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge[1] in o.restricted_tools_observed,
    )


def c5_delegation(graph: WorkflowGraph, observations: Iterable[CoverageObservation]) -> float:
    return _coverage(
        observations,
        set(graph.delegation_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge in o.delegated_edges,
    )


def coverage_report(
    graph: WorkflowGraph,
    observations: Iterable[CoverageObservation],
) -> dict[str, float]:
    obs = list(observations)
    return {
        "C1_agent":           c1_agent(graph, obs),
        "C2_allowed_tool":    c2_allowed_tool(graph, obs),
        "C4_restricted_tool": c4_restricted_tool(graph, obs),
        "C5_delegation":      c5_delegation(graph, obs),
    }


def adequacy_score(
    graph: WorkflowGraph,
    observations: Iterable[CoverageObservation],
    weights: dict[str, float] | None = None,
) -> float:
    report = coverage_report(graph, observations)
    if weights is None:
        weights = {name: 1.0 / len(report) for name in report}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = {name: value / total for name, value in weights.items()}
    return sum(report[name] * normalized.get(name, 0.0) for name in report)
