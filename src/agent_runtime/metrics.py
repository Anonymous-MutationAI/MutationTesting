from __future__ import annotations

from collections.abc import Callable, Iterable

from .graph_model import CoverageObservation, SkillGraph

Witness = Callable[[CoverageObservation, object], bool]

EXECUTION_MODES = {"validate", "dry-run", "execute"}


def coverage(
    observations: Iterable[CoverageObservation],
    obligations: set[object],
    witness: Witness,
) -> float:
    """Generic coverage functional used by all seven criteria."""
    if not obligations:
        return 1.0
    obs = list(observations)
    covered = {item for item in obligations if any(witness(o, item) for o in obs)}
    return len(covered) / len(obligations)


def c1_agent_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    return coverage(
        observations,
        set(graph.reachable_agents),
        lambda o, a: o.target_agent == a,
    )


def c2_tool_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    return coverage(
        observations,
        set(graph.allowed_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge[1] in o.tools_used,
    )


def c3_tool_kind_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    return coverage(
        observations,
        set(graph.tool_kind_obligations),
        lambda o, kind: any(graph.tool_kinds.get(tool) == kind for tool in o.tools_used),
    )


def c4_skill_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    return coverage(
        observations,
        set(graph.skill_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge[1] in o.skills_used,
    )


def c5_failure_path_coverage(
    graph: SkillGraph,
    observations: Iterable[CoverageObservation],
) -> float:
    return coverage(
        observations,
        set(graph.denied_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge[1] in o.denied_tools,
    )


def c6_mode_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    del graph
    return coverage(observations, set(EXECUTION_MODES), lambda o, mode: o.mode == mode)


def c7_subagent_coverage(graph: SkillGraph, observations: Iterable[CoverageObservation]) -> float:
    return coverage(
        observations,
        set(graph.delegation_obligations),
        lambda o, edge: o.target_agent == edge[0] and edge in o.delegated_edges,
    )


def coverage_report(
    graph: SkillGraph,
    observations: Iterable[CoverageObservation],
) -> dict[str, float]:
    obs = list(observations)
    return {
        "C1_agent": c1_agent_coverage(graph, obs),
        "C2_tool": c2_tool_coverage(graph, obs),
        "C3_tool_kind": c3_tool_kind_coverage(graph, obs),
        "C4_skill": c4_skill_coverage(graph, obs),
        "C5_failure_path": c5_failure_path_coverage(graph, obs),
        "C6_mode": c6_mode_coverage(graph, obs),
        "C7_subagent": c7_subagent_coverage(graph, obs),
    }


def adequacy_score(
    graph: SkillGraph,
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
