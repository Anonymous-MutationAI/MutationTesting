from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .graph_model import SkillGraph
from .metrics import EXECUTION_MODES


@dataclass(frozen=True)
class GeneratedTest:
    """A structural test objective emitted by GENERATE."""

    name: str
    target_agent: str
    mode: str
    scenario: str
    expected_tools: tuple[str, ...] = ()
    expected_denials: tuple[str, ...] = ()
    expected_skills: tuple[str, ...] = ()
    expected_delegations: tuple[tuple[str, str], ...] = ()


def _assign_modes(count: int) -> list[str]:
    modes = sorted(EXECUTION_MODES)
    return [modes[index % len(modes)] for index in range(count)]


def generate_suite(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(graph, include_delegation=True, enforce_mode_coverage=True)


def generate_suite_without_mode_repair(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(graph, include_delegation=True, enforce_mode_coverage=False)


def generate_suite_without_delegation_rule(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(graph, include_delegation=False, enforce_mode_coverage=True)


def generate_suite_without_denial_rule(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(
        graph,
        include_delegation=True,
        enforce_mode_coverage=True,
        include_denials=False,
    )


def generate_suite_without_skill_rule(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(
        graph,
        include_delegation=True,
        enforce_mode_coverage=True,
        include_skills=False,
    )


def generate_suite_without_tool_rule(graph: SkillGraph) -> list[GeneratedTest]:
    return _generate_suite_base(
        graph,
        include_delegation=True,
        enforce_mode_coverage=True,
        include_tools=False,
    )


def _generate_suite_base(
    graph: SkillGraph,
    *,
    include_delegation: bool,
    enforce_mode_coverage: bool,
    include_tools: bool = True,
    include_skills: bool = True,
    include_denials: bool = True,
) -> list[GeneratedTest]:
    tests: list[GeneratedTest] = []

    for agent in sorted(graph.reachable_agents):
        tests.append(
            GeneratedTest(
                name=f"agent-smoke-{agent}",
                target_agent=agent,
                mode="validate",
                scenario=f"Run a smoke test for reachable agent '{agent}'.",
            )
        )

    if include_tools:
        for agent, tool in sorted(graph.allowed_obligations):
            tests.append(
                GeneratedTest(
                    name=f"tool-{agent}-{tool}",
                    target_agent=agent,
                    mode="execute",
                    scenario=f"Using agent '{agent}', complete a task that requires tool '{tool}'.",
                    expected_tools=(tool,),
                )
            )

    if include_skills:
        for agent, skill in sorted(graph.skill_obligations):
            tests.append(
                GeneratedTest(
                    name=f"skill-{agent}-{skill}",
                    target_agent=agent,
                    mode="dry-run",
                    scenario=f"Using agent '{agent}', invoke or rely on skill '{skill}'.",
                    expected_skills=(skill,),
                )
            )

    if include_denials:
        for agent, tool in sorted(graph.denied_obligations):
            tests.append(
                GeneratedTest(
                    name=f"deny-{agent}-{tool}",
                    target_agent=agent,
                    mode="execute",
                    scenario=(
                        f"Using agent '{agent}', attempt a task that would require "
                        f"forbidden tool '{tool}' and confirm denial."
                    ),
                    expected_denials=(tool,),
                )
            )

    if include_delegation:
        for parent, child in sorted(graph.delegation_obligations):
            tests.append(
                GeneratedTest(
                    name=f"delegate-{parent}-{child}",
                    target_agent=parent,
                    mode="execute",
                    scenario=(
                        f"Using agent '{parent}', complete a task that requires "
                        f"delegation to subagent '{child}'."
                    ),
                    expected_delegations=((parent, child),),
                )
            )

    return _ensure_mode_coverage(tests) if enforce_mode_coverage else tests


def generate_partial_suite(graph: SkillGraph) -> list[GeneratedTest]:
    """A weaker baseline: keep only a sparse subset of generated tests."""
    full = generate_suite(graph)
    if len(full) <= 3:
        return full[:1]

    selected: list[GeneratedTest] = []
    seen_modes: set[str] = set()

    for test in full:
        if test.name.startswith("agent-smoke-"):
            selected.append(test)
            seen_modes.add(test.mode)
            break

    for test in full:
        if test.expected_tools:
            selected.append(test)
            seen_modes.add(test.mode)
            break

    for test in full:
        if test.expected_skills:
            selected.append(test)
            seen_modes.add(test.mode)
            break

    for test in full:
        if len(selected) >= max(2, len(full) // 3):
            break
        if test not in selected:
            selected.append(test)
            seen_modes.add(test.mode)

    return _ensure_mode_coverage(selected)


def generate_agent_first_suite(graph: SkillGraph) -> list[GeneratedTest]:
    """A stronger heuristic baseline: one representative witness per family per agent."""
    selected: list[GeneratedTest] = []

    for agent in sorted(graph.reachable_agents):
        selected.append(
            GeneratedTest(
                name=f"heuristic-smoke-{agent}",
                target_agent=agent,
                mode="validate",
                scenario=f"Heuristic smoke test for reachable agent '{agent}'.",
            )
        )

        allowed = sorted(tool for (a, tool) in graph.allowed_obligations if a == agent)
        if allowed:
            tool = allowed[0]
            selected.append(
                GeneratedTest(
                    name=f"heuristic-tool-{agent}-{tool}",
                    target_agent=agent,
                    mode="execute",
                    scenario=f"Heuristic baseline uses representative tool '{tool}' from agent '{agent}'.",
                    expected_tools=(tool,),
                )
            )

        skills = sorted(skill for (a, skill) in graph.skill_obligations if a == agent)
        if skills:
            skill = skills[0]
            selected.append(
                GeneratedTest(
                    name=f"heuristic-skill-{agent}-{skill}",
                    target_agent=agent,
                    mode="dry-run",
                    scenario=f"Heuristic baseline invokes representative skill '{skill}' from agent '{agent}'.",
                    expected_skills=(skill,),
                )
            )

        denials = sorted(tool for (a, tool) in graph.denied_obligations if a == agent)
        if denials:
            tool = denials[0]
            selected.append(
                GeneratedTest(
                    name=f"heuristic-deny-{agent}-{tool}",
                    target_agent=agent,
                    mode="execute",
                    scenario=f"Heuristic baseline probes representative denied tool '{tool}' from agent '{agent}'.",
                    expected_denials=(tool,),
                )
            )

        delegations = sorted(child for (parent, child) in graph.delegation_obligations if parent == agent)
        if delegations:
            child = delegations[0]
            selected.append(
                GeneratedTest(
                    name=f"heuristic-delegate-{agent}-{child}",
                    target_agent=agent,
                    mode="execute",
                    scenario=f"Heuristic baseline asks '{agent}' to delegate to representative subagent '{child}'.",
                    expected_delegations=((agent, child),),
                )
            )

    return _ensure_mode_coverage(selected)


def generate_hub_first_suite(graph: SkillGraph) -> list[GeneratedTest]:
    """A stronger non-coverage baseline: spend budget on structurally central agents."""
    full = generate_suite(graph)
    if not full:
        return []

    budget = len(generate_agent_first_suite(graph))
    incident_score: dict[str, int] = {agent: 1 for agent in graph.reachable_agents}

    for agent, _tool in graph.allowed_obligations:
        incident_score[agent] = incident_score.get(agent, 0) + 1
    for agent, _skill in graph.skill_obligations:
        incident_score[agent] = incident_score.get(agent, 0) + 1
    for agent, _tool in graph.denied_obligations:
        incident_score[agent] = incident_score.get(agent, 0) + 1
    for parent, child in graph.delegation_obligations:
        incident_score[parent] = incident_score.get(parent, 0) + 2
        incident_score[child] = incident_score.get(child, 0) + 1

    ranked_agents = sorted(
        graph.reachable_agents,
        key=lambda agent: (-incident_score.get(agent, 0), agent),
    )

    grouped: dict[str, list[GeneratedTest]] = {agent: [] for agent in ranked_agents}
    for test in full:
        grouped.setdefault(test.target_agent, []).append(test)

    def priority(test: GeneratedTest) -> tuple[int, str]:
        if test.expected_delegations:
            return (0, test.name)
        if test.expected_denials:
            return (1, test.name)
        if test.expected_tools:
            return (2, test.name)
        if test.expected_skills:
            return (3, test.name)
        return (4, test.name)

    selected: list[GeneratedTest] = []
    for agent in ranked_agents:
        for test in sorted(grouped.get(agent, []), key=priority):
            if len(selected) >= budget:
                break
            selected.append(
                GeneratedTest(
                    name=f"hub-{test.name}",
                    target_agent=test.target_agent,
                    mode=test.mode,
                    scenario=f"Hub-first baseline: {test.scenario}",
                    expected_tools=test.expected_tools,
                    expected_denials=test.expected_denials,
                    expected_skills=test.expected_skills,
                    expected_delegations=test.expected_delegations,
                )
            )
        if len(selected) >= budget:
            break

    return _ensure_mode_coverage(selected)


def generate_smoke_suite(graph: SkillGraph) -> list[GeneratedTest]:
    """A very weak baseline: one smoke test per benchmark with mode backfill."""
    if not graph.reachable_agents:
        return []

    agent = sorted(graph.reachable_agents)[0]
    return _ensure_mode_coverage(
        [
            GeneratedTest(
                name=f"smoke-{agent}",
                target_agent=agent,
                mode="validate",
                scenario=f"Run a generic smoke test for '{agent}'.",
            )
        ]
    )


def generate_random_suite(graph: SkillGraph, seed: int | None = None) -> list[GeneratedTest]:
    """A reproducible random baseline that samples structural objectives."""
    candidates: list[GeneratedTest] = []

    for agent in sorted(graph.reachable_agents):
        candidates.append(
            GeneratedTest(
                name=f"rand-agent-{agent}",
                target_agent=agent,
                mode="validate",
                scenario=f"Random baseline smoke test for '{agent}'.",
            )
        )

    for agent, tool in sorted(graph.allowed_obligations):
        candidates.append(
            GeneratedTest(
                name=f"rand-tool-{agent}-{tool}",
                target_agent=agent,
                mode="execute",
                scenario=f"Random baseline uses tool '{tool}' from agent '{agent}'.",
                expected_tools=(tool,),
            )
        )

    for agent, skill in sorted(graph.skill_obligations):
        candidates.append(
            GeneratedTest(
                name=f"rand-skill-{agent}-{skill}",
                target_agent=agent,
                mode="dry-run",
                scenario=f"Random baseline invokes skill '{skill}' from agent '{agent}'.",
                expected_skills=(skill,),
            )
        )

    for agent, tool in sorted(graph.denied_obligations):
        candidates.append(
            GeneratedTest(
                name=f"rand-deny-{agent}-{tool}",
                target_agent=agent,
                mode="execute",
                scenario=f"Random baseline probes denied tool '{tool}' from agent '{agent}'.",
                expected_denials=(tool,),
            )
        )

    for parent, child in sorted(graph.delegation_obligations):
        candidates.append(
            GeneratedTest(
                name=f"rand-delegate-{parent}-{child}",
                target_agent=parent,
                mode="execute",
                scenario=f"Random baseline asks '{parent}' to delegate to '{child}'.",
                expected_delegations=((parent, child),),
            )
        )

    if not candidates:
        return []

    suite_size = max(3, len(candidates))
    resolved_seed = (
        seed
        if seed is not None
        else (
            len(graph.reachable_agents) * 97
            + len(graph.allowed_obligations) * 89
            + len(graph.skill_obligations) * 83
            + len(graph.denied_obligations) * 79
            + len(graph.delegation_obligations) * 73
        )
    )
    rng = Random(resolved_seed)

    sampled: list[GeneratedTest] = []
    for index in range(suite_size):
        candidate = rng.choice(candidates)
        sampled.append(
            GeneratedTest(
                name=f"{candidate.name}-sample-{index}",
                target_agent=candidate.target_agent,
                mode=candidate.mode,
                scenario=candidate.scenario,
                expected_tools=candidate.expected_tools,
                expected_denials=candidate.expected_denials,
                expected_skills=candidate.expected_skills,
                expected_delegations=candidate.expected_delegations,
            )
        )

    return _ensure_mode_coverage(sampled)


def _ensure_mode_coverage(tests: list[GeneratedTest]) -> list[GeneratedTest]:
    if not tests:
        return []

    required = sorted(EXECUTION_MODES)
    reassigned = list(tests)

    for index, mode in enumerate(_assign_modes(len(reassigned))):
        test = reassigned[index]
        reassigned[index] = GeneratedTest(
            name=test.name,
            target_agent=test.target_agent,
            mode=mode,
            scenario=test.scenario,
            expected_tools=test.expected_tools,
            expected_denials=test.expected_denials,
            expected_skills=test.expected_skills,
            expected_delegations=test.expected_delegations,
        )

    present = {test.mode for test in reassigned}
    missing = [mode for mode in required if mode not in present]
    if not missing:
        return reassigned

    template = reassigned[0]
    for mode in missing:
        reassigned.append(
            GeneratedTest(
                name=f"{template.name}-mode-{mode}",
                target_agent=template.target_agent,
                mode=mode,
                scenario=f"{template.scenario} [mode backfill: {mode}]",
                expected_tools=template.expected_tools,
                expected_denials=template.expected_denials,
                expected_skills=template.expected_skills,
                expected_delegations=template.expected_delegations,
            )
        )

    return reassigned
