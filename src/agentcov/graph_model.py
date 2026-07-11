"""Schema v3 graph model.

Four coverage criteria only: C1 agents, C2 allowed tools,
C4 restricted tools, C5 delegations. Tool surface kinds (mcp/http/py/exe)
are an execution-time concept and are not declared in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


SCHEMA_VERSION = "3.0"


class SideEffect(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class ToolBinding:
    """Resolvable runtime binding for a tool."""

    module: str
    callable: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    purpose: str
    side_effects: SideEffect
    binding: ToolBinding


@dataclass(frozen=True)
class AgentSpec:
    name: str
    reachable: bool
    role: str
    description: str
    system_prompt_hint: str


@dataclass(frozen=True)
class DelegationEdge:
    parent: str
    child: str
    trigger: str


@dataclass(frozen=True)
class SystemSpec:
    name: str
    entry_agent: str
    description: str


@dataclass(frozen=True)
class WorkflowGraph:
    """Schema v3 workflow graph (four criteria)."""

    schema_version: str
    system: SystemSpec
    agents: dict[str, AgentSpec]
    tools: dict[str, ToolSpec]
    allow_edges: set[tuple[str, str]]
    restrict_edges: set[tuple[str, str]]
    delegations: tuple[DelegationEdge, ...]

    # ---- Reachability ---------------------------------------------------
    @property
    def reachable_agents(self) -> set[str]:
        """BFS from entry through delegations, intersected with declared reachable."""
        adjacency: dict[str, list[str]] = {name: [] for name in self.agents}
        for edge in self.delegations:
            adjacency.setdefault(edge.parent, []).append(edge.child)

        seen: set[str] = set()
        frontier = [self.system.entry_agent]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            for child in adjacency.get(current, []):
                if child not in seen:
                    frontier.append(child)

        return {name for name in seen if self.agents[name].reachable}

    # ---- Obligations (C1, C2, C4, C5) -----------------------------------
    @property
    def agent_obligations(self) -> set[str]:
        return set(self.reachable_agents)

    @property
    def allowed_obligations(self) -> set[tuple[str, str]]:
        reach = self.reachable_agents
        return {(a, t) for (a, t) in self.allow_edges if a in reach}

    @property
    def restricted_obligations(self) -> set[tuple[str, str]]:
        reach = self.reachable_agents
        return {(a, t) for (a, t) in self.restrict_edges if a in reach}

    @property
    def delegation_obligations(self) -> set[tuple[str, str]]:
        reach = self.reachable_agents
        return {
            (edge.parent, edge.child)
            for edge in self.delegations
            if edge.parent in reach and edge.child in reach
        }


@dataclass(frozen=True)
class CoverageObservation:
    """Observed evidence emitted by one executed test."""

    target_agent: str
    tool_invocations: tuple[str, ...] = ()
    restricted_tools_observed: tuple[str, ...] = ()
    delegated_edges: tuple[tuple[str, str], ...] = ()

    @property
    def tools_used(self) -> set[str]:
        return set(self.tool_invocations)
