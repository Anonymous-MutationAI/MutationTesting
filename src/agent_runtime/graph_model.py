from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ToolKind(str, Enum):
    MCP = "mcp"
    EXE = "exe"
    PY = "py"


@dataclass(frozen=True)
class SkillGraph:
    """Graph abstraction used by the Paper B coverage model."""

    agents: set[str]
    reachable_agents: set[str]
    tools: set[str]
    skills: set[str]
    allow_edges: set[tuple[str, str]] = field(default_factory=set)
    deny_edges: set[tuple[str, str]] = field(default_factory=set)
    subagent_edges: set[tuple[str, str]] = field(default_factory=set)
    skill_edges: set[tuple[str, str]] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    tool_kinds: dict[str, ToolKind] = field(default_factory=dict)

    @property
    def allowed_obligations(self) -> set[tuple[str, str]]:
        return {
            (a, t)
            for (a, t) in self.allow_edges
            if a in self.reachable_agents and a in self.agents and t in self.tools
        }

    @property
    def denied_obligations(self) -> set[tuple[str, str]]:
        return {
            (a, t)
            for (a, t) in self.deny_edges
            if a in self.reachable_agents and a in self.agents and t in self.tools
        }

    @property
    def skill_obligations(self) -> set[tuple[str, str]]:
        return {
            (a, s)
            for (a, s) in self.skill_edges
            if a in self.reachable_agents and a in self.agents and s in self.skills
        }

    @property
    def delegation_obligations(self) -> set[tuple[str, str]]:
        return {
            (a, b)
            for (a, b) in self.subagent_edges
            if a in self.reachable_agents and b in self.reachable_agents
        }

    @property
    def tool_kind_obligations(self) -> set[ToolKind]:
        return {
            self.tool_kinds[t]
            for (_, t) in self.allowed_obligations
            if t in self.tool_kinds
        }

    @property
    def unresolved_allow_edges(self) -> set[tuple[str, str]]:
        return {
            (a, t)
            for (a, t) in self.allow_edges
            if a not in self.agents or a not in self.reachable_agents or t not in self.tools
        }

    @property
    def unresolved_deny_edges(self) -> set[tuple[str, str]]:
        return {
            (a, t)
            for (a, t) in self.deny_edges
            if a not in self.agents or a not in self.reachable_agents or t not in self.tools
        }

    @property
    def unresolved_skill_edges(self) -> set[tuple[str, str]]:
        return {
            (a, s)
            for (a, s) in self.skill_edges
            if a not in self.agents or a not in self.reachable_agents or s not in self.skills
        }

    @property
    def unresolved_subagent_edges(self) -> set[tuple[str, str]]:
        return {
            (a, b)
            for (a, b) in self.subagent_edges
            if a not in self.agents
            or a not in self.reachable_agents
            or b not in self.agents
            or b not in self.reachable_agents
        }


@dataclass(frozen=True)
class CoverageObservation:
    """Observed evidence emitted by a single generated or manual test."""

    target_agent: str
    mode: str
    tools_used: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    skills_used: set[str] = field(default_factory=set)
    delegated_edges: set[tuple[str, str]] = field(default_factory=set)
    # (acting_agent, tool) pairs: the agent that actually invoked each tool.
    tool_calls: set[tuple[str, str]] = field(default_factory=set)
