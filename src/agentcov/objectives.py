"""Schema v3 witness objectives.

Four objective families correspond to C1 / C2 / C4 / C5. Objectives are
produced deterministically from the workflow graph; only the prompt
realization step (a separate module) is non-deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from .graph_model import WorkflowGraph


class ObjectiveKind(str, Enum):
    REACH = "reach"                  # C1
    USE_TOOL = "use_tool"            # C2
    RESTRICT_TOOL = "restrict_tool"  # C4
    DELEGATE = "delegate"            # C5


@dataclass(frozen=True)
class Objective:
    """One structural test obligation."""

    objective_id: str
    kind: ObjectiveKind
    agent: str
    tool: str | None = None
    child_agent: str | None = None

    def witness_key(self) -> tuple:
        if self.kind is ObjectiveKind.REACH:
            return ("reach", self.agent)
        if self.kind is ObjectiveKind.USE_TOOL:
            return ("use_tool", self.agent, self.tool)
        if self.kind is ObjectiveKind.RESTRICT_TOOL:
            return ("restrict_tool", self.agent, self.tool)
        if self.kind is ObjectiveKind.DELEGATE:
            return ("delegate", self.agent, self.child_agent)
        raise ValueError(f"unknown objective kind {self.kind!r}")


def extract_objectives(graph: WorkflowGraph) -> list[Objective]:
    """Apply rules R1, R2, R4, R5 to derive the deterministic objective set."""
    out: list[Objective] = []

    # R1: agent reach
    for agent in sorted(graph.agent_obligations):
        out.append(
            Objective(
                objective_id=f"reach::{agent}",
                kind=ObjectiveKind.REACH,
                agent=agent,
            )
        )

    # R2: allowed tool
    for (agent, tool) in sorted(graph.allowed_obligations):
        out.append(
            Objective(
                objective_id=f"use_tool::{agent}::{tool}",
                kind=ObjectiveKind.USE_TOOL,
                agent=agent,
                tool=tool,
            )
        )

    # R4: restricted tool (negative)
    for (agent, tool) in sorted(graph.restricted_obligations):
        out.append(
            Objective(
                objective_id=f"restrict_tool::{agent}::{tool}",
                kind=ObjectiveKind.RESTRICT_TOOL,
                agent=agent,
                tool=tool,
            )
        )

    # R5: delegation
    for (parent, child) in sorted(graph.delegation_obligations):
        out.append(
            Objective(
                objective_id=f"delegate::{parent}::{child}",
                kind=ObjectiveKind.DELEGATE,
                agent=parent,
                child_agent=child,
            )
        )

    return out


def iter_objectives(graph: WorkflowGraph) -> Iterator[Objective]:
    yield from extract_objectives(graph)
