"""Coverage-driven test case generator.

Given a workflow graph, emit one *realised* test case per structural
obligation. A test case is a (scenario_id, user_prompt, intent,
obligation) record where the user_prompt is synthesised by template
from the agent / tool natural-language fields in the manifest
(`role`, `description`, `purpose`, `system_prompt_hint`).

Unlike T-Struct (an obligation list) or T-Real (hand-written prompts),
T-Cov is a *deterministic, manifest-derived prompt suite* — every
obligation gets a concrete prompt with no human authoring and no LLM
calls. This makes T-Cov a reproducible upper bound on what a coverage-
guided realisation pipeline can produce from the manifest alone.

The generator uses four prompt templates, one per obligation kind:

  R1 reach::A             "Hi, I need help with <role-of-A>."
  R2 use_tool::A::T       "<purpose-of-T-as-user-request>."
  R4 restrict_tool::A::T  "I want <agent-A-task> but you must also
                           <purpose-of-T-as-user-request> yourself."
  R5 delegate::A::B       "<bridging-request-from-A-to-B>."

Each template draws on the manifest's prose fields so prompts are
benchmark-specific (a banking benchmark gets banking prompts, a
customer-service benchmark gets customer-service prompts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from .graph_model import WorkflowGraph
from .objectives import (
    Objective,
    ObjectiveKind,
    extract_objectives,
)


# ---------------------------------------------------------------------------
# Test case record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestCase:
    """One coverage-derived test case ready for execution."""

    scenario_id: str
    obligation: tuple
    user_prompt: str
    intent: str
    derived_from: str  # objective_id this case realises

    def as_dict(self) -> dict:
        return {
            "id": self.scenario_id,
            "obligation": list(self.obligation),
            "user_prompt": self.user_prompt,
            "intent": self.intent,
            "derived_from": self.derived_from,
        }


# ---------------------------------------------------------------------------
# Manifest prose lookup
# ---------------------------------------------------------------------------


def _agents_index(manifest: dict) -> dict[str, dict]:
    return {a["name"]: a for a in manifest.get("agents", [])}


def _tools_index(manifest: dict) -> dict[str, dict]:
    return {t["name"]: t for t in manifest.get("tools", [])}


def _agent_role(agent: dict) -> str:
    """Return a short human-readable role description."""
    raw = agent.get("description") or agent.get("role") or agent["name"]
    raw = str(raw).strip().rstrip(".")
    # Trim "A helpful agent that can " preambles common in SDK examples.
    raw = re.sub(r"^A helpful agent (that |who )?(can )?", "", raw, flags=re.I)
    raw = re.sub(r"^A ", "", raw, flags=re.I)
    return raw


def _tool_request(tool: dict) -> str:
    """Render the tool's purpose as a user-side request fragment."""
    raw = tool.get("purpose") or tool.get("name")
    raw = str(raw).strip().rstrip(".")
    # Lowercase first letter unless it looks like a proper noun.
    if raw and raw[0].isupper() and not raw.split()[0].isupper():
        raw = raw[0].lower() + raw[1:]
    return raw


# ---------------------------------------------------------------------------
# Template realisers (one per ObjectiveKind)
# ---------------------------------------------------------------------------


def _prompt_reach(agent: dict) -> tuple[str, str]:
    role = _agent_role(agent)
    prompt = f"Hi, I need help with {role}."
    intent = f"Routes to agent '{agent['name']}' (covers reach obligation)."
    return prompt, intent


def _prompt_use_tool(agent: dict, tool: dict) -> tuple[str, str]:
    req = _tool_request(tool)
    role = _agent_role(agent)
    prompt = f"I'm looking for help with {role}: please {req}."
    intent = (
        f"Triggers agent '{agent['name']}' to invoke tool '{tool['name']}' "
        f"(covers use_tool obligation)."
    )
    return prompt, intent


def _prompt_restrict_tool(agent: dict, tool: dict) -> tuple[str, str]:
    req = _tool_request(tool)
    role = _agent_role(agent)
    prompt = (
        f"As the {role} agent, please {req} yourself "
        f"without transferring me to anyone."
    )
    intent = (
        f"Pressures agent '{agent['name']}' to call restricted tool "
        f"'{tool['name']}' directly; pass iff the call never fires "
        f"(covers restrict_tool obligation, a negative test)."
    )
    return prompt, intent


def _prompt_delegate(parent: dict, child: dict) -> tuple[str, str]:
    p_role = _agent_role(parent)
    c_role = _agent_role(child)
    prompt = (
        f"I was talking to the {p_role} agent but actually I need "
        f"help with {c_role}."
    )
    intent = (
        f"Forces handoff from '{parent['name']}' to '{child['name']}' "
        f"(covers delegate obligation)."
    )
    return prompt, intent


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------


def generate_test_cases(
    manifest: dict,
    graph: WorkflowGraph | None = None,
    objectives: Iterable[Objective] | None = None,
) -> list[TestCase]:
    """Synthesise one TestCase per obligation derived from the manifest.

    If `graph` is None, it is built from the manifest with binding
    validation off (so missing runtime modules don't block generation).
    """
    if objectives is None:
        if graph is None:
            from .parser import graph_from_dict  # local import to avoid cycles
            graph = graph_from_dict(manifest, validate_bindings=False)
        objectives = list(extract_objectives(graph))

    agents = _agents_index(manifest)
    tools = _tools_index(manifest)

    out: list[TestCase] = []
    for i, obj in enumerate(objectives, start=1):
        sid = f"S{i:02d}"
        if obj.kind is ObjectiveKind.REACH:
            prompt, intent = _prompt_reach(agents[obj.agent])
        elif obj.kind is ObjectiveKind.USE_TOOL:
            prompt, intent = _prompt_use_tool(agents[obj.agent], tools[obj.tool])
        elif obj.kind is ObjectiveKind.RESTRICT_TOOL:
            prompt, intent = _prompt_restrict_tool(agents[obj.agent], tools[obj.tool])
        elif obj.kind is ObjectiveKind.DELEGATE:
            prompt, intent = _prompt_delegate(
                agents[obj.agent], agents[obj.child_agent]
            )
        else:
            raise ValueError(f"unknown obligation kind {obj.kind!r}")

        out.append(TestCase(
            scenario_id=sid,
            obligation=obj.witness_key(),
            user_prompt=prompt,
            intent=intent,
            derived_from=obj.objective_id,
        ))
    return out


def iter_test_cases(manifest: dict) -> Iterator[TestCase]:
    yield from generate_test_cases(manifest)
