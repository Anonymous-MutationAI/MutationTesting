from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from agents import Agent, Runner, RunContextWrapper, function_tool

from .generator import GeneratedTest
from .graph_model import CoverageObservation
from .live_runtime_backends import LocalRuntimeState, ensure_tool_backends


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

try:
    from agents import set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    set_default_openai_client(
        AsyncOpenAI(timeout=30.0, max_retries=2), use_for_tracing=False
    )
except Exception:  # noqa: BLE001 - never block startup on this
    pass


@dataclass
class FrameworkTrace:
    used_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    used_skills: list[str] = field(default_factory=list)
    used_pairs: list[tuple[str, str]] = field(default_factory=list)
    runtime_state: LocalRuntimeState | None = None


@dataclass
class FrameworkExecution:
    observation: CoverageObservation
    final_output: str
    delegated_edges: list[tuple[str, str]]
    raw_items: list[dict[str, Any]]


def _allowed_tools(manifest: dict[str, Any], agent_name: str) -> set[str]:
    return {
        item["tool"]
        for item in manifest.get("permissions", {}).get("allow", [])
        if item["agent"] == agent_name
    }


def _denied_tools(manifest: dict[str, Any], agent_name: str) -> set[str]:
    return {
        item["tool"]
        for item in manifest.get("permissions", {}).get("deny", [])
        if item["agent"] == agent_name
    }


def _skills(manifest: dict[str, Any], agent_name: str) -> list[str]:
    for agent in manifest.get("agents", []):
        if agent["name"] == agent_name:
            return list(agent.get("skills", []))
    return []


def _subagents(manifest: dict[str, Any], agent_name: str) -> list[str]:
    for agent in manifest.get("agents", []):
        if agent["name"] == agent_name:
            return list(agent.get("subagents", []))
    return []


def _reachable(manifest: dict[str, Any]) -> set[str]:
    return {agent["name"] for agent in manifest.get("agents", []) if agent.get("reachable", True)}


def _raw_item_to_dict(item: Any) -> dict[str, Any]:
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return {"type": getattr(item, "type", type(item).__name__)}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)
    return {"repr": repr(raw), "type": getattr(item, "type", type(item).__name__)}


def _make_tool(agent_name: str, tool_name: str):
    def _tool(ctx: RunContextWrapper[FrameworkTrace]) -> str:
        trace: FrameworkTrace = ctx.context
        if trace.runtime_state is None:
            raise RuntimeError("runtime state is required for live tool execution")
        if tool_name not in trace.used_tools:
            trace.used_tools.append(tool_name)
        if (agent_name, tool_name) not in trace.used_pairs:
            trace.used_pairs.append((agent_name, tool_name))
        return trace.runtime_state.execute_tool(tool_name, agent_name)

    _tool.__name__ = f"tool_{tool_name}"
    _tool.__doc__ = (
        f"Execute the live local backend for tool '{tool_name}' when the task explicitly requires it."
    )
    return function_tool(_tool, name_override=tool_name)


def _make_skill_backend(agent_name: str, skill_name: str):
    def _tool(ctx: RunContextWrapper[FrameworkTrace]) -> str:
        trace: FrameworkTrace = ctx.context
        if trace.runtime_state is None:
            raise RuntimeError("runtime state is required for live skill execution")
        if skill_name not in trace.used_skills:
            trace.used_skills.append(skill_name)
        return trace.runtime_state.execute_skill(skill_name, agent_name)

    _tool.__name__ = f"use_skill_{skill_name}"
    _tool.__doc__ = f"Execute the live local backend for skill '{skill_name}'."
    return function_tool(_tool, name_override=f"use_skill__{skill_name}")


def _make_denial_probe(denied: set[str]):
    def _tool(ctx: RunContextWrapper[FrameworkTrace], tool_name: str) -> str:
        trace: FrameworkTrace = ctx.context
        if tool_name in denied and tool_name not in trace.denied_tools:
            trace.denied_tools.append(tool_name)
            return f"tool {tool_name} denied"
        return f"tool {tool_name} not denied"

    _tool.__name__ = "probe_denied_tool"
    _tool.__doc__ = "Record a denied-tool attempt when the task asks for a forbidden tool."
    return function_tool(_tool, name_override="probe_denied_tool")


def _instructions_for_agent(manifest: dict[str, Any], agent_name: str) -> str:
    allowed = sorted(_allowed_tools(manifest, agent_name))
    denied = sorted(_denied_tools(manifest, agent_name))
    skills = sorted(_skills(manifest, agent_name))
    children = sorted(_subagents(manifest, agent_name))
    return (
        f"You are agent '{agent_name}'. "
        f"Allowed tools: {allowed}. "
        f"Denied tools: {denied}. "
        f"Skills: {skills}. "
        f"Subagents available for handoff: {children}. "
        "Follow these rules strictly: "
        "if the task says it requires tool X, call tool X exactly once. "
        "if the task says invoke or rely on skill Y, call use_skill__Y exactly once. "
        "if the task says forbidden tool X, call probe_denied_tool with tool_name=X. "
        "if the task says it requires delegation to subagent Z, hand off to Z. "
        "Do not invent tools or subagents not listed above. "
        "After completing the required action, briefly summarize what you did."
    )


def build_openai_agent_graph(manifest: dict[str, Any], *, model: str | None = None) -> dict[str, Agent]:
    model_name = model or DEFAULT_MODEL
    reachable = _reachable(manifest)
    ensure_tool_backends({tool["name"] for tool in manifest.get("tools", [])})
    cache: dict[str, Agent] = {}

    def build(agent_name: str) -> Agent:
        if agent_name in cache:
            return cache[agent_name]

        tools = [_make_tool(agent_name, tool_name) for tool_name in sorted(_allowed_tools(manifest, agent_name))]
        tools.extend(
            _make_skill_backend(agent_name, skill_name) for skill_name in sorted(_skills(manifest, agent_name))
        )
        tools.append(_make_denial_probe(_denied_tools(manifest, agent_name)))

        agent = Agent(
            name=agent_name,
            handoff_description=f"Handoff to subagent '{agent_name}'.",
            instructions=_instructions_for_agent(manifest, agent_name),
            tools=tools,
            handoffs=[],
            model=model_name,
        )
        cache[agent_name] = agent

        handoffs = [build(child) for child in sorted(_subagents(manifest, agent_name)) if child in reachable]
        agent.handoffs.extend(handoffs)
        return agent

    for agent_name in sorted(reachable):
        build(agent_name)
    return cache


def execute_generated_test_with_openai_framework(
    manifest: dict[str, Any],
    test: GeneratedTest,
    *,
    model: str | None = None,
) -> FrameworkExecution:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required to execute the OpenAI Agents SDK runtime."
        )

    agents = build_openai_agent_graph(manifest, model=model)
    if test.target_agent not in agents:
        raise ValueError(f"target agent '{test.target_agent}' is not reachable in the manifest")

    context = FrameworkTrace(runtime_state=LocalRuntimeState())
    result = Runner.run_sync(
        agents[test.target_agent],
        test.scenario,
        context=context,
        max_turns=6,
    )

    delegated_edges: list[tuple[str, str]] = []
    for item in result.new_items:
        if getattr(item, "type", None) == "handoff_output_item":
            source = getattr(getattr(item, "source_agent", None), "name", None)
            target = getattr(getattr(item, "target_agent", None), "name", None)
            if source and target:
                delegated_edges.append((source, target))

    observation = CoverageObservation(
        target_agent=test.target_agent,
        mode=test.mode,
        tools_used=set(context.used_tools),
        denied_tools=set(context.denied_tools),
        skills_used=set(context.used_skills),
        delegated_edges=set(delegated_edges),
        tool_calls=set(context.used_pairs),
    )

    return FrameworkExecution(
        observation=observation,
        final_output=str(result.final_output),
        delegated_edges=delegated_edges,
        raw_items=[_raw_item_to_dict(item) for item in result.new_items],
    )


def manifest_from_path(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
