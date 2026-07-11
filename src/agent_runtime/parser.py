from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from .graph_model import SkillGraph, ToolKind


def _coerce_tool_kind(value: str) -> ToolKind:
    normalized = value.strip().lower()
    if normalized == "mcp":
        return ToolKind.MCP
    if normalized in {"exe", "executable", "shell"}:
        return ToolKind.EXE
    if normalized in {"py", "python"}:
        return ToolKind.PY
    raise ValueError(f"unsupported tool kind: {value}")


def graph_from_dict(data: dict[str, Any]) -> SkillGraph:
    """
    Build a SkillGraph from a normalized manifest-like dictionary.

    Expected shape:
    {
        "agents": [{"name": "...", "reachable": true, "subagents": [...], "skills": [...]}],
        "tools": [{"name": "...", "kind": "mcp|exe|py"}],
        "permissions": {
            "allow": [{"agent": "...", "tool": "..."}],
            "deny": [{"agent": "...", "tool": "..."}]
        }
    }
    """
    agents = {agent["name"] for agent in data.get("agents", [])}
    reachable_agents = {
        agent["name"] for agent in data.get("agents", []) if agent.get("reachable", True)
    }
    tools = {tool["name"] for tool in data.get("tools", [])}
    skills = {
        skill
        for agent in data.get("agents", [])
        for skill in agent.get("skills", [])
    }

    allow_edges = {
        (item["agent"], item["tool"])
        for item in data.get("permissions", {}).get("allow", [])
    }
    deny_edges = {
        (item["agent"], item["tool"])
        for item in data.get("permissions", {}).get("deny", [])
    }
    subagent_edges = {
        (agent["name"], subagent)
        for agent in data.get("agents", [])
        for subagent in agent.get("subagents", [])
    }
    skill_edges = {
        (agent["name"], skill)
        for agent in data.get("agents", [])
        for skill in agent.get("skills", [])
    }
    tool_kinds = {
        tool["name"]: _coerce_tool_kind(tool["kind"])
        for tool in data.get("tools", [])
        if "kind" in tool
    }

    return SkillGraph(
        agents=agents,
        reachable_agents=reachable_agents,
        tools=tools,
        skills=skills,
        allow_edges=allow_edges,
        deny_edges=deny_edges,
        subagent_edges=subagent_edges,
        skill_edges=skill_edges,
        imports=set(data.get("imports", [])),
        tool_kinds=tool_kinds,
    )


def graph_from_json(path: str | Path) -> SkillGraph:
    with Path(path).open("r", encoding="utf-8") as handle:
        return graph_from_dict(json.load(handle))
