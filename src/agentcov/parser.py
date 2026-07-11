"""Strict schema v3 parser and validator.

Loads a workflow manifest from JSON and validates it. Hard-fails with
clear error messages on any departure from the schema.

Validation rules:
  R1. schema_version must equal "3.0".
  R2. system.entry_agent must be a declared agent and reachable=true.
  R3. Every referenced agent (in permissions, delegations) must be declared.
  R4. Every referenced tool (in permissions) must be declared.
  R5. No agent-tool pair may appear in both allow and restrict.
  R6. Every tool must declare a non-empty purpose, side_effects, and a
      binding (module + callable) that imports successfully.
  R7. Delegation endpoints must both be declared agents.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .graph_model import (
    SCHEMA_VERSION,
    AgentSpec,
    DelegationEdge,
    SideEffect,
    SystemSpec,
    ToolBinding,
    ToolSpec,
    WorkflowGraph,
)


class SchemaError(ValueError):
    """Raised when a manifest does not conform to schema v3."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _require_keys(payload: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise SchemaError(f"{context}: missing required keys {missing}")


def _parse_tool(payload: dict[str, Any], validate_bindings: bool) -> ToolSpec:
    _require_keys(payload, ["name", "purpose", "side_effects", "binding"], "tool")

    binding_payload = payload["binding"]
    _require(isinstance(binding_payload, dict), "tool.binding must be an object")
    _require_keys(binding_payload, ["module", "callable"], "tool.binding")

    if validate_bindings:
        try:
            module = importlib.import_module(binding_payload["module"])
        except Exception as exc:  # noqa: BLE001
            raise SchemaError(
                f"tool '{payload['name']}' binding module '{binding_payload['module']}' "
                f"failed to import: {exc}"
            ) from exc

        if not hasattr(module, binding_payload["callable"]):
            raise SchemaError(
                f"tool '{payload['name']}' binding callable "
                f"'{binding_payload['module']}.{binding_payload['callable']}' not found"
            )

    return ToolSpec(
        name=payload["name"],
        purpose=payload["purpose"],
        side_effects=SideEffect(payload["side_effects"]),
        binding=ToolBinding(module=binding_payload["module"], callable=binding_payload["callable"]),
    )


def _parse_agent(payload: dict[str, Any]) -> AgentSpec:
    _require_keys(
        payload,
        ["name", "reachable", "role", "description", "system_prompt_hint"],
        "agent",
    )
    return AgentSpec(
        name=payload["name"],
        reachable=bool(payload["reachable"]),
        role=payload["role"],
        description=payload["description"],
        system_prompt_hint=payload["system_prompt_hint"],
    )


def _parse_delegation(payload: dict[str, Any]) -> DelegationEdge:
    _require_keys(payload, ["from", "to", "trigger"], "delegation")
    return DelegationEdge(parent=payload["from"], child=payload["to"], trigger=payload["trigger"])


def graph_from_dict(data: dict[str, Any], *, validate_bindings: bool = True) -> WorkflowGraph:
    _require_keys(data, ["schema_version", "system", "agents", "tools", "permissions", "delegations"], "manifest")
    _require(
        data["schema_version"] == SCHEMA_VERSION,
        f"unsupported schema_version {data['schema_version']!r}; expected {SCHEMA_VERSION!r}",
    )

    # ---- system
    system_payload = data["system"]
    _require(isinstance(system_payload, dict), "system must be an object")
    _require_keys(system_payload, ["name", "entry_agent", "description"], "system")
    system = SystemSpec(
        name=system_payload["name"],
        entry_agent=system_payload["entry_agent"],
        description=system_payload["description"],
    )

    # ---- agents
    agents: dict[str, AgentSpec] = {}
    for agent_payload in data["agents"]:
        spec = _parse_agent(agent_payload)
        if spec.name in agents:
            raise SchemaError(f"duplicate agent declaration: {spec.name}")
        agents[spec.name] = spec
    _require(
        system.entry_agent in agents,
        f"system.entry_agent '{system.entry_agent}' is not a declared agent",
    )
    _require(
        agents[system.entry_agent].reachable,
        f"entry_agent '{system.entry_agent}' must have reachable=true",
    )

    # ---- tools
    tools: dict[str, ToolSpec] = {}
    for tool_payload in data["tools"]:
        spec = _parse_tool(tool_payload, validate_bindings=validate_bindings)
        if spec.name in tools:
            raise SchemaError(f"duplicate tool declaration: {spec.name}")
        tools[spec.name] = spec

    # ---- permissions
    permissions = data["permissions"]
    _require(isinstance(permissions, dict), "permissions must be an object")
    allow_payload = permissions.get("allow", [])
    restrict_payload = permissions.get("restrict", [])
    _require(isinstance(allow_payload, list), "permissions.allow must be a list")
    _require(isinstance(restrict_payload, list), "permissions.restrict must be a list")

    allow_edges: set[tuple[str, str]] = set()
    for entry in allow_payload:
        _require_keys(entry, ["agent", "tool"], "permissions.allow entry")
        _require(entry["agent"] in agents, f"allow references unknown agent '{entry['agent']}'")
        _require(entry["tool"] in tools, f"allow references unknown tool '{entry['tool']}'")
        allow_edges.add((entry["agent"], entry["tool"]))

    restrict_edges: set[tuple[str, str]] = set()
    for entry in restrict_payload:
        _require_keys(entry, ["agent", "tool"], "permissions.restrict entry")
        _require(entry["agent"] in agents, f"restrict references unknown agent '{entry['agent']}'")
        _require(entry["tool"] in tools, f"restrict references unknown tool '{entry['tool']}'")
        restrict_edges.add((entry["agent"], entry["tool"]))

    conflict = allow_edges & restrict_edges
    _require(
        not conflict,
        f"agent-tool pair(s) {sorted(conflict)} appear in both allow and restrict",
    )

    # ---- delegations
    delegations_payload = data["delegations"]
    _require(isinstance(delegations_payload, list), "delegations must be a list")
    delegations: list[DelegationEdge] = []
    for delegation_payload in delegations_payload:
        edge = _parse_delegation(delegation_payload)
        _require(edge.parent in agents, f"delegation references unknown agent '{edge.parent}'")
        _require(edge.child in agents, f"delegation references unknown agent '{edge.child}'")
        delegations.append(edge)

    return WorkflowGraph(
        schema_version=SCHEMA_VERSION,
        system=system,
        agents=agents,
        tools=tools,
        allow_edges=allow_edges,
        restrict_edges=restrict_edges,
        delegations=tuple(delegations),
    )


def graph_from_json(path: str | Path, *, validate_bindings: bool = True) -> WorkflowGraph:
    with Path(path).open("r", encoding="utf-8") as handle:
        return graph_from_dict(json.load(handle), validate_bindings=validate_bindings)


def main() -> None:
    """CLI: validate a manifest file and print a summary."""

    import argparse

    cli = argparse.ArgumentParser(description="Validate a schema v3 workflow manifest.")
    cli.add_argument("path", help="Path to the manifest JSON file.")
    args = cli.parse_args()

    graph = graph_from_json(args.path)
    print(f"OK: schema {graph.schema_version}, system={graph.system.name}")
    print(f"  agents:      {len(graph.agents)} (reachable={len(graph.reachable_agents)})")
    print(f"  tools:       {len(graph.tools)}")
    print(f"  allow:       {len(graph.allow_edges)} ({len(graph.allowed_obligations)} reachable)")
    print(f"  restrict:    {len(graph.restrict_edges)} ({len(graph.restricted_obligations)} reachable)")
    print(f"  delegations: {len(graph.delegations)} ({len(graph.delegation_obligations)} reachable)")


if __name__ == "__main__":
    main()
