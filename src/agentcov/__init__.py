"""Coverage library for agent workflows (schema v3).

Self-contained vendor copy used by the replication scripts. Defines the
workflow graph model, manifest parser, witness objective extraction, and
the four coverage criteria (C1 reach, C2 use_tool, C4 restrict_tool,
C5 delegate) used in the paper.
"""

from __future__ import annotations

from .graph_model import (
    SCHEMA_VERSION,
    AgentSpec,
    CoverageObservation,
    DelegationEdge,
    SideEffect,
    SystemSpec,
    ToolBinding,
    ToolSpec,
    WorkflowGraph,
)
from .metrics import (
    adequacy_score,
    c1_agent,
    c2_allowed_tool,
    c4_restricted_tool,
    c5_delegation,
    coverage_report,
)
from .objectives import (
    Objective,
    ObjectiveKind,
    extract_objectives,
)
from .parser import (
    SchemaError,
    graph_from_dict,
    graph_from_json,
)
from .generator import (
    TestCase,
    generate_test_cases,
    iter_test_cases,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentSpec",
    "CoverageObservation",
    "DelegationEdge",
    "SideEffect",
    "SystemSpec",
    "ToolBinding",
    "ToolSpec",
    "WorkflowGraph",
    "SchemaError",
    "graph_from_dict",
    "graph_from_json",
    "Objective",
    "ObjectiveKind",
    "extract_objectives",
    "TestCase",
    "generate_test_cases",
    "iter_test_cases",
    "c1_agent",
    "c2_allowed_tool",
    "c4_restricted_tool",
    "c5_delegation",
    "coverage_report",
    "adequacy_score",
]
