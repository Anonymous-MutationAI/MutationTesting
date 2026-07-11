"""Coverage model scaffolding for Paper B."""

from .graph_model import CoverageObservation, SkillGraph, ToolKind
from .generator import (
    GeneratedTest,
    generate_suite,
    generate_suite_without_delegation_rule,
    generate_suite_without_denial_rule,
    generate_suite_without_mode_repair,
    generate_suite_without_skill_rule,
    generate_suite_without_tool_rule,
)
from .metrics import adequacy_score, coverage_report
from .parser import graph_from_dict, graph_from_json

__all__ = [
    "CoverageObservation",
    "GeneratedTest",
    "SkillGraph",
    "ToolKind",
    "adequacy_score",
    "coverage_report",
    "generate_suite",
    "generate_suite_without_delegation_rule",
    "generate_suite_without_denial_rule",
    "generate_suite_without_mode_repair",
    "generate_suite_without_skill_rule",
    "generate_suite_without_tool_rule",
    "graph_from_dict",
    "graph_from_json",
]
