"""Mutation testing for multi-agent workflows.

Sibling package of ``agentcov``. Operates on the same schema-v3 manifest:

* ``operators`` — 14 mutation operators across 4 families:
  Del-Rem / Del-Red / Del-Inv / Del-Add        (delegation faults)
  Tool-Rem / Tool-Red / Tool-Fault / Tool-SigMut (tool faults)
  Res-Relax / Res-Tighten / Res-Rem            (restriction faults)
  Ag-Rem / Ag-Swap / Ag-PromptDrop             (agent / instruction faults)
* ``equivalence`` — static + heuristic equivalent-mutant filter
* ``kill_struct`` — structural kill criterion K-Struct: a mutant is killed
  if its obligation set (under agentcov.extract_objectives) differs from
  the original's set. No LLM execution required.
* ``runner`` — generate mutants for one manifest and score them under
  K-Struct against an arbitrary deterministic test suite (the structural
  suite has full coverage by construction, so K-Struct kills 100% of
  non-equivalent mutants; the interesting numbers come from running
  K-Outcome/K-Safety against under-specified suites).

"""

from __future__ import annotations

from .operators import (
    Mutant,
    MutationOperator,
    OPERATORS,
    OPERATOR_FAMILIES,
    apply_operator,
    generate_mutants,
)
from .equivalence import is_equivalent
from .kill_struct import kill_struct, kill_struct_against_suite
from .simulator import Scenario, Trace, TraceStep, simulate
from .kill_runtime import kill_outcome, kill_safety
from .suites import SUITES, build_suites
from .scoring import (
    CRITERIA,
    BenchmarkScore,
    CellScore,
    aggregate_cells,
    cap_per_operator,
    classify_kstruct,
    overall_score,
    per_family_score,
    score_benchmark,
)

__all__ = [
    "Mutant",
    "MutationOperator",
    "OPERATORS",
    "OPERATOR_FAMILIES",
    "apply_operator",
    "generate_mutants",
    "CRITERIA",
    "BenchmarkScore",
    "CellScore",
    "aggregate_cells",
    "cap_per_operator",
    "classify_kstruct",
    "overall_score",
    "per_family_score",
    "score_benchmark",
    "is_equivalent",
    "kill_struct",
    "kill_struct_against_suite",
    "Scenario",
    "Trace",
    "TraceStep",
    "simulate",
    "kill_outcome",
    "kill_safety",
    "SUITES",
    "build_suites",
]
