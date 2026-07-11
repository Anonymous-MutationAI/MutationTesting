"""Equivalent-mutant filter.

Two mutants are *structurally equivalent* to the original if their
agentcov obligation sets are identical. This is a sound, conservative
filter for K-Struct: a mutant that produces the same obligations
contributes nothing to detection by any deterministic structural suite.

Heuristic equivalences (Tool-Fault, Tool-SigMut, Ag-PromptDrop) leave
the graph unchanged and are therefore K-Struct-equivalent by definition;
we still report them so that K-Outcome and K-Safety experiments can pick
them up.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling ``agentcov`` package importable when this module is
# loaded directly from the source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcov import (  # noqa: E402
    extract_objectives,
    graph_from_dict,
)


def _obligation_set(manifest: dict) -> frozenset[tuple]:
    g = graph_from_dict(manifest, validate_bindings=False)
    return frozenset(o.witness_key() for o in extract_objectives(g))


def is_equivalent(original: dict, mutant: dict) -> bool:
    """True iff the mutant's obligation set equals the original's."""
    try:
        return _obligation_set(original) == _obligation_set(mutant)
    except Exception:
        # A mutant whose schema no longer parses is *not* equivalent —
        # any deterministic suite would catch the parser error.
        return False
