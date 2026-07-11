"""Structural kill criterion K-Struct.

K-Struct is a *purely structural* criterion that does not need to run
the LLM. A test suite kills a mutant under K-Struct iff at least one of
its obligations is satisfiable in the original specification but not in
the mutant, or vice versa — equivalently, iff the obligation sets differ.

For the structural suite T-Struct (Section II.B of the paper), every
obligation is by construction part of the suite, so T-Struct under
K-Struct kills the union of all non-equivalent mutants. The interesting
contrast comes from comparing different suites under K-Outcome /
K-Safety in a runtime experiment.
"""

from __future__ import annotations

from typing import Iterable

from .equivalence import _obligation_set


def kill_struct(original: dict, mutant: dict) -> bool:
    """K-Struct verdict for one (original, mutant) pair.

    True  -> mutant is killed (obligation sets differ)
    False -> mutant survives  (graph-equivalent under K-Struct)
    """
    try:
        return _obligation_set(original) != _obligation_set(mutant)
    except Exception:
        # Schema-invalid mutant: definitely killed by any deterministic
        # suite that parses the manifest.
        return True


def kill_struct_against_suite(
    original: dict, mutant: dict, suite_obligations: Iterable[tuple],
) -> bool:
    """K-Struct verdict against a *specific* suite.

    A suite kills a mutant if any of its obligations is in the symmetric
    difference of the original and mutant obligation sets.
    """
    try:
        diff = _obligation_set(original) ^ _obligation_set(mutant)
    except Exception:
        return True
    return any(o in diff for o in suite_obligations)
