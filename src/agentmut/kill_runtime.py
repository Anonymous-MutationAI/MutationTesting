"""Runtime kill criteria K-Outcome and K-Safety.

Both criteria run the deterministic trace simulator against the
original spec and each mutant, and compare results.

K-Outcome: a mutant is killed if the trace verdict OR the step
sequence differs from the original for at least one scenario in the
suite. This catches mutants that change the runtime behaviour of any
test, including those that K-Struct cannot see because they leave the
obligation set unchanged (Tool-Fault, Ag-PromptDrop, Tool-SigMut).

K-Safety: a mutant is killed if any scenario in the suite produces a
trace step that fires a tool that is in restrict for that agent in the
*original* spec. This catches Res-Relax / Res-Rem mutants that silently
allow forbidden calls; K-Struct counts these as killable only because
the obligation set differs, but K-Safety is the criterion that matches
the safety property the paper cares about.
"""

from __future__ import annotations

from typing import Iterable

from .simulator import Scenario, Trace, simulate


def kill_outcome(
    original: dict, mutant: dict, scenarios: Iterable[Scenario]
) -> tuple[bool, list[str]]:
    """K-Outcome verdict for one (original, mutant) pair over a scenario set.

    Returns (killed, list_of_diverging_scenario_ids).
    """
    diverging: list[str] = []
    for sc in scenarios:
        t_orig = simulate(original, sc)
        t_mut = simulate(mutant, sc)
        if (t_orig.verdict, t_orig.steps) != (t_mut.verdict, t_mut.steps):
            diverging.append(sc.scenario_id)
    return (len(diverging) > 0, diverging)


def kill_safety(
    original: dict, mutant: dict, scenarios: Iterable[Scenario]
) -> tuple[bool, list[str]]:
    """K-Safety verdict for one (original, mutant) pair over a scenario set.

    A mutant is killed if executing it under the suite causes any tool
    call that would be forbidden under the original spec's restrict set.
    """
    restrict = {(e["agent"], e["tool"])
                for e in original.get("permissions", {}).get("restrict", [])}
    violations: list[str] = []
    for sc in scenarios:
        t_mut = simulate(mutant, sc)
        for step in t_mut.steps:
            if step.action == "tool_ok" and (step.agent, step.target) in restrict:
                violations.append(sc.scenario_id)
                break
    return (len(violations) > 0, violations)
