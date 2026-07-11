"""Mutation-score aggregation library.

Three orthogonal axes the runner iterates over:

  axis 1 — test suite      (T-Dev*, T-Rand*, T-E2E*, T-Struct)
  axis 2 — kill criterion  (K-Struct, K-Outcome, K-Safety)
  axis 3 — operator        (12 operators in 4 families)

For each (suite, criterion, operator) we produce two scores:

  MS_strict  — denominator = killable mutants only (excludes graph-
               equivalent and add-only mutants under K-Struct, and
               UNKILLABLE mutants under K-Outcome/K-Safety where the
               simulator cannot witness the perturbation).
  MS_liberal — denominator = total mutants minus graph-equivalent only.
               This includes add-only mutants and exposes K-Struct's
               intrinsic blind spot in the headline number.

The runner also caps each (operator, benchmark) cell at MAX_PER_CELL
mutants via a seeded uniform sample, so a single combinatorially-heavy
operator on a large benchmark does not dominate the mean.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .operators import OPERATORS, OPERATOR_FAMILIES, Mutant
from .simulator import Scenario, simulate
from .kill_struct import kill_struct_against_suite
from .kill_runtime import kill_outcome, kill_safety


CRITERIA = ["K-Struct", "K-Outcome", "K-Safety"]


# ---------------------------------------------------------------------------
# Mutant classification under K-Struct
# ---------------------------------------------------------------------------

def classify_kstruct(orig_obs: set[tuple], mut_obs: Optional[set[tuple]]) -> str:
    """Return one of 'graph_equiv', 'add_only', 'killable', 'invalid'."""
    if mut_obs is None:
        return "invalid"
    if mut_obs == orig_obs:
        return "graph_equiv"
    diff = mut_obs ^ orig_obs
    if not (diff & orig_obs):
        return "add_only"
    return "killable"


# ---------------------------------------------------------------------------
# Operator capping
# ---------------------------------------------------------------------------

def cap_per_operator(mutants: list[Mutant], max_per_op: int,
                     rng: random.Random) -> list[Mutant]:
    """Sample at most ``max_per_op`` mutants per operator, deterministically."""
    grouped: dict[str, list[Mutant]] = {}
    for m in mutants:
        grouped.setdefault(m.operator, []).append(m)
    capped: list[Mutant] = []
    for op_name in sorted(grouped):
        bucket = grouped[op_name]
        if len(bucket) <= max_per_op:
            capped.extend(bucket)
        else:
            capped.extend(rng.sample(bucket, k=max_per_op))
    return capped


# ---------------------------------------------------------------------------
# Per-benchmark scoring
# ---------------------------------------------------------------------------

@dataclass
class CellScore:
    """One (operator, suite, criterion) cell across one or more benchmarks.

    Denominator policy:

      * ``invalid`` mutants are mutants whose manifest could not be
        parsed back into a graph (schema violations introduced by the
        operator). They are NEVER counted as killed and are excluded
        from BOTH the strict and the liberal mutation-score
        denominators. They are reported separately via
        ``well_formed_rate`` and ``invalid_rate``.
      * Equivalence is tracked per criterion in three distinct
        buckets so that ``ms_liberal`` is well-defined under each
        criterion:

          - ``graph_equiv``    -- K-Struct equivalence (obligation set
                                  unchanged).
          - ``outcome_equiv``  -- K-Outcome equivalence (deterministic
                                  simulator cannot witness any trace
                                  divergence on T-Struct).
          - ``safety_equiv``   -- K-Safety equivalence (deterministic
                                  simulator cannot witness any
                                  forbidden-tool call on T-Struct).

        For any given (operator, suite, criterion) cell, at most one of
        these three buckets is populated by ``score_benchmark``; the
        liberal denominator subtracts whichever bucket applies, so
        K-Outcome and K-Safety are not contaminated by K-Struct's
        graph-equivalent count.
      * ``add_only`` mutants strictly grow the obligation set; they are
        excluded from the strict denominator but kept in the liberal
        denominator to expose K-Struct's add-only blind spot.
    """
    total: int = 0
    graph_equiv: int = 0       # K-Struct  equivalence bucket
    outcome_equiv: int = 0     # K-Outcome equivalence bucket
    safety_equiv: int = 0      # K-Safety  equivalence bucket
    add_only: int = 0
    invalid: int = 0
    killable: int = 0          # mutants we could in principle kill under this criterion
    killed: int = 0

    @property
    def ms_strict(self) -> float:
        # Strict: kills divided by killable mutants (invalid are not
        # killable; the score_benchmark loop never increments killable
        # for an invalid mutant).
        return self.killed / self.killable if self.killable > 0 else float("nan")

    @property
    def ms_liberal(self) -> float:
        # Liberal denominator excludes the cell's own equivalence bucket
        # AND invalid mutants. Only one of the three equiv buckets is
        # populated per cell (by criterion), so summing them yields the
        # correct per-cell denominator without storing the criterion
        # name on the CellScore itself. Add-only mutants stay in the
        # denominator to surface the K-Struct blind spot.
        denom = (self.total
                 - self.graph_equiv
                 - self.outcome_equiv
                 - self.safety_equiv
                 - self.invalid)
        return self.killed / denom if denom > 0 else float("nan")

    @property
    def well_formed_rate(self) -> float:
        """Fraction of generated mutants that parse back into a graph."""
        if self.total <= 0:
            return float("nan")
        return (self.total - self.invalid) / self.total

    @property
    def invalid_rate(self) -> float:
        """Fraction of generated mutants that fail to parse."""
        if self.total <= 0:
            return float("nan")
        return self.invalid / self.total

    def merge(self, other: "CellScore") -> "CellScore":
        return CellScore(
            total=self.total + other.total,
            graph_equiv=self.graph_equiv + other.graph_equiv,
            outcome_equiv=self.outcome_equiv + other.outcome_equiv,
            safety_equiv=self.safety_equiv + other.safety_equiv,
            add_only=self.add_only + other.add_only,
            invalid=self.invalid + other.invalid,
            killable=self.killable + other.killable,
            killed=self.killed + other.killed,
        )


@dataclass
class BenchmarkScore:
    benchmark: str
    suite_sizes: dict[str, int] = field(default_factory=dict)
    cells: dict[tuple[str, str, str], CellScore] = field(default_factory=dict)
    # cells[(operator, suite, criterion)] -> CellScore

    def cell(self, op: str, suite: str, criterion: str) -> CellScore:
        return self.cells.setdefault((op, suite, criterion), CellScore())


def score_benchmark(
    benchmark: str,
    manifest: dict,
    mutants: list[Mutant],
    suites: dict[str, list[tuple]],
    orig_obs: set[tuple],
    *,
    enable_runtime: bool = True,
) -> BenchmarkScore:
    """Score one benchmark across (mutants × suites × criteria)."""
    bs = BenchmarkScore(benchmark=benchmark,
                        suite_sizes={s: len(o) for s, o in suites.items()})

    # Precompute per-suite scenario lists for the runtime simulator.
    suite_scenarios: dict[str, list[Scenario]] = {
        s: [Scenario(f"{benchmark}::{s}::{i}", o) for i, o in enumerate(obs)]
        for s, obs in suites.items()
    }

    for mut in mutants:
        # Pre-classify under K-Struct using the mutant's obligation set.
        try:
            from .equivalence import _obligation_set
            mut_obs = _obligation_set(mut.manifest)
        except Exception:
            mut_obs = None
        klass = classify_kstruct(orig_obs, mut_obs)

        for suite_name, obs_list in suites.items():
            obs_set = set(obs_list)

            # ---- K-Struct ----
            cell = bs.cell(mut.operator, suite_name, "K-Struct")
            cell.total += 1
            if klass == "graph_equiv":
                cell.graph_equiv += 1
            elif klass == "add_only":
                cell.add_only += 1
            elif klass == "invalid":
                # Invalid mutants are not killed by any suite; they are
                # excluded from both denominators and reported via
                # well_formed_rate.
                cell.invalid += 1
            else:
                cell.killable += 1
                if kill_struct_against_suite(manifest, mut.manifest, obs_set):
                    cell.killed += 1

            if not enable_runtime:
                continue

            # ---- K-Outcome ----
            cell = bs.cell(mut.operator, suite_name, "K-Outcome")
            cell.total += 1
            if klass == "invalid":
                # Cannot simulate a graph that does not parse.
                cell.invalid += 1
            else:
                # graph_equiv -> may still be killed by K-Outcome (Tool-Fault)
                killed_out, _ = kill_outcome(manifest, mut.manifest, suite_scenarios[suite_name])
                # Killability under K-Outcome: a mutant is killable iff the
                # simulator can observe a divergence on SOME suite (we use
                # T-Struct as the killability oracle to be consistent across
                # suites). Otherwise the mutant is K-Outcome-equivalent and
                # excluded from the strict denominator.
                killable_out, _ = kill_outcome(
                    manifest, mut.manifest, suite_scenarios["T-Struct"]
                )
                if killable_out:
                    cell.killable += 1
                    if killed_out:
                        cell.killed += 1
                else:
                    # Outcome-equivalent under the deterministic simulator.
                    cell.outcome_equiv += 1

            # ---- K-Safety ----
            cell = bs.cell(mut.operator, suite_name, "K-Safety")
            cell.total += 1
            if klass == "invalid":
                cell.invalid += 1
            else:
                killed_saf, _ = kill_safety(manifest, mut.manifest, suite_scenarios[suite_name])
                killable_saf, _ = kill_safety(
                    manifest, mut.manifest, suite_scenarios["T-Struct"]
                )
                if killable_saf:
                    cell.killable += 1
                    if killed_saf:
                        cell.killed += 1
                else:
                    # Safety-equivalent under the deterministic simulator.
                    cell.safety_equiv += 1

    return bs


# ---------------------------------------------------------------------------
# Aggregation across benchmarks
# ---------------------------------------------------------------------------

def aggregate_cells(
    per_benchmark: Iterable[BenchmarkScore],
) -> dict[tuple[str, str, str], CellScore]:
    out: dict[tuple[str, str, str], CellScore] = {}
    for bs in per_benchmark:
        for key, cell in bs.cells.items():
            out[key] = out.get(key, CellScore()).merge(cell)
    return out


def per_family_score(
    cells: dict[tuple[str, str, str], CellScore],
    suite: str, criterion: str,
) -> dict[str, CellScore]:
    out: dict[str, CellScore] = {}
    for fam, ops in OPERATOR_FAMILIES.items():
        acc = CellScore()
        for op in ops:
            c = cells.get((op, suite, criterion))
            if c is not None:
                acc = acc.merge(c)
        out[fam] = acc
    return out


def overall_score(
    cells: dict[tuple[str, str, str], CellScore],
    suite: str, criterion: str,
) -> CellScore:
    acc = CellScore()
    for op in (o.name for o in OPERATORS):
        c = cells.get((op, suite, criterion))
        if c is not None:
            acc = acc.merge(c)
    return acc
