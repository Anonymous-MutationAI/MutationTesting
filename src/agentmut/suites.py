"""Baseline test suites for mutation testing.


  T-Dev*    — entry-path proxy for developer-shipped scenarios.
              The SDK examples we ingest are interactive demos with no
              static scenario file; we therefore approximate "what a
              developer's smoke-test exercises" as the obligations on
              the BFS frontier of distance <= D_DEV from the entry agent,
              plus the allow edges of those reachable agents. D_DEV = 2.

  T-Rand*   — random-obligation proxy for random LLM prompts.
              We sample RAND_RATIO = 10% of all obligations uniformly
              with a seeded RNG. This OVERESTIMATES the structural
              coverage of a random-prompt suite (a random prompt is not
              guaranteed to exercise any obligation) and so provides a
              CONSERVATIVE proxy: real-suite scores will be lower.

  T-E2E*    — end-to-end proxy.
              Includes (a) one reach obligation per reachable agent
              (the agent is observed in some trace) AND (b) one
              use_tool obligation per reachable agent that has any
              allow edge (the task suite must trigger that agent's
              primary capability). Excludes negative and structural
              corner cases.

  T-Struct  — the structural witness suite from agentcov.
              By construction this equals the full obligation set; no
              proxy required.
"""

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentcov import extract_objectives, graph_from_dict  # noqa: E402


SUITES = ["T-Dev*", "T-Rand*", "T-E2E*", "T-Struct"]

D_DEV = 2
RAND_RATIO = 0.10


def _all_obligations(manifest: dict) -> list[tuple]:
    g = graph_from_dict(manifest, validate_bindings=False)
    return [o.witness_key() for o in extract_objectives(g)]


def _bfs_at_depth(manifest: dict, depth: int) -> set[str]:
    entry = manifest["system"]["entry_agent"]
    adj: dict[str, list[str]] = {a["name"]: [] for a in manifest["agents"]}
    for e in manifest["delegations"]:
        adj.setdefault(e["from"], []).append(e["to"])
    seen = {entry: 0}
    q = deque([entry])
    while q:
        cur = q.popleft()
        d = seen[cur]
        if d >= depth:
            continue
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen[nxt] = d + 1
                q.append(nxt)
    return set(seen.keys())


def t_dev_star(manifest: dict) -> list[tuple]:
    near = _bfs_at_depth(manifest, D_DEV)
    out: list[tuple] = [("reach", n) for n in sorted(near)]
    for p in manifest.get("permissions", {}).get("allow", []):
        if p["agent"] in near:
            out.append(("use_tool", p["agent"], p["tool"]))
    return out


def t_rand_star(manifest: dict, rng: random.Random,
                ratio: float = RAND_RATIO) -> list[tuple]:
    obs = _all_obligations(manifest)
    n = max(1, int(round(len(obs) * ratio)))
    return rng.sample(obs, k=min(n, len(obs)))


def t_e2e_star(manifest: dict) -> list[tuple]:
    obs = _all_obligations(manifest)
    out = [o for o in obs if o[0] == "reach"]
    seen_agents: set[str] = set()
    for o in obs:
        if o[0] == "use_tool" and o[1] not in seen_agents:
            out.append(o)
            seen_agents.add(o[1])
    return out


def t_struct(manifest: dict) -> list[tuple]:
    return _all_obligations(manifest)


def build_suites(manifest: dict, rng: random.Random) -> dict[str, list[tuple]]:
    return {
        "T-Dev*":   t_dev_star(manifest),
        "T-Rand*":  t_rand_star(manifest, rng),
        "T-E2E*":   t_e2e_star(manifest),
        "T-Struct": t_struct(manifest),
    }
