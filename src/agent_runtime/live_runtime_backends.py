from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from typing import Callable


ToolBackend = Callable[["LocalRuntimeState", str, str], str]


SEARCH_CORPUS = {
    "policy_note": "Refund approvals require explicit review for sensitive claims.",
    "release_runbook": "Deployment requires green CI, release notes, and rollback verification.",
    "incident_playbook": "Containment starts with host isolation and stakeholder notification.",
    "research_note": "Dataset comparison should consider provenance, schema, and licensing.",
    "faq_entry": "Common customer issues can be routed through the knowledge base and FAQ index.",
}


@dataclass
class LocalRuntimeState:
    """Local executable state used by the real framework-backed runtime."""

    tempdir: tempfile.TemporaryDirectory[str] = field(
        default_factory=lambda: tempfile.TemporaryDirectory(prefix="paper_b_live_runtime_")
    )
    action_log: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.tempdir.name)
        self.workspace = self.root / "workspace"
        self.notes = self.root / "notes"
        self.data = self.root / "data"
        self.logs = self.root / "logs"
        for directory in (self.workspace, self.notes, self.data, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self._seed_files()
        self._seed_json()
        self._seed_database()

    def _seed_files(self) -> None:
        (self.workspace / "brief.txt").write_text(
            "This workspace contains reusable artifacts for agent-skill benchmarks.\n",
            encoding="utf-8",
        )
        (self.workspace / "deployment.md").write_text(
            "Deployment checklist: tests, release notes, rollback, approval.\n",
            encoding="utf-8",
        )
        (self.notes / "summary.md").write_text(
            "Initial notes workspace for live framework-backed runs.\n",
            encoding="utf-8",
        )
        corpus_path = self.data / "search_corpus.json"
        corpus_path.write_text(json.dumps(SEARCH_CORPUS, indent=2), encoding="utf-8")

    def _seed_json(self) -> None:
        fixtures = {
            "customers.json": {
                "acme-retail": {"tier": "gold", "region": "NA", "balance": 129.50},
                "northwind": {"tier": "standard", "region": "EU", "balance": 42.00},
            },
            "tickets.json": {
                "T-100": {"status": "open", "priority": "high", "summary": "refund escalation"},
                "T-200": {"status": "triaged", "priority": "medium", "summary": "policy clarification"},
            },
            "datasets.json": {
                "support_cases": {"rows": 3200, "license": "internal"},
                "incident_events": {"rows": 1800, "license": "internal"},
            },
            "git_host.json": {
                "branches": ["main", "release/1.2", "hotfix/refund-copy"],
                "pull_requests": [{"id": 17, "title": "prepare release notes", "status": "open"}],
            },
            "deploy_api.json": {"environment": "staging", "status": "ready"},
            "ci_dashboard.json": {"latest_run": "green", "coverage": 0.91},
            "audit_api.json": {"entries": ["artifact initialized"]},
            "siem_api.json": {
                "alerts": [
                    {"id": "A-1", "severity": "high", "host": "db-01"},
                    {"id": "A-2", "severity": "medium", "host": "api-02"},
                ]
            },
            "hosts.json": {
                "db-01": {"isolated": False},
                "api-02": {"isolated": False},
            },
            "network.json": {"policy": "restricted-egress", "zone": "internal"},
        }
        for filename, content in fixtures.items():
            (self.data / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")

    def _seed_database(self) -> None:
        db_path = self.data / "runtime.db"
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "create table if not exists metrics (name text primary key, value real)"
            )
            connection.execute("delete from metrics")
            connection.executemany(
                "insert into metrics(name, value) values(?, ?)",
                [
                    ("coverage", 0.91),
                    ("latency_ms", 182.0),
                    ("tickets_open", 2.0),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def _json_path(self, name: str) -> Path:
        return self.data / name

    def read_json(self, name: str) -> dict:
        return json.loads(self._json_path(name).read_text(encoding="utf-8"))

    def write_json(self, name: str, payload: dict) -> None:
        self._json_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record(self, kind: str, name: str, agent_name: str, detail: str) -> None:
        entry = {"kind": kind, "name": name, "agent": agent_name, "detail": detail}
        self.action_log.append(entry)
        log_path = self.logs / "activity.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def execute_tool(self, tool_name: str, agent_name: str) -> str:
        backend = TOOL_BACKENDS[tool_name]
        result = backend(self, tool_name, agent_name)
        self.record("tool", tool_name, agent_name, result)
        return result

    def execute_skill(self, skill_name: str, agent_name: str) -> str:
        note_path = self.notes / f"{agent_name}_{skill_name}.md"
        note_path.write_text(
            f"Executed skill {skill_name} for agent {agent_name}.\n",
            encoding="utf-8",
        )
        result = f"executed local skill {skill_name}; note saved to {note_path.name}"
        self.record("skill", skill_name, agent_name, result)
        return result


def _search_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    corpus = state.read_json("search_corpus.json")
    top = [{"doc": key, "snippet": value} for key, value in sorted(corpus.items())[:3]]
    return f"{tool_name} searched local corpus for {agent_name}: {json.dumps(top)}"


def _filesystem_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    output_path = state.workspace / f"{agent_name}_{tool_name}.txt"
    output_path.write_text(
        f"{agent_name} used {tool_name} in the live local runtime.\n",
        encoding="utf-8",
    )
    files = sorted(path.name for path in state.workspace.iterdir())
    return f"{tool_name} wrote {output_path.name}; workspace files={files}"


def _shell_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    result = subprocess.run(
        ["pwd"],
        capture_output=True,
        text=True,
        check=True,
        cwd=state.workspace,
    )
    return f"{tool_name} executed pwd for {agent_name}: {result.stdout.strip()}"


def _python_runner_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    connection = sqlite3.connect(state.data / "runtime.db")
    try:
        rows = connection.execute(
            "select name, value from metrics order by name limit 3"
        ).fetchall()
    finally:
        connection.close()
    return f"{tool_name} computed local metrics for {agent_name}: {rows}"


def _database_query_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    connection = sqlite3.connect(state.data / "runtime.db")
    try:
        row_count = connection.execute("select count(*) from metrics").fetchone()[0]
    finally:
        connection.close()
    return f"{tool_name} queried sqlite runtime.db for {agent_name}: metric_rows={row_count}"


def _crm_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    customers = state.read_json("customers.json")
    account, details = next(iter(customers.items()))
    return f"{tool_name} loaded customer {account} for {agent_name}: {details}"


def _ticket_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    tickets = state.read_json("tickets.json")
    ticket_id, details = next(iter(tickets.items()))
    details["status"] = "reviewed"
    tickets[ticket_id] = details
    state.write_json("tickets.json", tickets)
    return f"{tool_name} updated ticket {ticket_id} for {agent_name}: {details}"


def _refund_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    customers = state.read_json("customers.json")
    balance = customers["acme-retail"]["balance"]
    refund = round(balance * 0.25, 2)
    return f"{tool_name} calculated deterministic refund for {agent_name}: {refund}"


def _dataset_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    datasets = state.read_json("datasets.json")
    return f"{tool_name} listed datasets for {agent_name}: {sorted(datasets)}"


def _git_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("git_host.json")
    return f"{tool_name} inspected local git metadata for {agent_name}: {payload}"


def _ci_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("ci_dashboard.json")
    return f"{tool_name} returned CI status for {agent_name}: {payload}"


def _deploy_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("deploy_api.json")
    payload["status"] = "scheduled"
    state.write_json("deploy_api.json", payload)
    return f"{tool_name} scheduled deployment for {agent_name}: {payload}"


def _audit_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("audit_api.json")
    payload["entries"].append(f"audit event from {agent_name}")
    state.write_json("audit_api.json", payload)
    return f"{tool_name} appended audit event for {agent_name}: total={len(payload['entries'])}"


def _notes_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    note_path = state.notes / f"{agent_name}_{tool_name}.md"
    note_path.write_text(f"Notes updated by {agent_name} using {tool_name}.\n", encoding="utf-8")
    return f"{tool_name} updated notes file {note_path.name}"


def _siem_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("siem_api.json")
    return f"{tool_name} returned alerts for {agent_name}: {payload['alerts']}"


def _host_isolation_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("hosts.json")
    payload["db-01"]["isolated"] = True
    state.write_json("hosts.json", payload)
    return f"{tool_name} isolated host db-01 for {agent_name}"


def _timeline_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    alerts = state.read_json("siem_api.json")["alerts"]
    timeline = [f"{index + 1}:{alert['id']}:{alert['severity']}" for index, alert in enumerate(alerts)]
    return f"{tool_name} built local incident timeline for {agent_name}: {timeline}"


def _network_backend(state: LocalRuntimeState, tool_name: str, agent_name: str) -> str:
    payload = state.read_json("network.json")
    return f"{tool_name} inspected local network policy for {agent_name}: {payload}"


TOOL_BACKENDS: dict[str, ToolBackend] = {
    "audit_api": _audit_backend,
    "bash": _shell_backend,
    "ci_dashboard": _ci_backend,
    "crm_api": _crm_backend,
    "database_query": _database_query_backend,
    "dataset_registry": _dataset_backend,
    "deploy_api": _deploy_backend,
    "faq_index": _search_backend,
    "filesystem": _filesystem_backend,
    "git_host": _git_backend,
    "host_isolation": _host_isolation_backend,
    "knowledge_base": _search_backend,
    "network": _network_backend,
    "notebook_runner": _python_runner_backend,
    "notes_workspace": _notes_backend,
    "paper_search": _search_backend,
    "python_runner": _python_runner_backend,
    "refund_calculator": _refund_backend,
    "shell": _shell_backend,
    "shell_check": _shell_backend,
    "siem_api": _siem_backend,
    "ticket_api": _ticket_backend,
    "timeline_builder": _timeline_backend,
    "vector_search": _search_backend,
    "web_search": _search_backend,
    "workspace_files": _filesystem_backend,
}


def ensure_tool_backends(tool_names: set[str]) -> None:
    missing = sorted(tool_name for tool_name in tool_names if tool_name not in TOOL_BACKENDS)
    if missing:
        raise ValueError(f"missing local runtime backends for tools: {missing}")

