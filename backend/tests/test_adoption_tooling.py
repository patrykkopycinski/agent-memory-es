"""Tests for adoption tooling: importers, MCP server wire protocol, doctor."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(REPO, "scripts")
import _safety  # noqa: F401  (script-path guard: conftest bypass)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def importer():
    return _load("import_memory")


GOV = os.path.expanduser(
    "~/Downloads/Users-dkirchan-workspace-vellum")
GOV2 = os.path.expanduser(
    "~/Downloads/Users-dkirchan-workspace-kibana-agent-mcp")


class TestGovernanceParser:
    def test_parses_real_store(self, importer):
        if not os.path.isdir(GOV):
            pytest.skip("dkirchan vellum store not present")
        entries = list(importer.parse_governance_store(GOV))
        assert entries, "no entries parsed"
        kinds = {e["kind"] for e in entries}
        assert "semantic" in kinds          # decisions
        assert "procedural" in kinds        # profile
        assert all(e["visibility"] in ("private", "team") for e in entries)

    def test_decisions_split_per_entry(self, importer):
        if not os.path.isdir(GOV):
            pytest.skip("store not present")
        entries = [e for e in importer.parse_governance_store(GOV)
                   if "decisions-log" in e["source"]]
        assert len(entries) >= 3            # real log has many decisions
        assert all(e["kind"] == "semantic" and e["visibility"] == "team"
                   for e in entries)
        # each entry carries its own date
        assert all(e["occurred_at"] and e["occurred_at"].endswith("T00:00:00Z")
                   for e in entries)

    def test_second_store_real(self, importer):
        if not os.path.isdir(GOV2):
            pytest.skip("second store not present")
        entries = list(importer.parse_governance_store(GOV2))
        assert entries
        assert any("kibana-agent-mcp" in e["source"] for e in entries)

    def test_handoff_and_changelog_skipped_by_default(self, importer, tmp_path):
        store = tmp_path / "demo-store"
        store.mkdir()
        (store / "profile.md").write_text("# Profile\n\noverview text\n")
        (store / "session-handoff.md").write_text("# Handoff\n\ntransient state\n")
        (store / "changelog.md").write_text(
            "# Changelog\n\n## 2026-01-01\n- did a thing\n")
        entries = list(importer.parse_governance_store(str(store)))
        assert len(entries) == 1
        assert entries[0]["kind"] == "procedural"

    def test_changelog_opt_in(self, importer, tmp_path):
        store = tmp_path / "s2"
        store.mkdir()
        (store / "changelog.md").write_text("# Changelog\n\n## 2026-01-01\n- did a thing\n")
        importer._FLAGS["include_changelog"] = True
        try:
            entries = list(importer.parse_governance_store(str(store)))
        finally:
            importer._FLAGS["include_changelog"] = False
        assert len(entries) == 1
        assert entries[0]["occurred_at"] == "2026-01-01T00:00:00Z"

    def test_detect_governance(self, importer, tmp_path):
        store = tmp_path / "g"
        store.mkdir()
        (store / "profile.md").write_text("x")
        assert importer._detect(str(store)) == "governance"

    def test_detect_markdown(self, importer):
        assert importer._detect("AGENTS.md") == "markdown"


class TestHindsightParser:
    def test_jsonl(self, importer, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text(json.dumps({"text": "fact one", "kind": "fact",
                                 "created_at": "2026-01-01T00:00:00Z"}) + "\n" +
                     json.dumps({"content": "note two", "type": "note"}) + "\n")
        entries = list(importer.parse_hindsight_export(str(p)))
        assert len(entries) == 2
        assert entries[0]["kind"] == "semantic"
        assert entries[1]["kind"] == "episodic"
        assert entries[0]["occurred_at"] == "2026-01-01T00:00:00Z"

    def test_json_array(self, importer, tmp_path):
        p = tmp_path / "h.json"
        p.write_text(json.dumps([{"text": "t", "procedure": True}]))
        entries = list(importer.parse_hindsight_export(str(p)))
        assert len(entries) == 1

    def test_empty_skipped(self, importer, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text(json.dumps({"text": "  "}) + "\n")
        assert list(importer.parse_hindsight_export(str(p))) == []


class TestMarkdownParser:
    def test_sections(self, importer, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("preamble text\n\n## One\n\nbody one\n\n## Two\n\nbody two\n")
        entries = list(importer.parse_markdown(str(p)))
        assert len(entries) == 3               # preamble + 2 sections
        assert all(e["kind"] == "procedural" for e in entries)

    def test_no_sections_single_entry(self, importer, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("just text\n")
        assert len(list(importer.parse_markdown(str(p)))) == 1


class TestMcpServer:
    def test_tool_schemas_openai_parameters(self):
        """Hermes sanitizer contract: tools must use inputSchema with properties."""
        m = _load("ames_mcp_server")
        names = {t["name"] for t in m.TOOLS}
        assert names == {"ames_recall", "ames_retain", "ames_reflect"}
        for t in m.TOOLS:
            schema = t["inputSchema"]
            assert schema.get("type") == "object"
            assert schema.get("properties"), f"{t['name']} has no properties"

    def test_stdio_rpc_handshake(self):
        """Full JSON-RPC handshake over stdio without a live backend."""
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "ames_mcp_server.py"),
             "--api-key", "test"],
            input="\n".join(json.dumps(m) for m in msgs) + "\n",
            capture_output=True, text=True, timeout=30)
        lines = [json.loads(l) for l in proc.stdout.strip().splitlines()]
        assert lines[0]["result"]["serverInfo"]["name"] == "agent-memory-es"
        tools = lines[1]["result"]["tools"]
        assert len(tools) == 3

    def test_missing_key_exits(self):
        env = {k: v for k, v in os.environ.items() if k != "AMES_SERVICE_KEY"}
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "ames_mcp_server.py")],
            capture_output=True, text=True, timeout=30, env=env)
        assert proc.returncode == 2


class TestDoctor:
    def test_runs_and_reports(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "ames_doctor.py")],
            capture_output=True, text=True, timeout=60)
        assert "ames doctor" in proc.stdout
        assert proc.returncode in (0, 1)   # both are valid outcomes
