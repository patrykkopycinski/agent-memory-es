"""Prompt-contract tests for the v2 extraction prompt (commit: single-sentence, <=300 chars, <=12 facts)."""
from unittest.mock import patch
import pytest
from app import facts


def test_extract_prompt_states_caps():
    """The prompt must tell the model every cap the validator enforces, so schema
    failures are model drift, not a contract the model was never told about."""
    with patch.object(facts.llm, "chat_json", return_value={"facts": []}) as cj:
        facts._extract("doc", "2020-01-02")
        system = cj.call_args[0][0]
    assert "at most 12" in system
    assert "at most 300" in system
    assert "one single complete sentence" in system
    assert facts.PROMPT_VERSION == "document-v2"


def test_prompt_version_bumped_with_prompt_change():
    """Cache keys include PROMPT_VERSION; a prompt change without a bump would
    silently reuse v1 completions cached under the old contract."""
    assert facts.PROMPT_VERSION != "document-v1"
