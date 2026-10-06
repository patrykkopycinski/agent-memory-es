"""Over-12 and malformed-item tolerance in _extract: keep good facts, never silent."""
import pytest

from app import facts


def run_extract(payload):
    def fake_chat(system, user, timeout=120):
        return payload

    orig = facts.llm.chat_json
    facts.llm.chat_json = fake_chat
    try:
        return facts._extract("document text", "2024-05-01")
    finally:
        facts.llm.chat_json = orig


def reset_counters():
    facts.DROPPED["invalid"] = 0
    facts.DROPPED["overflow"] = 0


def test_over_12_keeps_first_12_in_order_with_overflow_counter():
    reset_counters()
    out = run_extract({"facts": [f"Fact number {i} is durable and specific."
                                 for i in range(1, 16)]})
    assert len(out) == facts.MAX_FACTS
    assert out[0] == "[2024-05-01] Fact number 1 is durable and specific."
    assert out[11] == "[2024-05-01] Fact number 12 is durable and specific."
    assert facts.DROPPED["overflow"] == 3


def test_bad_item_among_good_is_skipped_and_counted():
    reset_counters()
    out = run_extract({"facts": ["A perfectly fine durable fact.", 7, "x" * 5,
                                 "Another concrete durable fact."]})
    assert out == ["[2024-05-01] A perfectly fine durable fact.",
                   "[2024-05-01] Another concrete durable fact."]
    assert facts.DROPPED["invalid"] == 2


def test_all_invalid_non_empty_raises():
    reset_counters()
    with pytest.raises(ValueError, match="invalid fact sentence"):
        run_extract({"facts": [7, None, "short"]})


def test_non_list_raises():
    reset_counters()
    with pytest.raises(ValueError, match="invalid facts array"):
        run_extract({"facts": "nope"})


def test_empty_list_is_no_facts():
    reset_counters()
    assert run_extract({"facts": []}) == []
