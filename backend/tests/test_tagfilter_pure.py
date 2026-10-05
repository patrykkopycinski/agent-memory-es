"""Pure unit tests for tag filtering (no Elasticsearch, no LLM, no network).

Fixture vocabulary is invented and neutral on purpose (fruit, colours, planets): the product code
carries no benchmark vocabulary, and neither do its tests.

    docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend ames-backend \
        python -m pytest tests/test_tagfilter_pure.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import tagfilter as tf  # noqa: E402


# ── normalisation ─────────────────────────────────────────────────────────────

def test_value_normalisation_makes_separators_equivalent():
    assert tf.norm_value("Error_Handling") == "error handling"
    assert tf.norm_value("error-handling") == "error handling"
    assert tf.norm_value("  Error   Handling!! ") == "error handling"
    assert tf.norm_value("CI/CD") == "ci cd"


def test_key_normalisation_collapses_to_underscores():
    assert tf.norm_key("Fruit Kind") == "fruit_kind"
    assert tf.norm_key("--a..b--") == "a_b"


def test_make_tag_rejects_empty_halves():
    assert tf.make_tag("name", "Pear") == "name:pear"
    assert tf.make_tag("", "x") is None
    assert tf.make_tag("name", "!!!") is None


@pytest.mark.parametrize("bad", ["nocolon", "", ":v", "k:", 5, None, ["k:v"]])
def test_norm_tag_rejects_malformed(bad):
    with pytest.raises(tf.FilterError):
        tf.norm_tag(bad)


def test_norm_tag_keeps_only_the_first_colon_as_separator():
    assert tf.norm_tag("note:time 10:30") == "note:time 10 30"


def test_norm_tag_rejects_overlong():
    with pytest.raises(tf.FilterError):
        tf.norm_tag("k:" + "x" * 200)


def test_norm_tags_dedups_in_order_and_caps():
    assert tf.norm_tags(["a:b", "A:B", "c:d"]) == ["a:b", "c:d"]
    assert tf.norm_tags(None) == []
    with pytest.raises(tf.FilterError):
        tf.norm_tags(["k:v%d" % i for i in range(tf.MAX_TAGS_PER_WRITE + 1)])
    with pytest.raises(tf.FilterError):
        tf.norm_tags("k:v")


# ── label groups ──────────────────────────────────────────────────────────────

def _g(**kw):
    base = {"key": "colour", "type": "value", "description": "the colour", "optional": False,
            "values": [{"value": "red", "description": "r"}, {"value": "Blue"}]}
    base.update(kw)
    return base


def test_validate_labels_normalises_and_defaults():
    out = tf.validate_labels([_g()])
    assert out[0]["key"] == "colour" and out[0]["optional"] is False
    assert [v["value"] for v in out[0]["values"]] == ["red", "blue"]
    assert tf.validate_labels(None) == [] and tf.validate_labels([]) == []
    assert tf.validate_labels([{"key": "n", "type": "multi-text"}])[0]["optional"] is True


@pytest.mark.parametrize("bad", [
    [{"type": "value", "values": ["a"]}],                          # no key
    [{"key": "k", "type": "bogus"}],                                # bad type
    [{"key": "k", "type": "value"}],                                # value needs values
    [{"key": "k", "type": "multi-value", "values": []}],            # empty values
    [_g(), _g()],                                                   # duplicate key
    ["not a dict"],
])
def test_validate_labels_rejects(bad):
    with pytest.raises(tf.FilterError):
        tf.validate_labels(bad)


def test_validate_labels_caps_groups():
    with pytest.raises(tf.FilterError):
        tf.validate_labels([{"key": "k%d" % i, "type": "text"} for i in range(tf.MAX_LABEL_GROUPS + 1)])


# ── extraction parsing: nothing is invented ──────────────────────────────────

def test_parse_extraction_drops_out_of_vocabulary_and_empty_values():
    groups = tf.validate_labels([_g(optional=True)])
    tags, info = tf.parse_extraction({"colour": "GREEN"}, groups)
    assert tags == [] and info["missing"] == []
    tags, _ = tf.parse_extraction({"colour": "Blue"}, groups)
    assert tags == ["colour:blue"]
    tags, _ = tf.parse_extraction({"colour": ""}, groups)
    assert tags == []


def test_required_group_without_a_valid_value_is_reported_missing_not_defaulted():
    groups = tf.validate_labels([_g(optional=False)])
    tags, info = tf.parse_extraction({"colour": "mauve"}, groups)
    assert tags == [] and info["missing"] == ["colour"]


def test_value_type_keeps_one_and_multi_keeps_all():
    one = tf.validate_labels([_g(type="value", optional=True)])
    many = tf.validate_labels([_g(type="multi-value", optional=True)])
    assert tf.parse_extraction({"colour": ["red", "blue"]}, one)[0] == ["colour:red"]
    assert tf.parse_extraction({"colour": ["red", "blue"]}, many)[0] == ["colour:red", "colour:blue"]


def test_open_vocabulary_text_groups_take_free_values_and_dedup():
    g = tf.validate_labels([{"key": "name", "type": "multi-text"}])
    tags, _ = tf.parse_extraction({"name": ["Pear", "pear", "Pyrus", "", 7, None, True]}, g)
    assert tags == ["name:pear", "name:pyrus", "name:7"]
    g1 = tf.validate_labels([{"key": "name", "type": "text"}])
    assert tf.parse_extraction({"name": ["Pear", "Pyrus"]}, g1)[0] == ["name:pear"]


def test_extraction_value_cap_is_reported():
    g = tf.validate_labels([{"key": "n", "type": "multi-text"}])
    tags, info = tf.parse_extraction({"n": ["v%d" % i for i in range(tf.MAX_EXTRACTED_PER_GROUP + 5)]}, g)
    assert len(tags) == tf.MAX_EXTRACTED_PER_GROUP and info["capped"] is True


def test_non_object_reply_is_an_extraction_error():
    g = tf.validate_labels([{"key": "n", "type": "text"}])
    for bad in (None, [], "x", 3):
        with pytest.raises(tf.ExtractionError):
            tf.parse_extraction(bad, g)


def test_extract_tags_no_groups_makes_no_llm_call():
    calls = []
    out = tf.extract_tags("text", [], chat_json=lambda *a: calls.append(a) or {})
    assert out[0] == [] and calls == []


def test_extract_tags_retries_then_succeeds():
    g = tf.validate_labels([{"key": "n", "type": "text"}])
    seq = iter([RuntimeError("boom"), ValueError("not json"), {"n": "Pear"}])

    def fake(system, user):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v
    tags, info = tf.extract_tags("a pear", g, chat_json=fake)
    assert tags == ["n:pear"] and info["truncated"] is False


def test_extract_tags_fails_closed_after_the_retry_budget():
    g = tf.validate_labels([{"key": "n", "type": "text"}])
    n = []

    def boom(system, user):
        n.append(1)
        raise RuntimeError("down")
    with pytest.raises(tf.ExtractionError):
        tf.extract_tags("t", g, chat_json=boom)
    assert len(n) == tf.EXTRACT_RETRIES + 1


def test_extraction_prompt_carries_groups_and_caps_the_text(monkeypatch):
    monkeypatch.setattr(tf, "EXTRACT_MAX_CHARS", 10)
    g = tf.validate_labels([_g()])
    system, user, truncated = tf.build_extraction_prompt("x" * 50, g)
    assert truncated is True and "x" * 11 not in user and "x" * 10 in user
    assert '"key": "colour"' in user and "blue" in user
    assert "never invent" in system


def test_allowed_values_are_bare_so_the_model_cannot_echo_a_description():
    import json as _json
    g = tf.validate_labels([_g()])
    _, user, _ = tf.build_extraction_prompt("t", g)
    spec = _json.loads(user.split("\n")[1])
    assert spec["allowed_values"] == ["red", "blue"]          # exactly the vocabulary, nothing appended
    assert spec["value_descriptions"] == {"red": "r"}         # descriptions travel separately
    assert "(" not in "".join(spec["allowed_values"])


# ── filter parsing ────────────────────────────────────────────────────────────

def test_no_filter_forms_parse_to_none():
    for spec in (None, {}, {"all": []}, {"any": [[]]}, {"none": []}, {"narrow_any": None}):
        assert tf.parse_filter(spec) is None, spec


def test_parse_filter_normalises_every_clause():
    p = tf.parse_filter({"all": ["Colour:Red"], "any": [["a:B", "a:c"], []], "none": ["state:Old_Version"],
                         "narrow_any": [{"tags": ["name:Pear Tree"], "resolve": "fuzzy"}]})
    assert p["all"] == ["colour:red"] and p["any"] == [["a:b", "a:c"]]
    assert p["none"] == ["state:old version"]
    assert p["narrow_any"] == [{"tags": ["name:pear tree"], "resolve": "fuzzy"}]


@pytest.mark.parametrize("bad", [
    "str", [], {"bogus": []}, {"all": "k:v"}, {"all": ["nocolon"]}, {"any": ["k:v"]},
    {"any": "x"}, {"none": 3}, {"narrow_any": "x"}, {"narrow_any": ["x"]},
    {"narrow_any": [{"tags": ["k:v"], "resolve": "soft"}]},
    {"narrow_any": [{"tags": ["k:v"], "extra": 1}]},
])
def test_parse_filter_rejects_malformed(bad):
    with pytest.raises(tf.FilterError):
        tf.parse_filter(bad)


def test_present_but_unusable_narrow_any_is_kept_as_empty_not_dropped():
    # decision record A4: the only "empty means match nothing" case
    p = tf.parse_filter({"narrow_any": [{"tags": []}]})
    assert p is not None and p["narrow_any"] == []
    p = tf.parse_filter({"narrow_any": []})
    assert p is not None and p["narrow_any"] == []


def test_exact_leaf_defaults_when_resolve_omitted():
    p = tf.parse_filter({"narrow_any": [{"tags": ["n:x"]}]})
    assert p["narrow_any"][0]["resolve"] == "exact"


# ── trigram similarity / fuzzy resolution ────────────────────────────────────

def test_trigrams_match_the_pg_trgm_definition():
    assert tf.trigrams("cat") == {"  c", " ca", "cat", "at "}
    assert tf.trigrams("") == set()
    assert tf.trigrams("a b") == {"  a", " a ", "  b", " b "}


def test_similarity_known_values():
    assert tf.similarity("word", "word") == 1.0
    assert tf.similarity("word", "") == 0.0
    # pg_trgm documents similarity('word','two words') = 0.363636
    assert abs(tf.similarity("word", "two words") - 0.363636) < 1e-5


def test_fuzzy_threshold_is_the_pg_trgm_default_and_fixed():
    assert tf.FUZZY_MIN_SIM == 0.3


def test_fuzzy_resolution_catches_a_misspelling_and_respects_the_key():
    vocab = ["name:strawberry", "name:blueberry", "other:strawbery", "name:plum"]
    assert tf.resolve_fuzzy("name:strawbery", vocab) == ["name:strawberry"]
    assert tf.resolve_fuzzy("name:zzzz", vocab) == []
    assert tf.resolve_fuzzy("name:plum", vocab) == ["name:plum"]


def test_fuzzy_threshold_boundary_is_inclusive():
    vocab = ["k:abc"]
    s = tf.similarity("abd", "abc")
    assert tf.resolve_fuzzy("k:abd", vocab, min_sim=s) == ["k:abc"]
    assert tf.resolve_fuzzy("k:abd", vocab, min_sim=s + 0.001) == []


def test_resolve_narrow_exact_never_asks_for_the_vocabulary():
    asked = []
    tags, res = tf.resolve_narrow([{"tags": ["n:x", "n:y"], "resolve": "exact"}],
                                  lambda keys: asked.append(keys) or [])
    assert tags == ["n:x", "n:y"] and asked == [] and res == {"n:x": ["n:x"], "n:y": ["n:y"]}


def test_resolve_narrow_fuzzy_asks_once_for_the_keys_involved_and_ors_the_leaves():
    asked = []

    def vocab(keys):
        asked.append(keys)
        return ["n:strawberry", "n:plum"]
    tags, res = tf.resolve_narrow([{"tags": ["n:strawbery"], "resolve": "fuzzy"},
                                   {"tags": ["n:plum"], "resolve": "exact"},
                                   {"tags": ["n:nothing"], "resolve": "fuzzy"}], vocab)
    assert asked == [["n"]]
    assert tags == ["n:plum", "n:strawberry"]
    assert "n:nothing" not in res


def test_resolve_narrow_fuzzy_with_no_match_yields_empty_set():
    tags, res = tf.resolve_narrow([{"tags": ["n:qqqq"], "resolve": "fuzzy"}], lambda k: ["n:apple"])
    assert tags == [] and res == {}


# ── ES clause compilation ─────────────────────────────────────────────────────

def test_es_clauses_for_every_clause_type():
    parsed = tf.parse_filter({"all": ["a:1", "b:2"], "any": [["c:3", "c:4"]], "none": ["d:5"],
                              "narrow_any": [{"tags": ["n:x"]}]})
    flt, must_not = tf.es_clauses(parsed, ["n:x", "n:y"])
    assert {"term": {"tags": "a:1"}} in flt and {"term": {"tags": "b:2"}} in flt
    assert {"terms": {"tags": ["c:3", "c:4"]}} in flt
    assert {"terms": {"tags": ["n:x", "n:y"]}} in flt
    assert must_not == [{"terms": {"tags": ["d:5"]}}]


def test_es_clauses_omit_clauses_that_are_absent():
    flt, must_not = tf.es_clauses(tf.parse_filter({"all": ["a:1"]}), None)
    assert flt == [{"term": {"tags": "a:1"}}] and must_not == []
    flt, must_not = tf.es_clauses(tf.parse_filter({"none": ["a:1"]}), None)
    assert flt == [] and must_not == [{"terms": {"tags": ["a:1"]}}]


def test_empty_narrow_set_never_compiles_to_match_everything():
    flt, _ = tf.es_clauses(tf.parse_filter({"narrow_any": []}), [])
    assert flt == [{"terms": {"tags": []}}]       # an empty terms clause matches nothing in ES
