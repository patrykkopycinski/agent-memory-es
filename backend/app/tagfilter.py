"""Tag filtering for recall: normalisation, label-group extraction, filter compilation, fuzzy resolution.

Decision record: /opt/orca-base/tasks/ames-filtering-decision.md (written before this code).

Everything here is GENERIC. There are no default label groups, no label wording and no domain
vocabulary in this module: callers declare the label groups they want extracted (`labels` on
retain) and the filter they want applied (`filter` on recall). Identity labels (open vocabulary)
and state/currency labels (closed vocabulary) are the same mechanism with different `type`s.

Nothing in this module runs unless the caller passes `tags`, `labels` or `filter`, so the
no-filter / no-label paths stay byte-identical to what they were.
"""
import json
import os
import re
from typing import Optional

TAG_FIELD = "tags"

MAX_TAGS_PER_WRITE = 64
MAX_TAG_CHARS = 128
MAX_LABEL_GROUPS = 16
MAX_GROUP_VALUES = 64
MAX_EXTRACTED_PER_GROUP = 32

# pg_trgm's published default similarity threshold. FIXED before any benchmark run
# (decision record 2.4); an operator may override it by env, but it is never swept.
FUZZY_MIN_SIM = float(os.environ.get("AMES_FUZZY_MIN_SIM", "0.3"))
FUZZY_VOCAB_CAP = int(os.environ.get("AMES_FUZZY_VOCAB_CAP", "5000"))
EXTRACT_MAX_CHARS = int(os.environ.get("AMES_EXTRACT_MAX_CHARS", "8000"))
EXTRACT_RETRIES = 2

LABEL_TYPES = ("value", "multi-value", "text", "multi-text")
FILTER_KEYS = ("all", "any", "none", "narrow_any")
RESOLVE_MODES = ("exact", "fuzzy")


class FilterError(ValueError):
    """The caller's tags / labels / filter are malformed (HTTP 422)."""


class ExtractionError(RuntimeError):
    """Label extraction failed; the write is refused (HTTP 502, nothing written)."""


# ── normalisation ─────────────────────────────────────────────────────────────

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_KEY_SEP = re.compile(r"[^a-z0-9]+")


def norm_key(key: str) -> str:
    k = _KEY_SEP.sub("_", str(key).lower()).strip("_")
    return k


def norm_value(value: str) -> str:
    """Lowercase, words separated by single spaces; `_`, `-` and punctuation are separators."""
    return " ".join(_WORDS.findall(str(value).lower()))


def make_tag(key: str, value: str) -> Optional[str]:
    k, v = norm_key(key), norm_value(value)
    if not k or not v:
        return None
    return "%s:%s" % (k, v)


def norm_tag(tag) -> str:
    """`key:value` -> canonical tag. Raises FilterError for anything unusable."""
    if not isinstance(tag, str) or ":" not in tag:
        raise FilterError("tag must be a string 'key:value', got %r" % (tag,))
    key, _, value = tag.partition(":")
    out = make_tag(key, value)
    if out is None:
        raise FilterError("tag has an empty key or value: %r" % (tag,))
    if len(out) > MAX_TAG_CHARS:
        raise FilterError("tag longer than %d characters: %r" % (MAX_TAG_CHARS, tag[:40]))
    return out


def norm_tags(tags) -> list:
    """Caller-supplied write tags -> canonical, de-duplicated, order-preserving list."""
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)):
        raise FilterError("tags must be a list of 'key:value' strings")
    out = []
    for t in tags:
        c = norm_tag(t)
        if c not in out:
            out.append(c)
    if len(out) > MAX_TAGS_PER_WRITE:
        raise FilterError("at most %d tags per write" % MAX_TAGS_PER_WRITE)
    return out


# ── label groups ──────────────────────────────────────────────────────────────

def validate_labels(labels) -> list:
    """Normalise caller-declared label groups. Raises FilterError on a malformed group."""
    if labels is None:
        return []
    if not isinstance(labels, (list, tuple)):
        raise FilterError("labels must be a list of label groups")
    if len(labels) > MAX_LABEL_GROUPS:
        raise FilterError("at most %d label groups per write" % MAX_LABEL_GROUPS)
    out, seen = [], set()
    for g in labels:
        if not isinstance(g, dict):
            raise FilterError("a label group must be an object")
        key = norm_key(g.get("key", ""))
        if not key:
            raise FilterError("a label group needs a non-empty key")
        if key in seen:
            raise FilterError("duplicate label group key %r" % key)
        seen.add(key)
        typ = g.get("type")
        if typ not in LABEL_TYPES:
            raise FilterError("label %r: type must be one of %s" % (key, LABEL_TYPES))
        values = None
        if typ in ("value", "multi-value"):
            raw = g.get("values")
            if not isinstance(raw, (list, tuple)) or not raw:
                raise FilterError("label %r: type %r needs a non-empty `values` list" % (key, typ))
            if len(raw) > MAX_GROUP_VALUES:
                raise FilterError("label %r: at most %d values" % (key, MAX_GROUP_VALUES))
            values = []
            for item in raw:
                v = item.get("value") if isinstance(item, dict) else item
                nv = norm_value(v if v is not None else "")
                if not nv:
                    raise FilterError("label %r: empty value" % key)
                desc = item.get("description", "") if isinstance(item, dict) else ""
                values.append({"value": nv, "description": str(desc)})
        out.append({"key": key, "type": typ, "description": str(g.get("description", "")),
                    "optional": bool(g.get("optional", True)), "values": values})
    return out


EXTRACT_SYSTEM = (
    "You label a piece of text for a memory store. For each label group below, read the TEXT and "
    "answer exactly as that group's description says. Use only what the TEXT supports; never "
    "invent. Reply with ONE JSON object and nothing else: keys are the group keys, values are "
    "a string (types value/text) or a list of strings (types multi-value/multi-text). For groups "
    "with allowed_values, answer with EXACTLY one of those strings (no descriptions). Use an "
    "empty list or empty string when a group does not apply."
)


def build_extraction_prompt(text: str, groups: list) -> tuple:
    """-> (system, user, truncated). The text is capped at EXTRACT_MAX_CHARS."""
    truncated = len(text) > EXTRACT_MAX_CHARS
    body = text[:EXTRACT_MAX_CHARS]
    lines = []
    for g in groups:
        spec = {"key": g["key"], "type": g["type"], "description": g["description"]}
        if g["values"]:
            # Bare values only: a model asked to pick from "current (active)" echoes that whole
            # string back, which then (correctly) fails the vocabulary check. Descriptions go in
            # a separate field so the answer is exactly one of `allowed_values`.
            spec["allowed_values"] = [v["value"] for v in g["values"]]
            described = {v["value"]: v["description"] for v in g["values"] if v["description"]}
            if described:
                spec["value_descriptions"] = described
        lines.append(json.dumps(spec, ensure_ascii=False))
    user = "LABEL GROUPS (one JSON object per line):\n" + "\n".join(lines) + "\n\nTEXT:\n" + body
    return EXTRACT_SYSTEM, user, truncated


def _as_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [x for x in raw if isinstance(x, (str, int, float)) and not isinstance(x, bool)]
    if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        return [raw]
    return []


def parse_extraction(raw_json, groups: list) -> tuple:
    """Validate the model's answer against the declared groups. NOTHING IS INVENTED:
    out-of-vocabulary values and empty strings are dropped, a required group with no valid
    value yields no tag and is reported as missing. -> (tags, info)."""
    if not isinstance(raw_json, dict):
        raise ExtractionError("extractor did not return a JSON object")
    tags, missing, capped = [], [], False
    for g in groups:
        got = []
        for item in _as_list(raw_json.get(g["key"])):
            nv = norm_value(item)
            if not nv:
                continue
            if g["values"] is not None:
                allowed = {v["value"] for v in g["values"]}
                if nv not in allowed:
                    continue
            if nv not in got:
                got.append(nv)
        if g["type"] in ("value", "text"):
            got = got[:1]
        if len(got) > MAX_EXTRACTED_PER_GROUP:
            got, capped = got[:MAX_EXTRACTED_PER_GROUP], True
        group_tags = [t for t in (make_tag(g["key"], v) for v in got)
                      if t and len(t) <= MAX_TAG_CHARS]
        if not group_tags and not g["optional"]:
            missing.append(g["key"])
        tags.extend(group_tags)
    return tags, {"missing": missing, "capped": capped}


def extract_tags(text: str, groups: list, chat_json=None) -> tuple:
    """One chat-completion call for all groups (temperature 0). Retries EXTRACT_RETRIES times;
    on persistent failure raises ExtractionError so the caller refuses the write (fail closed)."""
    if not groups:
        return [], {"missing": [], "capped": False, "truncated": False}
    if chat_json is None:
        from . import llm as _llm
        chat_json = _llm.chat_json
    system, user, truncated = build_extraction_prompt(text, groups)
    last = None
    for _ in range(EXTRACT_RETRIES + 1):
        try:
            tags, info = parse_extraction(chat_json(system, user), groups)
            info["truncated"] = truncated
            return tags, info
        except Exception as e:                 # transport, non-JSON, wrong shape: all fail closed
            last = e
    raise ExtractionError("label extraction failed after %d attempts: %s"
                          % (EXTRACT_RETRIES + 1, last))


# ── recall filter ─────────────────────────────────────────────────────────────

def _tag_list(value, name) -> list:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise FilterError("filter.%s must be a list" % name)
    return [norm_tag(t) for t in value]


def parse_filter(spec) -> Optional[dict]:
    """Validate and normalise a recall filter. -> None when it holds no usable constraint
    (== no filter), else {all, any, none, narrow_any(or None)}. Raises FilterError (HTTP 422)."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise FilterError("filter must be an object")
    unknown = sorted(set(spec) - set(FILTER_KEYS))
    if unknown:
        raise FilterError("unknown filter key(s): %s (allowed: %s)" % (unknown, list(FILTER_KEYS)))
    all_t = _tag_list(spec.get("all"), "all")
    none_t = _tag_list(spec.get("none"), "none")
    any_g = []
    raw_any = spec.get("any")
    if raw_any is not None:
        if not isinstance(raw_any, (list, tuple)):
            raise FilterError("filter.any must be a list of tag lists")
        for grp in raw_any:
            if not isinstance(grp, (list, tuple)):
                raise FilterError("filter.any must be a list of tag lists")
            g = _tag_list(grp, "any[]")
            if g:
                any_g.append(g)
    narrow = None
    if "narrow_any" in spec and spec["narrow_any"] is not None:
        raw = spec["narrow_any"]
        if not isinstance(raw, (list, tuple)):
            raise FilterError("filter.narrow_any must be a list of leaves")
        narrow = []
        for leaf in raw:
            if not isinstance(leaf, dict):
                raise FilterError("a narrow_any leaf must be an object {tags, resolve}")
            unknown = sorted(set(leaf) - {"tags", "resolve"})
            if unknown:
                raise FilterError("unknown narrow_any leaf key(s): %s" % unknown)
            mode = leaf.get("resolve", "exact")
            if mode not in RESOLVE_MODES:
                raise FilterError("narrow_any.resolve must be one of %s" % (RESOLVE_MODES,))
            tags = _tag_list(leaf.get("tags"), "narrow_any[].tags")
            if tags:
                narrow.append({"tags": tags, "resolve": mode})
    if not (all_t or none_t or any_g or narrow is not None):
        return None
    return {"all": all_t, "any": any_g, "none": none_t, "narrow_any": narrow}


def trigrams(word_or_text: str) -> set:
    """pg_trgm: per word, pad two spaces in front and one behind, take every 3-gram."""
    grams = set()
    for w in _WORDS.findall(str(word_or_text).lower()):
        padded = "  " + w + " "
        for i in range(len(padded) - 2):
            grams.add(padded[i:i + 3])
    return grams


def similarity(a: str, b: str) -> float:
    ga, gb = trigrams(a), trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def resolve_fuzzy(candidate: str, vocabulary, min_sim: Optional[float] = None) -> list:
    """Candidate 'key:value' -> every stored tag of the SAME key whose value's trigram
    similarity is >= min_sim. Deterministic (sorted)."""
    thr = FUZZY_MIN_SIM if min_sim is None else min_sim
    key, _, value = candidate.partition(":")
    out = []
    for stored in vocabulary:
        skey, _, svalue = stored.partition(":")
        if skey == key and similarity(value, svalue) >= thr:
            out.append(stored)
    return sorted(set(out))


def resolve_narrow(narrow: list, vocab_fn) -> tuple:
    """Expand narrow_any leaves to the concrete stored tags they match.
    `vocab_fn(keys) -> iterable of stored tags for those keys` is only called when a fuzzy leaf
    exists. -> (tags (OR set, sorted), resolved_by_candidate)."""
    resolved, tags = {}, set()
    fuzzy_keys = sorted({t.partition(":")[0] for leaf in narrow if leaf["resolve"] == "fuzzy"
                         for t in leaf["tags"]})
    vocab = list(vocab_fn(fuzzy_keys)) if fuzzy_keys else []
    for leaf in narrow:
        for cand in leaf["tags"]:
            if leaf["resolve"] == "exact":
                tags.add(cand)
                resolved.setdefault(cand, [cand])
            else:
                hits = resolve_fuzzy(cand, vocab)
                if hits:
                    tags.update(hits)
                    resolved[cand] = hits
    return sorted(tags), resolved


def es_clauses(parsed: dict, narrow_tags) -> tuple:
    """Compile a parsed filter into ES bool-filter clauses.
    -> (filter_clauses, must_not_clauses). `narrow_tags` is the resolved OR set (or None when the
    filter has no narrow_any). An empty narrow set is the caller's cue to abstain BEFORE calling
    ES; this function never builds a clause that matches everything in that case."""
    flt, must_not = [], []
    for t in parsed["all"]:
        flt.append({"term": {TAG_FIELD: t}})
    for grp in parsed["any"]:
        flt.append({"terms": {TAG_FIELD: list(grp)}})
    if parsed["narrow_any"] is not None:
        flt.append({"terms": {TAG_FIELD: list(narrow_tags or [])}})
    if parsed["none"]:
        must_not.append({"terms": {TAG_FIELD: list(parsed["none"])}})
    return flt, must_not
