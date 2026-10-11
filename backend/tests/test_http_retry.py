"""Transient-HTTP retry helper, job-level transient requeue, and AMES_LLM_TEMPERATURE."""
import copy
import http.client
import json
import re
import socket
import urllib.error
from unittest.mock import patch

import pytest

from app import embeddings, facts, http_retry, llm


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(http_retry.time, "sleep", lambda _: None)


def http_error(code, headers=None):
    return urllib.error.HTTPError("http://x/y", code, "err",
                                  headers or {}, None)


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _ok():
    return Resp({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})


# --- helper -----------------------------------------------------------

def test_503_then_503_then_200_succeeds_after_two_retries():
    sleeps = []
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise http_error(503)
        return _ok()

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        with patch.object(http_retry, "budget_seconds", lambda: 300):
            out = http_retry.urlopen(None, sleep=sleeps.append, clock=lambda: 0)
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert all(0 <= s <= 60 for s in sleeps)


def test_429_with_retry_after_7_sleeps_7():
    sleeps = []

    def fake_urlopen(req, timeout=None):
        if not sleeps:
            raise http_error(429, {"Retry-After": "7"})
        return _ok()

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        with patch.object(http_retry, "budget_seconds", lambda: 300):
            http_retry.urlopen(None, sleep=sleeps.append, clock=lambda: 0)
    assert sleeps == [7.0]


def test_retry_after_http_date_honoured_and_capped():
    import email.utils
    when = email.utils.formatdate(None, usegmt=True)  # now
    sleeps = []

    def fake_urlopen(req, timeout=None):
        if not sleeps:
            raise http_error(503, {"Retry-After": when})
        return _ok()

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        with patch.object(http_retry, "budget_seconds", lambda: 300):
            http_retry.urlopen(None, sleep=sleeps.append, clock=lambda: 0)
    assert len(sleeps) == 1 and 0 <= sleeps[0] < 5

    # huge Retry-After caps at 120s
    sleeps.clear()

    def fake2(req, timeout=None):
        if not sleeps:
            raise http_error(429, {"Retry-After": "9999"})
        return _ok()

    with patch.object(http_retry.urllib.request, "urlopen", fake2):
        with patch.object(http_retry, "budget_seconds", lambda: 300):
            http_retry.urlopen(None, sleep=sleeps.append, clock=lambda: 0)
    assert sleeps == [120.0]


def test_budget_exhausted_raises_original_error():
    err = http_error(503)
    t = {"now": 0}

    def fake_urlopen(req, timeout=None):
        raise err

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(urllib.error.HTTPError) as ei:
            http_retry.urlopen(None, sleep=lambda s: None,
                               clock=lambda: t.__setitem__("now", t["now"] + 10) or t["now"],
                               budget=10)
    assert ei.value is err
    # no sleeps once the deadline is passed
    # (first attempt + sleeps fit inside budget; budget reached -> raise)


def test_non_transient_4xx_no_retry():
    for code in (400, 401, 404):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None, _code=code):
            calls["n"] += 1
            raise http_error(_code)

        with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
            with pytest.raises(urllib.error.HTTPError):
                http_retry.urlopen(None, sleep=lambda s: pytest.fail("slept"),
                                   clock=lambda: 0, budget=300)
        assert calls["n"] == 1


def test_value_error_no_retry():
    def fake_urlopen(req, timeout=None):
        raise ValueError("bad json")

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        with pytest.raises(ValueError):
            http_retry.urlopen(None, sleep=lambda s: pytest.fail("slept"),
                               clock=lambda: 0, budget=300)


def test_connection_refused_retried():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionRefusedError()
        return _ok()

    with patch.object(http_retry.urllib.request, "urlopen", fake_urlopen):
        out = http_retry.urlopen(None, sleep=lambda s: None, clock=lambda: 0,
                                 budget=300)
    assert calls["n"] == 2


def test_remote_disconnected_is_transient():
    assert http_retry.is_transient(http.client.RemoteDisconnected())
    assert http_retry.is_transient(socket.timeout())
    assert http_retry.is_transient(urllib.error.URLError("refused"))
    assert not http_retry.is_transient(http_error(400))
    assert not http_retry.is_transient(ValueError("x"))


# --- chat_json / embeddings._post use the helper ------------------------

def test_chat_json_uses_retry_helper():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http_error(503)
        return _ok()

    with patch.object(llm.http_retry.urllib.request, "urlopen", fake_urlopen):
        llm.chat_json("s", "u")
    assert calls["n"] == 2


def test_embeddings_post_uses_retry_helper():
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http_error(429, {"Retry-After": "1"})
        return Resp({"data": [{"embedding": [0.0]}]})

    with patch.object(embeddings.http_retry.urllib.request, "urlopen", fake_urlopen):
        out = embeddings._post("http://x", {"input": ["a"]})
    assert calls["n"] == 2 and out["data"][0]["embedding"] == [0.0]


# --- run_once job-level behaviour ---------------------------------------

class Docs:
    def __init__(self):
        self.data = {}
        self.updates = []

    def __call__(self, method, path, body=None):
        index, key = path.split('/')[1], path.split('/')[-1].split('?')[0]
        if method == 'GET':
            if (index, key) not in self.data:
                raise RuntimeError('-> 404: missing')
            doc, seq = self.data[index, key]
            return {'_source': copy.deepcopy(doc), '_seq_no': seq, '_primary_term': 1}
        assert method == 'POST' and '/_update/' in path
        doc, seq = self.data[index, key]
        assert int(re.search(r'if_seq_no=(\d+)', path).group(1)) == seq
        self.updates.append((index, key, copy.deepcopy(body['doc'])))
        doc.update(body['doc'])
        self.data[index, key] = doc, seq + 1
        return {}


def _run_once_with(exc):
    db = Docs()
    db.data[facts.QUEUE, 'j'] = ({'owner_id': 'alice', 'source_id': 's1',
                                   'state': 'pending', 'attempts': 2,
                                   'transient_retries': 0,
                                   'lease_token': 'tok'}, 3)
    with patch.object(facts, 'es', db):
        with patch.object(facts, 'claim', lambda *a: ('j', db.data[facts.QUEUE, 'j'][0], 'tok')):
            with patch.object(facts, 'process', side_effect=exc):
                facts.run_once()
    return db


def test_run_once_transient_requeues_without_consuming_attempt():
    db = _run_once_with(http_error(503))
    doc = db.data[facts.QUEUE, 'j'][0]
    assert doc['state'] == 'pending'
    assert doc['attempts'] == 2  # unchanged
    assert doc['transient_retries'] == 1
    # next_at = now + min(900, 60 * 2**1) = +120s
    assert 'finished_at' not in doc
    scheduled = facts.dt.datetime.fromisoformat(doc['next_at'].replace('Z', '+00:00'))
    assert 115 < (scheduled - facts._now()).total_seconds() <= 120


def test_run_once_non_transient_consumes_attempt():
    db = _run_once_with(ValueError("bad json"))
    doc = db.data[facts.QUEUE, 'j'][0]
    assert doc['attempts'] == 3
    assert doc['state'] == 'pending'
    assert doc.get('transient_retries', 0) == 0


def test_run_once_transient_cap_reached_fails():
    db = Docs()
    db.data[facts.QUEUE, 'j'] = ({'owner_id': 'alice', 'source_id': 's1',
                                   'state': 'pending', 'attempts': 0,
                                   'transient_retries': http_retry.MAX_TRANSIENT_RETRIES,
                                   'lease_token': 'tok'}, 3)
    with patch.object(facts, 'es', db):
        with patch.object(facts, 'claim', lambda *a: ('j', db.data[facts.QUEUE, 'j'][0], 'tok')):
            with patch.object(facts, 'process', side_effect=http_error(429)):
                facts.run_once()
    doc = db.data[facts.QUEUE, 'j'][0]
    assert doc['state'] == 'failed'
    assert doc['attempts'] == 0  # still never consumed an attempt
    assert doc['transient_retries'] == http_retry.MAX_TRANSIENT_RETRIES + 1


def test_ensure_queue_puts_additive_mapping():
    puts = []

    def fake_es(method, path, body=None):
        if method == "PUT" and path.endswith("/_mapping"):
            puts.append((path, body))
            return {"acknowledged": True}
        if method == "GET":
            return {"ok": True}  # indices exist -> no create attempted
        raise AssertionError(f"unexpected {method} {path}")

    with patch.object(facts, 'es', fake_es):
        facts.ensure_queue()
    assert puts == [(f"/{facts.QUEUE}/_mapping",
                     {"properties": {"transient_retries": {"type": "integer"}}})]


# --- ADDENDUM: AMES_LLM_TEMPERATURE --------------------------------------

def _capture_payload(fn):
    reqs = []

    def fake_urlopen(req, timeout=None):
        reqs.append(json.loads(req.data.decode()))
        return _ok()

    with patch.object(llm.http_retry.urllib.request, "urlopen", fake_urlopen):
        fn()
    return reqs[0]


def _all_three_payloads():
    p1 = _capture_payload(lambda: llm.chat("q", [{"id": "am_1", "text": "t"}]))
    p2 = _capture_payload(lambda: llm.rewrite_queries("q"))
    p3 = _capture_payload(lambda: llm.chat_json("s", "u"))
    return p1, p2, p3


def test_temperature_default_sends_current_values(monkeypatch):
    monkeypatch.delenv("AMES_LLM_TEMPERATURE", raising=False)
    c, r, j = _all_three_payloads()
    assert c["temperature"] == 0
    assert r["temperature"] == 0.3
    assert j["temperature"] == 0



def test_temperature_omit_sends_no_key(monkeypatch):
    monkeypatch.setenv("AMES_LLM_TEMPERATURE", "omit")
    for payload in _all_three_payloads():
        assert "temperature" not in payload
    monkeypatch.setenv("AMES_LLM_TEMPERATURE", "")
    for payload in _all_three_payloads():
        assert "temperature" not in payload


def test_temperature_explicit_value_used(monkeypatch):
    monkeypatch.setenv("AMES_LLM_TEMPERATURE", "0.7")
    for payload in _all_three_payloads():
        assert payload["temperature"] == 0.7


def test_chat_json_user_message_keeps_json_word():
    reqs = []

    def fake_urlopen(req, timeout=None):
        reqs.append(json.loads(req.data.decode()))
        return _ok()

    with patch.object(llm.http_retry.urllib.request, "urlopen", fake_urlopen):
        llm.chat_json("system text", "user text")
    user = reqs[0]["messages"][-1]["content"]
    assert "json" in user.lower()


def test_temperature_unsupported_400_not_retried():
    body = ("{'error': {'message': \"Unsupported parameter: 'temperature' is not "
            "supported with this model.\"}}").encode()

    def fake_urlopen(req, timeout=None):
        exc = http_error(400)
        exc.read = lambda: body
        raise exc

    calls = {"n": 0}

    def counting(req, timeout=None):
        calls["n"] += 1
        return fake_urlopen(req, timeout)

    with patch.object(llm.http_retry.urllib.request, "urlopen", counting):
        with pytest.raises(urllib.error.HTTPError):
            llm.chat_json("s", "u", )
    assert calls["n"] == 1  # non-transient: no retry
