"""Manual recovery is scoped, conditional, and does not replay committed history."""
import copy
import io
import json
import re
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app import facts, llm, main


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
        assert 'if_primary_term=1' in path
        self.updates.append((index, key, copy.deepcopy(body['doc'])))
        doc.update(body['doc'])
        self.data[index, key] = doc, seq + 1
        return {}


def test_retry_job_scoped_bounded_idempotent_and_lease_aware():
    db = Docs()
    db.data[facts.QUEUE, 'j'] = ({'owner_id': 'alice', 'state': 'failed',
                                   'attempts': 4, 'lease_token': 'old', 'error': 'bad'}, 3)
    with patch.object(facts, 'es', db):
        with pytest.raises(ValueError):
            facts.retry_failed_job('bob', 'j')
        assert facts.retry_failed_job('alice', 'j')['state'] == 'pending'
        assert db.data[facts.QUEUE, 'j'][0]['attempts'] == 0
        assert db.data[facts.QUEUE, 'j'][0]['lease_token'] is None
        assert facts.retry_failed_job('alice', 'j')['state'] == 'pending'
        assert len(db.updates) == 1
        db.data[facts.QUEUE, 'j'][0]['state'] = 'running'
        with pytest.raises(ValueError):
            facts.retry_failed_job('alice', 'j')
        db.data[facts.QUEUE, 'j'][0]['state'] = 'completed'
        with pytest.raises(ValueError):
            facts.retry_failed_job('alice', 'j')


def test_retry_backfill_preserves_cursor_and_no_restart_of_active():
    db = Docs()
    key = facts._backfill_id('alice', 'private')
    db.data[facts.BACKFILLS, key] = ({'owner_id': 'alice', 'visibility': 'private',
        'state': 'failed', 'cursor': '["2020-01-01", "doc"]', 'queued': 10,
        'lease_token': 'old', 'error': 'failed'}, 2)
    with patch.object(facts, 'es', db), patch.object(facts, 'backfill_status', return_value={'state': 'pending'}):
        with pytest.raises(RuntimeError, match='404'):
            facts.retry_failed_backfill('bob', 'private')
        assert facts.retry_failed_backfill('alice', 'private')['state'] == 'pending'
        assert 'cursor' not in db.updates[0][2] and 'queued' not in db.updates[0][2]
        assert db.data[facts.BACKFILLS, key][0]['cursor'] == '["2020-01-01", "doc"]'
        facts.retry_failed_backfill('alice', 'private')
        assert len(db.updates) == 1
        db.data[facts.BACKFILLS, key][0]['state'] = 'running'
        with pytest.raises(ValueError):
            facts.retry_failed_backfill('alice', 'private')


def test_endpoint_auth_and_owner_forwarding():
    with pytest.raises(HTTPException) as exc:
        main.caller(None)
    assert exc.value.status_code == 401
    with patch.object(main.facts, 'retry_failed_job', return_value={'state': 'pending'}) as retry:
        assert main.fact_job_retry('j', {'owner_id': 'alice'}) == {'state': 'pending'}
        retry.assert_called_once_with('alice', 'j')
    with patch.object(main.facts, 'retry_failed_backfill', return_value={'state': 'pending'}) as retry:
        main.fact_backfill_retry('private', {'owner_id': 'alice'})
        retry.assert_called_once_with('alice', 'private')


def test_malformed_json_metadata_never_contains_content_or_parser_doc():
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b''
    payload = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"secret":"CANARY"'}}]}
    with patch.object(llm.urllib.request, 'urlopen', return_value=Response()), patch.object(llm.json, 'load', return_value=payload):
        with pytest.raises(ValueError) as exc:
            llm.chat_json('system', 'user')
    assert 'finish_reason=\'length\'' in str(exc.value)
    assert 'raw_length=' in str(exc.value)
    assert 'CANARY' not in str(exc.value)
    assert exc.value.__cause__ is None
