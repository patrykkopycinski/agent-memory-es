"""Bounded retries for upstream HTTP calls, not store.es."""
import datetime as dt
import email.utils
import http.client
import json
import logging
import math
import os
import random
import socket
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)
TRANSIENT_STATUSES = {429, 502, 503, 504}
MAX_TRANSIENT_RETRIES = 12


def is_transient(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_STATUSES
    return isinstance(exc, (urllib.error.URLError, ConnectionRefusedError,
                            ConnectionResetError, http.client.RemoteDisconnected,
                            socket.timeout))


def budget_seconds():
    value = float(os.environ.get("AMES_HTTP_RETRY_BUDGET_S", "300"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("AMES_HTTP_RETRY_BUDGET_S must be finite and positive")
    return value


def _sleep_seconds(exc, attempt):
    value = exc.headers.get("Retry-After") if isinstance(exc, urllib.error.HTTPError) and exc.headers else None
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
                delay = (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = None
        if delay is not None and math.isfinite(delay):
            return min(120.0, max(0.0, delay))
    return random.uniform(0, min(60.0, 2.0 * 2 ** min(attempt - 1, 5)))


def _retry(operation, timeout, sleep=None, clock=None, budget=None):
    sleep = time.sleep if sleep is None else sleep
    clock = time.monotonic if clock is None else clock
    deadline = clock() + (budget_seconds() if budget is None else budget)
    attempt = 0
    while True:
        try:
            return operation(min(timeout, max(0.001, deadline - clock())))
        except Exception as exc:
            if not is_transient(exc):
                raise
            attempt += 1
            delay = _sleep_seconds(exc, attempt)
            if clock() + delay >= deadline:
                raise
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
            log.warning("HTTP retry status=%s attempt=%d sleep=%.3fs", status, attempt, delay)
            sleep(delay)
            if clock() >= deadline:
                raise


def urlopen(req, timeout=120, **kwargs):
    """Retry transport failures; clock and sleep are injectable."""
    return _retry(lambda remaining: urllib.request.urlopen(req, timeout=remaining), timeout, **kwargs)


def request_json(req, timeout=120, **kwargs):
    """Include response reads in transport retries; never retry JSON errors."""
    def operation(remaining):
        with urllib.request.urlopen(req, timeout=remaining) as response:
            return json.load(response)
    return _retry(operation, timeout, **kwargs)
