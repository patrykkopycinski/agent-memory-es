"""Shared probe for tests that need the live reflect LLM gateway (AMES_LLM_BASE).

`test_hermes_adapter.py` and `test_reflect_llm.py` are script-style modules (module-level code that
calls the gateway), so an unreachable gateway used to crash COLLECTION of the whole suite with a
URLError. Call `require_llm(__name__)` at the top of such a module: under pytest it skips the module
with a visible reason; run as a script (`python tests/test_x.py`) it prints `SKIP: <reason>` to stderr
and exits 2.

One short, unretried probe (no http_retry: its default budget is 300 s; timeout from
AMES_LLM_PROBE_TIMEOUT_S, default 2). Any HTTP answer, even a 401/404, means the gateway is up;
only a transport failure means it is not.
"""
import os
import sys
import urllib.error
import urllib.request

PROBE_TIMEOUT_S = float(os.environ.get("AMES_LLM_PROBE_TIMEOUT_S", "2"))
_cache = {}


def llm_unreachable_reason():
    """None when the LLM gateway answers, else a one-line reason naming the endpoint."""
    from app import llm
    base = llm.BASE.rstrip("/")
    if base not in _cache:
        try:
            urllib.request.urlopen(base + "/models", timeout=PROBE_TIMEOUT_S).close()
            _cache[base] = None
        except urllib.error.HTTPError:
            _cache[base] = None
        except Exception as exc:
            _cache[base] = (f"LLM gateway {base} (AMES_LLM_BASE) is unreachable "
                            f"({type(exc).__name__}); this test needs a live LLM")
    return _cache[base]


def require_llm(module_name):
    """Module-level guard. Skips under pytest, prints to stderr and exits 2 when run as a script, no-op when reachable."""
    reason = llm_unreachable_reason()
    if reason is None:
        return
    if module_name == "__main__":
        print(f"SKIP: {reason}", file=sys.stderr)
        sys.exit(2)
    import pytest
    pytest.skip(reason, allow_module_level=True)
