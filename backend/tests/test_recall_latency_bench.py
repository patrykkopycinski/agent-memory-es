"""Unit tests for scripts/recall_latency_bench.py statistics (no ES, no server)."""
import importlib.util
import os

SPEC = importlib.util.spec_from_file_location(
    "recall_latency_bench",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "recall_latency_bench.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_nearest_rank_percentiles():
    v = list(range(1, 101))
    assert bench.percentile(v, 50) == 50
    assert bench.percentile(v, 95) == 95
    assert bench.percentile(v, 99) == 99
    assert bench.percentile([7], 99) == 7
    assert bench.percentile([], 50) is None
    assert bench.percentile([3, 1, 2], 99) == 3     # n < 100: p99 is the max


def test_summarize_counts_rerank_fallbacks_and_abstains():
    mk = lambda total, rr, ab: {"total": total, "es_took": 2, "wall": total + 1, "context_chars": 100,
                                "abstained": ab, "reranked": rr, "has_timings": True}
    cell = bench.summarize([mk(10, True, False), mk(30, False, False), mk(20, False, True),
                            mk(40, True, False)], errors=1, rerank=True)
    assert cell["n"] == 4 and cell["errors"] == 1
    assert cell["rerank_fallbacks"] == 2 and cell["rerank_fallback_rate"] == 0.5
    assert cell["abstain_rate"] == 0.25
    assert cell["median_context_chars"] == 100
    assert cell["total_ms"]["p50"] == 20
    off = bench.summarize([mk(10, False, False)], errors=0, rerank=False)
    assert "rerank_fallbacks" not in off


def test_warmup_excluded_from_stats(monkeypatch):
    calls = []

    def fake_post(url, key, body, timeout):
        calls.append(body)
        t = 1000.0 if len(calls) <= 2 else 5.0       # first two (warm-up) are slow outliers
        return {"timings_ms": {"total": t, "es_took": 1}, "context_chars": 10,
                "abstained": False, "reranked": False}, t
    monkeypatch.setattr(bench, "post_recall", fake_post)

    class A:
        url, key, size, timeout, warmup, repeat, filter = "u", "k", 8, 1.0, 2, 1, None
    cell = bench.run_cell(A, [{"query": "q%d" % i} for i in range(6)], rerank=False, use_filter=False)
    assert len(calls) == 6 and cell["n"] == 4
    assert cell["total_ms"]["p99"] == 5.0


def test_summarize_reports_rerank_ms_and_fallback_reasons():
    def mk(rr, ms, why):
        return {"total": 50, "es_took": 5, "wall": 55, "context_chars": 100, "abstained": False,
                "reranked": rr, "rerank_ms": ms, "fallback_reason": why, "has_timings": True}
    cell = bench.summarize([mk(True, 40, None), mk(False, 2001, "timeout"), mk(False, 2002, "timeout"),
                            mk(False, 3, None)], errors=0, rerank=True)
    assert cell["rerank_fallback_reasons"] == {"timeout": 2, "unreported": 1}
    assert cell["rerank_ms"]["p50"] == 40 or cell["rerank_ms"]["n"] == 4
