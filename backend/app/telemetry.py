"""OpenTelemetry wiring for the AMES backend.

Three things live here:

* `configure()` — builds the SDK TracerProvider/MeterProvider with an OTLP
  http/protobuf exporter and DELTA metrics temporality (the EDOT collector's
  elasticsearch exporter drops cumulative histograms), resource attributes
  `service.name` (OTEL_SERVICE_NAME, default `agent-memory-es`) and
  `deployment.environment.name` (AMES_DEPLOYMENT_ENV, default `prod`).
* `span()` — the one context manager the app uses for its own spans. Never
  raises, never crashes the request when telemetry is off or broken.
* attribute helpers (`set_attrs`, `record_recall`, `record_retain`, …) so the
  privacy rule lives in ONE place: spans carry kinds, visibility, counts,
  lengths and timings. Never memory text, never an API key, never an owner id
  in the clear (only the same truncated sha256 the recall log line uses).

Everything degrades to a no-op when opentelemetry is not installed, so the
package still imports in a minimal environment.
"""
from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

# --- optional imports -------------------------------------------------------
try:  # pragma: no cover - exercised implicitly in the otel-less environment
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

    _HAVE_API = True
except Exception:  # pragma: no cover
    trace = None
    Span = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    _HAVE_API = False

try:  # pragma: no cover
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAVE_SDK = True
except Exception:  # pragma: no cover
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    _HAVE_SDK = False

INSTRUMENTING_NAME = "ames.backend"
INSTRUMENTING_VERSION = "0.1.0"

DEFAULT_SERVICE_NAME = "agent-memory-es"
DEFAULT_OTLP_ENDPOINT = os.environ.get("AMES_OTEL_ENDPOINT", "http://localhost:4318")

# Module-level handles. `configure()` owns them; tests inject their own
# provider so they never depend on the process-global provider (which the SDK
# only lets you set once).
_state: dict = {"tracer_provider": None, "meter_provider": None, "configured": False}


def _truthy(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """AMES_OTEL_ENABLED wins; otherwise enabled unless explicitly off.

    Off by default only when the operator opted out or the API is missing —
    a backend that cannot reach the collector still exports into a void
    (the exporter swallows and retries), which is not a reason to disable.
    """
    if not _HAVE_API:
        return False
    return _truthy(os.environ.get("AMES_OTEL_ENABLED"), True)


def owner_hash(owner_id: str) -> str:
    """Same truncated digest the `ames.recall` log line already carries."""
    return hashlib.sha256((owner_id or "").encode()).hexdigest()[:12]


def _otlp_endpoint(suffix: str) -> str:
    """Full endpoint for signals, honouring the standard OTEL env vars."""
    explicit = os.environ.get(f"OTEL_EXPORTER_OTLP_{suffix.upper()}_ENDPOINT")
    if explicit:
        return explicit
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or DEFAULT_OTLP_ENDPOINT
    return base.rstrip("/") + f"/v1/{suffix}"


def configure(force: bool = False, tracer_provider=None,
              span_exporter=None) -> Optional[Any]:
    """Install the telemetry pipeline. Idempotent; returns the TracerProvider.

    tracer_provider / span_exporter are test seams: pass a provider built with
    a TestSpanProcessor and nothing touches the network.
    """
    if _state["configured"] and not force and tracer_provider is None:
        return _state["tracer_provider"]
    if not enabled() and tracer_provider is None:
        _state["configured"] = True
        return None

    provider = tracer_provider
    if provider is None:
        if not _HAVE_SDK:
            _state["configured"] = True
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            resource_attrs = {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME),
                "deployment.environment.name": os.environ.get(
                    "AMES_DEPLOYMENT_ENV", "prod"),
                "service.version": os.environ.get("AMES_VERSION", "0.2.0"),
            }
            provider = TracerProvider(resource=Resource.create(resource_attrs))
            exporter = span_exporter or OTLPSpanExporter(
                endpoint=_otlp_endpoint("traces"), timeout=10)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            _maybe_start_metrics(provider)
        except Exception as exc:  # telemetry must never take the service down
            _state["configured"] = True
            _warn(f"telemetry disabled: {type(exc).__name__}: {exc}")
            return None

    _state["tracer_provider"] = provider
    _state["configured"] = True
    try:  # best effort: the SDK keeps the first global provider it is given
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    return provider


def _maybe_start_metrics(provider) -> None:
    """OTLP metrics with DELTA temporality (the collector needs per-interval)."""
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk import metrics as _sm
        from opentelemetry.sdk.metrics.export import (
            AggregationTemporality,
            PeriodicExportingMetricReader,
        )

        # DELTA for every instrument type: the EDOT elasticsearch exporter
        # drops cumulative-temporality histograms. Class names move between
        # SDK versions (sync Gauge is `_Gauge` in some), so resolve defensively.
        _delta = {}
        for _name in ("Counter", "UpDownCounter", "Histogram", "Gauge",
                      "ObservableCounter", "ObservableUpDownCounter",
                      "ObservableGauge"):
            _cls = getattr(_sm, _name, None) or getattr(_sm, "_" + _name, None)
            if _cls is not None:
                _delta[_cls] = AggregationTemporality.DELTA

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=_otlp_endpoint("metrics"),
                               preferred_temporality=_delta),
            export_interval_millis=int(os.environ.get("AMES_OTEL_METRIC_INTERVAL_MS", "15000")),
        )
        mp = _sm.MeterProvider(metric_readers=[reader], resource=provider.resource)
        _state["meter_provider"] = mp
        try:
            from opentelemetry import metrics

            metrics.set_meter_provider(mp)
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - metrics are additive
        _warn(f"metrics disabled: {type(exc).__name__}: {exc}")


_warned: set = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        import logging

        logging.getLogger("ames.telemetry").warning(msg)


def tracer():
    """The tracer spans are created from (module provider, else global)."""
    if not _HAVE_API:
        return None
    provider = _state.get("tracer_provider")
    if provider is not None:
        return provider.get_tracer(INSTRUMENTING_NAME, INSTRUMENTING_VERSION)
    return trace.get_tracer(INSTRUMENTING_NAME, INSTRUMENTING_VERSION)


def current_span():
    if not _HAVE_API:
        return None
    try:
        span = trace.get_current_span()
    except Exception:
        return None
    if span is None or not span.is_recording():
        provider = _state.get("tracer_provider")
        if provider is not None and hasattr(provider, "get_current_span"):
            span = provider.get_current_span()
    return span


def set_attrs(attrs: dict) -> None:
    """Set attributes on the active span, dropping None / secret-ish values."""
    span = current_span()
    if span is None or attrs is None:
        return
    try:
        if not span.is_recording():
            return
        for key, value in attrs.items():
            clean = _clean(value)
            if clean is not None:
                span.set_attribute(key, clean)
    except Exception:
        pass


def add_event(name: str, attrs: Optional[dict] = None) -> None:
    span = current_span()
    if span is None:
        return
    try:
        if span.is_recording():
            span.add_event(name, {k: v for k, v in (attrs or {}).items()
                                  if _clean(v) is not None} or None)
    except Exception:
        pass


def _clean(value):
    """OTLP attributes must be scalars or homogeneous scalar sequences."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return None  # never flatten free-form dicts: they can hold text/keys
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [v for v in value if isinstance(v, (bool, int, float, str))]
        if not items:
            return None
        if len(items) > 32:
            items = items[:32]
        return items
    return str(value)[:256]


@contextmanager
def span(name: str, attrs: Optional[dict] = None, kind: Optional[str] = None):
    """Record a span; yields the live span (or None when telemetry is off).

    Never swallows the body's exception — it records it on the span and
    re-raises, so error handling stays in the routes.
    """
    tr = tracer()
    if tr is None:
        yield None
        return
    try:
        cm = tr.start_as_current_span(name)
    except Exception:
        yield None
        return
    with cm as sp:
        try:
            local = dict(attrs or {})
            if kind:
                local["memory.kind"] = kind
            set_attrs(local)
        except Exception:
            pass
        try:
            yield sp
        except Exception as exc:
            try:
                if sp is not None and sp.is_recording():
                    sp.record_exception(exc)
                    sp.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            except Exception:
                pass
            raise
        else:
            try:
                if sp is not None and sp.is_recording():
                    sp.set_status(Status(StatusCode.OK))
            except Exception:
                pass


class timer:
    """Context manager that records `<name>_ms` on the active span."""

    def __init__(self, attr: str, span_obj=None):
        self.attr = attr
        self._span = span_obj
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
        sp = self._span if self._span is not None else current_span()
        try:
            if sp is not None and sp.is_recording():
                sp.set_attribute(self.attr, ms)
        except Exception:
            pass
        return False


# --- attribute vocabulary ---------------------------------------------------

KINDS_ALL = ("episodic", "semantic", "procedural")


def _kind_attrs(kinds: Optional[Iterable[str]]) -> dict:
    resolved = [k for k in (kinds or KINDS_ALL)]
    out: dict = {}
    if len(resolved) == 1:
        out["memory.kind"] = resolved[0]
    out["memory.kinds"] = resolved
    return out


def recall_attrs(owner_id: str, query: str, kinds, size: int,
                 out: dict) -> dict:
    """Attributes for a recall: kinds, visibility mix, hit counts. No query text."""
    hits = out.get("fused") or out.get("results") or []
    by_kind: dict = {}
    by_vis: dict = {}
    for h in hits:
        by_kind[h.get("kind", "?")] = by_kind.get(h.get("kind", "?"), 0) + 1
        vis = h.get("visibility") or "unknown"
        by_vis[vis] = by_vis.get(vis, 0) + 1
    attrs = {
        "ames.owner_hash": owner_hash(owner_id),
        "memory.query.length": len(query or ""),
        "memory.size": size,
        "memory.hits.total": len(hits),
        "memory.abstained": bool(out.get("abstained")),
        "memory.doc_groups": out.get("doc_groups", 0),
        "memory.context_chars": out.get("context_chars", 0),
        "memory.visibility.distinct": sorted(by_vis),
    }
    attrs.update(_kind_attrs(kinds))
    for kind, count in by_kind.items():
        attrs[f"memory.hits.kind.{kind}"] = count
    for vis, count in by_vis.items():
        attrs[f"memory.visibility.{vis}"] = count
    rerank = out.get("rerank")
    if isinstance(rerank, dict):
        attrs["memory.rerank.requested"] = bool(rerank.get("requested"))
        attrs["memory.rerank.applied"] = bool(rerank.get("applied"))
    return attrs


def retain_attrs(owner_id: str, kind: str, text: str, visibility: str,
                 out: dict) -> dict:
    attrs = {
        "ames.owner_hash": owner_hash(owner_id),
        "memory.kind": kind,
        "memory.visibility": visibility,
        "memory.text_chars": len(text or ""),
    }
    if isinstance(out, dict):
        attrs["memory.written"] = bool(out.get("written", True))
        attrs["memory.deduplicated"] = bool(out.get("deduplicated"))
        passages = out.get("passages")
        if isinstance(passages, int):
            attrs["memory.passages"] = passages
        attrs["memory.doc_group.present"] = bool(out.get("doc_group"))
        attrs["memory.fact_extraction.queued"] = bool(out.get("fact_job"))
    return attrs


def promote_attrs(owner_id: str, kind: str, to_visibility: str) -> dict:
    return {"ames.owner_hash": owner_hash(owner_id), "memory.kind": kind,
            "memory.visibility": to_visibility}


def reflect_attrs(owner_id: str, question: str, out: dict) -> dict:
    attrs = {"ames.owner_hash": owner_hash(owner_id),
             "memory.query.length": len(question or "")}
    if isinstance(out, dict):
        sources = out.get("sources") or []
        evidence = out.get("evidence") or []
        queries = out.get("queries") or []
        attrs["memory.reflect.queries"] = len(queries)
        attrs["memory.hits.total"] = len(sources)
        attrs["memory.evidence.total"] = len(evidence)
        attrs["memory.synthesized"] = bool(out.get("synthesized"))
        attrs["memory.abstained"] = out.get("answer") == "INSUFFICIENT_EVIDENCE"
        tiers = sorted({s.get("tier") for s in evidence if isinstance(s, dict) and s.get("tier")})
        if tiers:
            attrs["memory.tiers"] = tiers
    return attrs


def es_attrs(method: str, path: str, response: Optional[dict]) -> dict:
    """ES client span attributes: index pattern + outcome counts (never bodies)."""
    seg = [s for s in (path or "").split("?")[0].split("/") if s]
    index = seg[0] if seg else ""
    operation = seg[1] if len(seg) > 1 and seg[1].startswith("_") else (
        seg[-1] if seg else "")
    attrs = {
        "db.system": "elasticsearch",
        "db.operation": method,
        "es.index": index,
        "es.index_pattern": _index_pattern(index),
        "server.address": os.environ.get("AMES_ES_HOST", ""),
    }
    if isinstance(response, dict):
        took = response.get("took")
        if isinstance(took, (int, float)):
            attrs["es.took_ms"] = float(took)
        hits = response.get("hits")
        if isinstance(hits, dict):
            total = hits.get("total")
            if isinstance(total, dict):
                attrs["es.hits.total"] = total.get("value", 0)
            elif isinstance(total, int):
                attrs["es.hits.total"] = total
        if "result" in response:
            attrs["es.result"] = str(response["result"])[:16]
        if response.get("errors"):
            attrs["es.errors"] = True
        indexed = response.get("items")
        if isinstance(indexed, list):
            attrs["es.items"] = len(indexed)
    return {k: v for k, v in attrs.items() if _clean(v) is not None}


def _index_pattern(index: str) -> str:
    """Normalise a concrete index name to the pattern it belongs to.

    amtest_am_semantic -> am_semantic_*   /  am_am_semantic -> am_semantic_*
    The AMES_INDEX_PREFIX is test-isolation noise; what an operator groups by is
    <collection>_<kind>_* so the kind stays in the group-by key.
    """
    if not index:
        return ""
    name = index
    prefix = os.environ.get("AMES_INDEX_PREFIX", "")
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    if name.startswith("_"):
        return name
    parts = name.split("_")
    return "_".join(parts[:2]) + "_*" if len(parts) > 1 else name


def instrument_fastapi(app, tracer_provider=None, force: bool = False) -> bool:
    """Attach FastAPI server spans (and W3C trace-context propagation).

    `tracer_provider` pins server spans to one specific provider. The tests
    need this: the SDK only honours the first process-global provider, so a
    suite that wants its spans in an in-memory exporter cannot get them there
    through the global. Explicit provider also bypasses the AMES_OTEL_ENABLED
    gate — a test that hands us a provider has opted in.
    """
    if tracer_provider is None and not enabled():
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        if getattr(app, "_ames_otel_instrumented", False):
            if not force:
                return True
            try:
                FastAPIInstrumentor.uninstrument_app(app)
            except Exception:
                pass
            app._ames_otel_instrumented = False
        kwargs = {"excluded_urls": "health,metrics"}
        provider = tracer_provider or _state.get("tracer_provider")
        if provider is not None:
            kwargs["tracer_provider"] = provider
        FastAPIInstrumentor.instrument_app(app, **kwargs)
        app._ames_otel_instrumented = True
        # The instrumenter hooks `build_middleware_stack`; Starlette caches the
        # built stack on `app.middleware_stack` on the first request. If any
        # request was served before we attached (test suites that hit the app
        # before this runs), the cache holds a stack WITHOUT the OTel middleware
        # and server spans silently vanish. Drop the cache so the patched
        # builder runs on the next request.
        if getattr(app, "middleware_stack", None) is not None:
            app.middleware_stack = None
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        _warn(f"fastapi instrumentation disabled: {type(exc).__name__}: {exc}")
        return False


def shutdown() -> None:
    """Flush exporters (used at app shutdown and by the tests)."""
    for key in ("tracer_provider", "meter_provider"):
        provider = _state.get(key)
        try:
            if provider is not None:
                provider.shutdown()
        except Exception:
            pass


def force_flush() -> None:
    for key in ("tracer_provider", "meter_provider"):
        provider = _state.get(key)
        try:
            if provider is not None:
                provider.force_flush()
        except Exception:
            pass


def reset() -> None:
    """Test helper: forget the configured provider."""
    _state.update({"tracer_provider": None, "meter_provider": None,
                   "configured": False})
