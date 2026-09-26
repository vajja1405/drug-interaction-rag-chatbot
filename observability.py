"""
observability.py
────────────────
Prometheus metrics and optional OpenTelemetry tracing for the API.

  • /metrics                       Prometheus text format (scraped by docker-compose Prometheus)
  • http_request_duration_seconds  per route template, method and status → P50/P95/P99 in Grafana
  • retrieval_stage_seconds        bm25 / dense / rerank stage latency
  • evidence_cache_events_total    hit / miss for the evidence-search cache
  • llm_request_seconds            per inference provider (self-hosted vs external)
  • llm_fallback_total             requests routed to the fallback provider, by reason

Tracing is enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set, so tests and the free
hosted demo carry no collector dependency.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["route", "method"],
                         buckets=LATENCY_BUCKETS)
RETRIEVAL_STAGE = Histogram("retrieval_stage_seconds", "Retrieval stage latency", ["stage"],
                            buckets=LATENCY_BUCKETS)
CACHE_EVENTS = Counter("evidence_cache_events_total", "Evidence search cache events", ["result"])
LLM_LATENCY = Histogram("llm_request_seconds", "LLM request latency", ["provider"], buckets=LATENCY_BUCKETS)
LLM_FALLBACK = Counter("llm_fallback_total", "Requests served by the fallback provider", ["reason"])


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = route_template(request)
        if route != "/metrics":
            HTTP_REQUESTS.labels(route, request.method, str(status)).inc()
            HTTP_LATENCY.labels(route, request.method).observe(time.perf_counter() - start)


def metrics_endpoint(_: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_tracer = None


def setup_tracing(app) -> bool:
    """Wire OpenTelemetry → OTLP if an endpoint is configured; returns True when active."""
    global _tracer
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTEL endpoint set but opentelemetry packages are not installed")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": "drug-interaction-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    _tracer = trace.get_tracer("drug-interaction-api")
    return True


@contextmanager
def span(name: str, **attrs):
    """A child span when tracing is on; a no-op otherwise."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
