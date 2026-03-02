from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from os import getenv as os_getenv
from time import perf_counter as time_perf_counter
from typing import Protocol, cast

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

_TELEMETRY_SINGLETON: Telemetry | None = None


class _CollectorLike(Protocol):
    """Marker protocol for Prometheus collector instances."""


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _collector_by_name(name: str) -> _CollectorLike | None:
    mapping = getattr(REGISTRY, "_names_to_collectors", {})
    if name in mapping:
        return cast(_CollectorLike, mapping[name])
    total_name = f"{name}_total"
    if total_name in mapping:
        return cast(_CollectorLike, mapping[total_name])
    return None


def _get_or_create_counter(name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Counter:
    existing = _collector_by_name(name)
    if isinstance(existing, Counter):
        return existing
    return Counter(name=name, documentation=documentation, labelnames=labelnames)


def _get_or_create_gauge(name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Gauge:
    existing = _collector_by_name(name)
    if isinstance(existing, Gauge):
        return existing
    return Gauge(name=name, documentation=documentation, labelnames=labelnames)


def _get_or_create_histogram(name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Histogram:
    existing = _collector_by_name(name)
    if isinstance(existing, Histogram):
        return existing
    return Histogram(name=name, documentation=documentation, labelnames=labelnames)


class Telemetry:
    """Telemetry facade for tracing and metrics emission."""

    def __init__(self) -> None:
        service_name = os_getenv("OTEL_SERVICE_NAME", "ffreis-stock-simulator")
        service_version = os_getenv("OTEL_SERVICE_VERSION", "0.1.0")
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )

        otlp_endpoint = os_getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        otlp_enabled = _as_bool(os_getenv("TELEMETRY_OTLP_ENABLED"), default=bool(otlp_endpoint))
        prometheus_enabled = _as_bool(os_getenv("TELEMETRY_PROMETHEUS_ENABLED"), default=True)

        tracer_provider = TracerProvider(resource=resource)
        metric_readers: list[PeriodicExportingMetricReader] = []

        if otlp_enabled and otlp_endpoint:
            span_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            metric_exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics")
            metric_readers.append(PeriodicExportingMetricReader(metric_exporter))

        trace.set_tracer_provider(tracer_provider)
        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)

        self._tracer = trace.get_tracer("stock_simulator")
        self._meter = metrics.get_meter("stock_simulator")
        self._prometheus_enabled = prometheus_enabled
        self._config_hash = "unset"

        self._steps_counter = self._meter.create_counter("sim_steps_total")
        self._orders_counter = self._meter.create_counter("sim_orders_total")
        self._fills_counter = self._meter.create_counter("sim_fills_total")
        self._episodes_counter = self._meter.create_counter("sim_episodes_total")
        self._equity_delta_counter = self._meter.create_up_down_counter("sim_equity_delta")
        self._step_latency_hist = self._meter.create_histogram("sim_step_latency_seconds")

        self._prom_steps_counter = _get_or_create_counter("sim_steps_total", "Total simulator steps")
        self._prom_orders_counter = _get_or_create_counter("sim_orders_total", "Total simulator orders")
        self._prom_fills_counter = _get_or_create_counter("sim_fills_total", "Total simulator fills")
        self._prom_episodes_counter = _get_or_create_counter("sim_episodes_total", "Total simulator episodes")
        self._prom_equity_gauge = _get_or_create_gauge("sim_equity", "Current simulator equity")
        self._prom_step_latency_hist = _get_or_create_histogram(
            "sim_step_latency_seconds", "Simulator step latency seconds"
        )
        self._prom_config_info = _get_or_create_gauge(
            "sim_config_info",
            "Config hash info gauge (always 1)",
            labelnames=("config_hash",),
        )

    @property
    def tracer(self) -> trace.Tracer:
        return self._tracer

    def set_config_hash(self, config_hash: str) -> None:
        """Set stable configuration hash emitted in telemetry labels."""
        self._config_hash = config_hash
        if self._prometheus_enabled:
            self._prom_config_info.labels(config_hash=config_hash).set(1)

    @contextmanager
    def step_span(self, *, use_numba: bool, action_side: str, action_type: str) -> Iterator[None]:
        """Create parent span around one environment step."""
        start = time_perf_counter()
        with self._tracer.start_as_current_span("env.step") as span:
            span.set_attribute("sim.use_numba", use_numba)
            span.set_attribute("sim.action_side", action_side)
            span.set_attribute("sim.action_type", action_type)
            span.set_attribute("sim.config_hash", self._config_hash)
            try:
                yield
            finally:
                elapsed = time_perf_counter() - start
                self._step_latency_hist.record(
                    elapsed,
                    {"use_numba": str(use_numba), "config_hash": self._config_hash},
                )
                if self._prometheus_enabled:
                    self._prom_step_latency_hist.observe(elapsed)

    @contextmanager
    def child_span(self, name: str, attributes: dict[str, str | bool | int]) -> Iterator[None]:
        """Create child span with low-cardinality attributes."""
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield

    def on_step(
        self,
        *,
        equity: float,
        equity_delta: float,
        use_numba: bool,
        config_hash: str | None = None,
    ) -> None:
        """Record per-step metrics."""
        cfg_hash = self._config_hash if config_hash is None else config_hash
        attrs = {"use_numba": str(use_numba), "config_hash": cfg_hash}
        self._steps_counter.add(1, attrs)
        self._equity_delta_counter.add(equity_delta, attrs)
        if self._prometheus_enabled:
            self._prom_steps_counter.inc()
            self._prom_equity_gauge.set(equity)

    def on_order(self, *, side: str, order_type: str, units: float) -> None:
        """Record order submission event and counters."""
        attrs = {"side": side, "order_type": order_type}
        self._orders_counter.add(1, attrs)
        span = trace.get_current_span()
        if span is not None:
            span.add_event(
                "order",
                {"side": side, "order_type": order_type, "units": units},
            )
        if self._prometheus_enabled:
            self._prom_orders_counter.inc()

    def on_fill(self, *, count: int) -> None:
        """Record fill counters/events."""
        if count <= 0:
            return
        self._fills_counter.add(count)
        span = trace.get_current_span()
        if span is not None:
            span.add_event("fill", {"count": count})
        if self._prometheus_enabled:
            self._prom_fills_counter.inc(count)

    def on_episode_end(self, *, steps: int, final_equity: float) -> None:
        """Record end-of-episode aggregates."""
        self._episodes_counter.add(1)
        span = trace.get_current_span()
        if span is not None:
            span.add_event(
                "episode_end",
                {"steps": steps, "final_equity": final_equity},
            )
        if self._prometheus_enabled:
            self._prom_episodes_counter.inc()
            self._prom_equity_gauge.set(final_equity)


def get_telemetry() -> Telemetry:
    global _TELEMETRY_SINGLETON
    if _TELEMETRY_SINGLETON is None:
        _TELEMETRY_SINGLETON = Telemetry()
    return _TELEMETRY_SINGLETON


telemetry = get_telemetry()
