"""
Prometheus metrics for WhisperLive server.

Exposes a /metrics HTTP endpoint on a configurable port for Prometheus scraping.
All metrics are optional — the server works fine without prometheus_client installed.
"""

import logging
import threading

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    CONNECTIONS_TOTAL = Counter(
        "whisperlive_connections_total",
        "Total WebSocket connections accepted",
    )
    CONNECTIONS_ACTIVE = Gauge(
        "whisperlive_connections_active",
        "Currently active WebSocket connections",
    )
    CONNECTIONS_REJECTED = Counter(
        "whisperlive_connections_rejected_total",
        "Connections rejected (server full or auth failure)",
        ["reason"],
    )
    TRANSCRIPTION_LATENCY = Histogram(
        "whisperlive_transcription_latency_seconds",
        "Time to transcribe a single audio chunk",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    AUDIO_PROCESSED = Counter(
        "whisperlive_audio_processed_seconds_total",
        "Total seconds of audio processed",
    )
    SEGMENTS_EMITTED = Counter(
        "whisperlive_segments_emitted_total",
        "Total transcription segments sent to clients",
        ["completed"],
    )
    REST_REQUESTS = Counter(
        "whisperlive_rest_requests_total",
        "Total REST API requests",
        ["endpoint", "status"],
    )
    ERRORS = Counter(
        "whisperlive_errors_total",
        "Total errors by type",
        ["type"],
    )

    _AVAILABLE = True

except ImportError:
    _AVAILABLE = False


def is_available():
    """Check if prometheus_client is installed."""
    return _AVAILABLE


def start_metrics_server(port=9091):
    """Start the Prometheus metrics HTTP server on the given port.

    Args:
        port (int): Port to serve /metrics on. Default 9091.
    """
    if not _AVAILABLE:
        logging.warning("prometheus_client not installed; metrics endpoint disabled")
        return
    try:
        start_http_server(port)
        logging.info(f"Prometheus metrics available at http://0.0.0.0:{port}/metrics")
    except Exception as e:
        logging.error(f"Failed to start metrics server: {e}")


def track_connection_opened():
    if _AVAILABLE:
        CONNECTIONS_TOTAL.inc()
        CONNECTIONS_ACTIVE.inc()


def track_connection_closed():
    if _AVAILABLE:
        CONNECTIONS_ACTIVE.dec()


def track_connection_rejected(reason="full"):
    if _AVAILABLE:
        CONNECTIONS_REJECTED.labels(reason=reason).inc()


def track_transcription_latency(seconds):
    if _AVAILABLE:
        TRANSCRIPTION_LATENCY.observe(seconds)


def track_audio_processed(seconds):
    if _AVAILABLE:
        AUDIO_PROCESSED.inc(seconds)


def track_segment_emitted(completed=True):
    if _AVAILABLE:
        SEGMENTS_EMITTED.labels(completed=str(completed).lower()).inc()


def track_rest_request(endpoint="/v1/audio/transcriptions", status="200"):
    if _AVAILABLE:
        REST_REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


def track_error(error_type="transcription"):
    if _AVAILABLE:
        ERRORS.labels(type=error_type).inc()
