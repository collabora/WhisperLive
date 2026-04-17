import unittest
from unittest.mock import patch, MagicMock

from whisper_live import metrics as wl_metrics

_skip_no_prometheus = unittest.skipUnless(
    wl_metrics.is_available(), "prometheus_client not installed"
)


class TestMetricsAvailability(unittest.TestCase):
    def test_is_available_returns_bool(self):
        self.assertIsInstance(wl_metrics.is_available(), bool)


@_skip_no_prometheus
class TestTrackConnectionOpened(unittest.TestCase):
    def test_increments_total_and_active(self):
        total_before = wl_metrics.CONNECTIONS_TOTAL._value.get()
        active_before = wl_metrics.CONNECTIONS_ACTIVE._value.get()
        wl_metrics.track_connection_opened()
        self.assertEqual(wl_metrics.CONNECTIONS_TOTAL._value.get(), total_before + 1)
        self.assertEqual(wl_metrics.CONNECTIONS_ACTIVE._value.get(), active_before + 1)


@_skip_no_prometheus
class TestTrackConnectionClosed(unittest.TestCase):
    def test_decrements_active(self):
        wl_metrics.track_connection_opened()
        active_before = wl_metrics.CONNECTIONS_ACTIVE._value.get()
        wl_metrics.track_connection_closed()
        self.assertEqual(wl_metrics.CONNECTIONS_ACTIVE._value.get(), active_before - 1)


@_skip_no_prometheus
class TestTrackConnectionRejected(unittest.TestCase):
    def test_rejected_full(self):
        before = wl_metrics.CONNECTIONS_REJECTED.labels(reason="full")._value.get()
        wl_metrics.track_connection_rejected(reason="full")
        self.assertEqual(wl_metrics.CONNECTIONS_REJECTED.labels(reason="full")._value.get(), before + 1)

    def test_rejected_auth(self):
        before = wl_metrics.CONNECTIONS_REJECTED.labels(reason="auth")._value.get()
        wl_metrics.track_connection_rejected(reason="auth")
        self.assertEqual(wl_metrics.CONNECTIONS_REJECTED.labels(reason="auth")._value.get(), before + 1)


@_skip_no_prometheus
class TestTrackTranscriptionLatency(unittest.TestCase):
    def test_observe_records_value(self):
        count_before = wl_metrics.TRANSCRIPTION_LATENCY._sum.get()
        wl_metrics.track_transcription_latency(0.5)
        self.assertAlmostEqual(wl_metrics.TRANSCRIPTION_LATENCY._sum.get(), count_before + 0.5, places=3)


@_skip_no_prometheus
class TestTrackAudioProcessed(unittest.TestCase):
    def test_increments_by_duration(self):
        before = wl_metrics.AUDIO_PROCESSED._value.get()
        wl_metrics.track_audio_processed(3.5)
        self.assertAlmostEqual(wl_metrics.AUDIO_PROCESSED._value.get(), before + 3.5, places=3)


@_skip_no_prometheus
class TestTrackSegmentEmitted(unittest.TestCase):
    def test_completed_true(self):
        before = wl_metrics.SEGMENTS_EMITTED.labels(completed="true")._value.get()
        wl_metrics.track_segment_emitted(completed=True)
        self.assertEqual(wl_metrics.SEGMENTS_EMITTED.labels(completed="true")._value.get(), before + 1)

    def test_completed_false(self):
        before = wl_metrics.SEGMENTS_EMITTED.labels(completed="false")._value.get()
        wl_metrics.track_segment_emitted(completed=False)
        self.assertEqual(wl_metrics.SEGMENTS_EMITTED.labels(completed="false")._value.get(), before + 1)


@_skip_no_prometheus
class TestTrackRestRequest(unittest.TestCase):
    def test_tracks_200(self):
        before = wl_metrics.REST_REQUESTS.labels(endpoint="transcriptions", status="200")._value.get()
        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
        self.assertEqual(wl_metrics.REST_REQUESTS.labels(endpoint="transcriptions", status="200")._value.get(), before + 1)

    def test_tracks_500(self):
        before = wl_metrics.REST_REQUESTS.labels(endpoint="transcriptions", status="500")._value.get()
        wl_metrics.track_rest_request(endpoint="transcriptions", status=500)
        self.assertEqual(wl_metrics.REST_REQUESTS.labels(endpoint="transcriptions", status="500")._value.get(), before + 1)


@_skip_no_prometheus
class TestTrackError(unittest.TestCase):
    def test_tracks_transcription_error(self):
        before = wl_metrics.ERRORS.labels(type="transcription")._value.get()
        wl_metrics.track_error("transcription")
        self.assertEqual(wl_metrics.ERRORS.labels(type="transcription")._value.get(), before + 1)

    def test_tracks_rest_error(self):
        before = wl_metrics.ERRORS.labels(type="rest_transcription")._value.get()
        wl_metrics.track_error("rest_transcription")
        self.assertEqual(wl_metrics.ERRORS.labels(type="rest_transcription")._value.get(), before + 1)


@_skip_no_prometheus
class TestStartMetricsServer(unittest.TestCase):
    @patch("whisper_live.metrics.start_http_server")
    def test_starts_on_given_port(self, mock_start):
        wl_metrics.start_metrics_server(9999)
        mock_start.assert_called_once_with(9999)

    @patch("whisper_live.metrics.start_http_server", side_effect=OSError("port in use"))
    def test_logs_error_on_failure(self, mock_start):
        with self.assertLogs(level="ERROR") as cm:
            wl_metrics.start_metrics_server(9999)
        self.assertTrue(any("Failed to start" in msg for msg in cm.output))


class TestNoOpWhenUnavailable(unittest.TestCase):
    """Verify helper functions are no-ops when _AVAILABLE is False."""

    def test_all_helpers_are_noop(self):
        original = wl_metrics._AVAILABLE
        try:
            wl_metrics._AVAILABLE = False
            # None of these should raise
            wl_metrics.track_connection_opened()
            wl_metrics.track_connection_closed()
            wl_metrics.track_connection_rejected("full")
            wl_metrics.track_transcription_latency(1.0)
            wl_metrics.track_audio_processed(1.0)
            wl_metrics.track_segment_emitted()
            wl_metrics.track_rest_request()
            wl_metrics.track_error()
        finally:
            wl_metrics._AVAILABLE = original


if __name__ == "__main__":
    unittest.main()
