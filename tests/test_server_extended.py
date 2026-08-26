import json
import time
import threading
import collections
import unittest
from http import HTTPStatus
from unittest import mock
from unittest.mock import MagicMock, patch
from websockets.http11 import Request

from whisper_live.server import TranscriptionServer, BackendType, ClientManager, _websocket_auth


class TestClientManagerAddRemove(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=2, max_connection_time=60)

    def test_add_and_get_client(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.assertIs(self.cm.get_client(ws), client)

    def test_get_nonexistent_client(self):
        ws = MagicMock()
        self.assertFalse(self.cm.get_client(ws))

    def test_remove_client_calls_cleanup(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.cm.remove_client(ws)
        client.cleanup.assert_called_once()
        self.assertNotIn(ws, self.cm.clients)
        self.assertNotIn(ws, self.cm.start_times)

    def test_remove_nonexistent_client_no_error(self):
        ws = MagicMock()
        self.cm.remove_client(ws)  # should not raise


class TestClientManagerThreadSafety(unittest.TestCase):
    def test_concurrent_add_remove(self):
        cm = ClientManager(max_clients=100, max_connection_time=600)
        errors = []

        def add_clients(start_idx):
            try:
                for i in range(50):
                    ws = MagicMock(name=f"ws-{start_idx}-{i}")
                    client = MagicMock(name=f"client-{start_idx}-{i}")
                    cm.add_client(ws, client)
            except Exception as e:
                errors.append(e)

        def remove_clients():
            try:
                for _ in range(25):
                    with cm.lock:
                        if cm.clients:
                            ws = next(iter(cm.clients))
                        else:
                            continue
                    cm.remove_client(ws)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_clients, args=(0,)),
            threading.Thread(target=add_clients, args=(1,)),
            threading.Thread(target=remove_clients),
            threading.Thread(target=remove_clients),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_get_client(self):
        cm = ClientManager(max_clients=100, max_connection_time=600)
        ws = MagicMock()
        client = MagicMock()
        cm.add_client(ws, client)
        errors = []
        results = []

        def get_many():
            try:
                for _ in range(100):
                    results.append(cm.get_client(ws))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertTrue(all(r is client for r in results))


class TestClientManagerServerFull(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=1, max_connection_time=60)

    def test_not_full_returns_false(self):
        ws = MagicMock()
        options = {"uid": "test"}
        self.assertFalse(self.cm.is_server_full(ws, options))

    def test_full_sends_wait_and_returns_true(self):
        ws1 = MagicMock()
        self.cm.add_client(ws1, MagicMock())

        ws2 = MagicMock()
        options = {"uid": "new-client"}
        self.assertTrue(self.cm.is_server_full(ws2, options))
        ws2.send.assert_called_once()
        sent = json.loads(ws2.send.call_args[0][0])
        self.assertEqual(sent["status"], "WAIT")
        self.assertEqual(sent["uid"], "new-client")


class TestClientManagerTimeout(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=4, max_connection_time=10)

    def test_not_timed_out(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.assertFalse(self.cm.is_client_timeout(ws))

    def test_timed_out(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.cm.start_times[ws] = time.time() - 20
        self.assertTrue(self.cm.is_client_timeout(ws))
        client.disconnect.assert_called_once()


class TestClientManagerGetWaitTime(unittest.TestCase):
    def test_no_clients_returns_zero(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        self.assertEqual(cm.get_wait_time(), 0)

    def test_single_client_wait_time(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        ws = MagicMock()
        cm.add_client(ws, MagicMock())
        cm.start_times[ws] = time.time() - 300
        wait = cm.get_wait_time()
        self.assertAlmostEqual(wait, 5.0, places=0)

    def test_multiple_clients_returns_minimum(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        ws1, ws2 = MagicMock(), MagicMock()
        cm.add_client(ws1, MagicMock())
        cm.add_client(ws2, MagicMock())
        cm.start_times[ws1] = time.time() - 100
        cm.start_times[ws2] = time.time() - 500
        wait = cm.get_wait_time()
        # ws2 has 100s remaining = ~1.67 minutes
        self.assertAlmostEqual(wait, 100 / 60, places=0)


class TestBackendType(unittest.TestCase):
    def test_valid_types(self):
        valid = BackendType.valid_types()
        self.assertIn("faster_whisper", valid)
        self.assertIn("tensorrt", valid)
        self.assertIn("openvino", valid)

    def test_is_valid(self):
        self.assertTrue(BackendType.is_valid("faster_whisper"))
        self.assertFalse(BackendType.is_valid("nonexistent"))

    def test_type_checks(self):
        self.assertTrue(BackendType.FASTER_WHISPER.is_faster_whisper())
        self.assertFalse(BackendType.FASTER_WHISPER.is_tensorrt())
        self.assertTrue(BackendType.TENSORRT.is_tensorrt())
        self.assertTrue(BackendType.OPENVINO.is_openvino())

    def test_enum_from_string(self):
        bt = BackendType("faster_whisper")
        self.assertEqual(bt, BackendType.FASTER_WHISPER)

    def test_invalid_enum_raises(self):
        with self.assertRaises(ValueError):
            BackendType("invalid_backend")


class TestTranscriptionServerInit(unittest.TestCase):
    def test_defaults(self):
        server = TranscriptionServer()
        self.assertIsNone(server.client_manager)
        self.assertTrue(server.use_vad)
        self.assertFalse(server.single_model)
        self.assertIsNone(server.batch_config)

    def test_run_invalid_backend_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, backend="nonexistent")

    def test_run_invalid_trt_path_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(
                host="localhost",
                port=9090,
                backend="tensorrt",
                whisper_tensorrt_path="/nonexistent/path",
            )

    def test_run_max_clients_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_clients=0)

    def test_run_max_clients_negative_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_clients=-1)

    def test_run_max_connection_time_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_connection_time=0)

    def test_run_batch_max_size_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, batch_enabled=True, batch_max_size=0)

    def test_run_batch_window_ms_negative_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, batch_enabled=True, batch_window_ms=-1)


class TestTranscriptionServerGetAudio(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()

    def test_end_of_audio_returns_false(self):
        ws = MagicMock()
        ws.recv.return_value = b"END_OF_AUDIO"
        result = self.server.get_audio_from_websocket(ws)
        self.assertFalse(result)

    def test_valid_audio_returns_numpy(self):
        import numpy as np
        ws = MagicMock()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ws.recv.return_value = audio.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        np.testing.assert_array_almost_equal(result, audio)

    def test_raw_pcm_input_normalizes_int16(self):
        import numpy as np
        self.server.raw_pcm_input = True
        ws = MagicMock()
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        ws.recv.return_value = pcm.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        expected = pcm.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected)
        self.assertTrue(result.dtype == np.float32)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_uint8_audio_format_normalizes_unsigned_pcm(self):
        import numpy as np
        ws = MagicMock()
        self.server.audio_formats[ws] = "uint8"
        pcm = np.array([0, 128, 255], dtype=np.uint8)
        ws.recv.return_value = pcm.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        expected = (pcm.astype(np.float32) - 128.0) / 128.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_raw_pcm_input_off_reads_float32(self):
        import numpy as np
        self.server.raw_pcm_input = False
        ws = MagicMock()
        audio = np.array([0.5, -0.5], dtype=np.float32)
        ws.recv.return_value = audio.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        np.testing.assert_array_almost_equal(result, audio)


class TestTranscriptionServerHandleNewConnection(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        self.server.cache_path = "~/.cache/whisper-live/"
        self.server.backend = BackendType.FASTER_WHISPER

    @mock.patch("websockets.WebSocketCommonProtocol")
    def test_invalid_json_returns_false(self, mock_ws):
        mock_ws.recv.return_value = "not valid json {{"
        result = self.server.handle_new_connection(mock_ws, None, None, False)
        self.assertFalse(result)

    @mock.patch("websockets.WebSocketCommonProtocol")
    def test_server_full_returns_false(self, mock_ws):
        # Fill server
        for i in range(4):
            self.server.client_manager.add_client(MagicMock(), MagicMock())

        mock_ws.recv.return_value = json.dumps({
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "tiny.en",
        })
        result = self.server.handle_new_connection(mock_ws, None, None, False)
        self.assertFalse(result)


class TestTranscriptionServerCleanup(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=4, max_connection_time=600)

    def test_cleanup_removes_client(self):
        ws = MagicMock()
        client = MagicMock()
        self.server.client_manager.add_client(ws, client)
        self.cleanup_server = self.server
        self.server.cleanup(ws)
        self.assertNotIn(ws, self.server.client_manager.clients)
        client.cleanup.assert_called_once()


class TestStreamTranscription(unittest.TestCase):
    """Tests for the SSE streaming endpoint (stream=true)."""

    def _make_app(self):
        """Create a FastAPI app with the transcribe endpoint that has streaming support."""
        from fastapi import FastAPI, UploadFile, Form

        app = FastAPI()
        server = TranscriptionServer()

        @app.post("/v1/audio/transcriptions")
        async def transcribe(
            file: UploadFile,
            stream: bool = Form(default=False),
            language: str = Form(default=None),
            response_format: str = Form(default="json"),
        ):
            if stream:
                return server._stream_transcription(
                    file, language, None, 0.0, None, None
                )
            return {"text": "non-streamed"}

        return app

    @patch("whisper_live.server.WhisperModel")
    def test_stream_returns_sse_content_type(self, mock_model_cls):
        mock_seg = MagicMock()
        mock_seg.id = 0
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = " hello "
        mock_seg.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98
        mock_info.duration = 1.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_model_cls.return_value = mock_model

        import io
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

    @patch("whisper_live.server.WhisperModel")
    def test_stream_yields_segment_and_done(self, mock_model_cls):
        mock_seg = MagicMock()
        mock_seg.id = 0
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.text = " hello world "
        mock_seg.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 1.5
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_model_cls.return_value = mock_model

        import io
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true"},
        )
        body = resp.text
        self.assertIn('"text": "hello world"', body)
        self.assertIn("[DONE]", body)

    @patch("whisper_live.server.WhisperModel")
    def test_stream_multiple_segments(self, mock_model_cls):
        segs = []
        for i in range(3):
            s = MagicMock()
            s.id = i
            s.start = float(i)
            s.end = float(i + 1)
            s.text = f" segment {i} "
            s.words = []
            segs.append(s)

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 3.0
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segs), mock_info)
        mock_model_cls.return_value = mock_model

        import io
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true"},
        )
        body = resp.text
        events = [line for line in body.split("\n") if line.startswith("data: ") and "[DONE]" not in line and '"type": "metadata"' not in line]
        self.assertEqual(len(events), 3)
        for i, event in enumerate(events):
            data = json.loads(event.removeprefix("data: "))
            self.assertEqual(data["text"], f"segment {i}")

    @patch("whisper_live.server.WhisperModel", side_effect=RuntimeError("model error"))
    def test_stream_error_yields_error_event(self, mock_model_cls):
        import io
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true"},
        )
        body = resp.text
        self.assertIn('"error"', body)
        self.assertIn("model error", body)

    def test_non_stream_still_works(self):
        import io
        from fastapi.testclient import TestClient

        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "false"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["text"], "non-streamed")


class TestRESTAPIParamWarnings(unittest.TestCase):
    """Test that unsupported OpenAI-compatible REST params produce warnings."""

    @classmethod
    def setUpClass(cls):
        """Build a FastAPI test app by extracting the endpoint definition."""
        import logging
        from fastapi import FastAPI, UploadFile, Form, File
        from fastapi.testclient import TestClient
        from typing import Optional, List

        app = FastAPI()

        @app.post("/v1/audio/transcriptions")
        async def transcribe(
            file: UploadFile,
            model: str = Form(default="whisper-1"),
            language: Optional[str] = Form(default=None),
            prompt: Optional[str] = Form(default=None),
            response_format: str = Form(default="json"),
            temperature: float = Form(default=0.0),
            timestamp_granularities: Optional[List[str]] = Form(default=None),
            chunking_strategy: Optional[str] = Form(default=None),
            include: Optional[List[str]] = Form(default=None),
            known_speaker_names: Optional[List[str]] = Form(default=None),
            known_speaker_references: Optional[List[UploadFile]] = File(default=None),
            stream: bool = Form(default=False),
        ):
            ignored_params = []
            if chunking_strategy:
                ignored_params.append(f"chunking_strategy='{chunking_strategy}'")
            if include:
                ignored_params.append(f"include={include}")
            if ignored_params:
                logging.warning(f"Unsupported OpenAI params ignored: {', '.join(ignored_params)}")
            # Return a JSON response with the ignored list for testing
            return {"text": "test", "ignored": ignored_params}

        cls.test_client = TestClient(app)

    def _post(self, **extra_fields):
        import io
        data = {**extra_fields}
        files = {"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")}
        return self.test_client.post("/v1/audio/transcriptions", data=data, files=files)

    def test_no_warnings_when_no_extra_params(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["ignored"], [])

    def test_chunking_strategy_warning(self):
        resp = self._post(chunking_strategy="auto")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertTrue(any("chunking_strategy" in p for p in ignored))

    def test_include_warning(self):
        resp = self._post(include="logprobs")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertTrue(any("include" in p for p in ignored))

    def test_known_speaker_names_supported(self):
        resp = self._post(known_speaker_names="alice")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertFalse(any("known_speaker_names" in p for p in ignored))

    def test_multiple_ignored_params(self):
        resp = self._post(chunking_strategy="auto", known_speaker_names="bob")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertEqual(len(ignored), 1)


class TestAPIKeyAuth(unittest.TestCase):
    """Test optional API key authentication middleware."""

    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse as JSONR

        app = FastAPI()

        @app.middleware("http")
        async def _check_api_key(request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth != "Bearer test-secret":
                return JSONR({"error": "Invalid or missing API key"}, status_code=401)
            return await call_next(request)

        @app.get("/ping")
        async def ping():
            return {"status": "ok"}

        cls.test_client = TestClient(app)

    def test_missing_key_returns_401(self):
        resp = self.test_client.get("/ping")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_key_returns_401(self):
        resp = self.test_client.get("/ping", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp.status_code, 401)

    def test_correct_key_returns_200(self):
        resp = self.test_client.get("/ping", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")


class TestRateLimiting(unittest.TestCase):
    """Test per-IP rate limiting middleware."""

    def _make_app(self, rpm_limit=3):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse as JSONR

        _rate_lock = threading.Lock()
        _rate_buckets: dict = {}

        app = FastAPI()

        @app.middleware("http")
        async def _rate_limit(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            with _rate_lock:
                bucket = _rate_buckets.setdefault(client_ip, collections.deque())
                while bucket and bucket[0] < now - 60:
                    bucket.popleft()
                if len(bucket) >= rpm_limit:
                    return JSONR({"error": "Rate limit exceeded"}, status_code=429)
                bucket.append(now)
            return await call_next(request)

        @app.get("/ping")
        async def ping():
            return {"status": "ok"}

        return TestClient(app)

    def test_within_limit_succeeds(self):
        client = self._make_app(rpm_limit=3)
        for _ in range(3):
            resp = client.get("/ping")
            self.assertEqual(resp.status_code, 200)

    def test_exceeding_limit_returns_429(self):
        client = self._make_app(rpm_limit=3)
        for _ in range(3):
            client.get("/ping")
        resp = client.get("/ping")
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Rate limit", resp.json()["error"])


class TestWebSocketAuth(unittest.TestCase):
    """Tests for the WebSocket process_request auth callback."""

    def _make_auth_handler(self, api_key):
        """Build the same auth function the server creates."""
        def _ws_auth(path, request_headers):
            connection = MagicMock()
            connection.respond.return_value = (401, [], b"Unauthorized\n")
            request = Request(path, request_headers)
            result = _websocket_auth(api_key, connection, request)
            if result is not None:
                connection.respond.assert_called_once_with(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
            return result
        return _ws_auth

    def test_valid_bearer_token(self):
        handler = self._make_auth_handler("my-secret")
        result = handler("/", {"Authorization": "Bearer my-secret"})
        self.assertIsNone(result)

    def test_invalid_bearer_token(self):
        handler = self._make_auth_handler("my-secret")
        result = handler("/", {"Authorization": "Bearer wrong"})
        self.assertEqual(result[0], 401)

    def test_missing_auth_header(self):
        handler = self._make_auth_handler("my-secret")
        result = handler("/", {})
        self.assertEqual(result[0], 401)

    def test_valid_query_token(self):
        handler = self._make_auth_handler("my-secret")
        result = handler("/?token=my-secret", {})
        self.assertIsNone(result)

    def test_invalid_query_token(self):
        handler = self._make_auth_handler("my-secret")
        result = handler("/?token=wrong", {})
        self.assertEqual(result[0], 401)


def _make_transcription_info(language="en", language_probability=0.97, duration=1.0):
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


def _make_segment(index=0, text=" hello ", start=0.0, end=1.0):
    seg = MagicMock()
    seg.id = index
    seg.seek = 0
    seg.start = start
    seg.end = end
    seg.text = text
    seg.tokens = []
    seg.temperature = 0.0
    seg.avg_logprob = -0.1
    seg.compression_ratio = 1.2
    seg.no_speech_prob = 0.01
    seg.words = []
    return seg


class TestHealthEndpoint(unittest.TestCase):
    """Tests for the REST /health route built by build_rest_app()."""

    def _make_client(self, api_key=None, **kwargs):
        from fastapi.testclient import TestClient

        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=3, max_connection_time=60)
        app = self.server.build_rest_app("faster_whisper", api_key=api_key, **kwargs)
        return TestClient(app)

    def test_health_reports_backend_model_and_clients(self):
        client = self._make_client()
        self.server.client_manager.add_client(MagicMock(), MagicMock())
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {
                "status": "ok",
                "backend": "faster_whisper",
                "model": "small",
                "clients": 1,
                "max_clients": 3,
            },
        )

    def test_health_reports_default_model_override(self):
        from fastapi.testclient import TestClient

        server = TranscriptionServer()
        server.default_model = "large-v3"
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)
        client = TestClient(server.build_rest_app("faster_whisper"))
        self.assertEqual(client.get("/health").json()["model"], "large-v3")

    def test_health_reports_custom_model_path(self):
        client = self._make_client(faster_whisper_custom_model_path="org/custom-model")
        self.assertEqual(client.get("/health").json()["model"], "org/custom-model")

    def test_health_needs_no_api_key(self):
        client = self._make_client(api_key="test-secret")
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_health_works_with_api_key_header(self):
        client = self._make_client(api_key="test-secret")
        resp = client.get("/health", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(resp.status_code, 200)

    def test_docs_routes_need_no_api_key(self):
        client = self._make_client(api_key="test-secret")
        for path in ("/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)

    def test_transcriptions_still_requires_api_key(self):
        import io

        client = self._make_client(api_key="test-secret")
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 401)


class TestRestModelSelection(unittest.TestCase):
    """Tests for _resolve_rest_model() and the REST model cache."""

    def setUp(self):
        self.server = TranscriptionServer()

    def test_whisper_1_alias_uses_default(self):
        self.assertEqual(self.server._resolve_rest_model("whisper-1"), "small")

    def test_missing_model_uses_default(self):
        self.assertEqual(self.server._resolve_rest_model(None), "small")
        self.assertEqual(self.server._resolve_rest_model(""), "small")

    def test_default_model_is_configurable(self):
        self.server.default_model = "medium"
        self.assertEqual(self.server._resolve_rest_model("whisper-1"), "medium")

    def test_stock_sizes_are_honoured(self):
        for name in ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"):
            with self.subTest(name=name):
                self.assertEqual(self.server._resolve_rest_model(name), name)

    def test_english_only_variants_are_honoured(self):
        for name in ("tiny.en", "base.en", "small.en", "medium.en"):
            with self.subTest(name=name):
                self.assertEqual(self.server._resolve_rest_model(name), name)

    def test_hf_repo_id_is_honoured(self):
        self.assertEqual(self.server._resolve_rest_model("deepdml/faster-whisper-x"), "deepdml/faster-whisper-x")

    def test_local_path_is_honoured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as model_dir:
            self.assertEqual(self.server._resolve_rest_model(model_dir), model_dir)

    def test_unknown_name_falls_back_to_default(self):
        self.server.default_model = "base"
        self.assertEqual(self.server._resolve_rest_model("gpt-4o-transcribe"), "base")

    @patch("whisper_live.server.WhisperModel")
    def test_model_is_loaded_once_per_name(self, mock_model_cls):
        first = self.server._get_rest_model("small")
        second = self.server._get_rest_model("small")
        self.assertIs(first, second)
        mock_model_cls.assert_called_once()

    @patch("whisper_live.server.WhisperModel")
    def test_different_names_load_separate_models(self, mock_model_cls):
        mock_model_cls.side_effect = [MagicMock(name="small"), MagicMock(name="medium")]
        small = self.server._get_rest_model("small")
        medium = self.server._get_rest_model("medium")
        self.assertIsNot(small, medium)
        self.assertEqual(mock_model_cls.call_count, 2)
        self.assertEqual(set(self.server.rest_models), {"small", "medium"})

    @patch("whisper_live.server.serve")
    def test_run_threads_default_model(self, mock_serve):
        server = TranscriptionServer()
        server.run("0.0.0.0", backend="faster_whisper", default_model="large-v3")
        self.assertEqual(server.default_model, "large-v3")

    @patch("whisper_live.server.serve")
    def test_custom_model_path_outranks_default_model(self, mock_serve):
        server = TranscriptionServer()
        server.run(
            "0.0.0.0",
            backend="faster_whisper",
            faster_whisper_custom_model_path="org/custom-model",
            default_model="large-v3",
        )
        self.assertEqual(server.default_model, "org/custom-model")


class TestRestTranscribeModelAndLanguage(unittest.TestCase):
    """Tests that the REST endpoint honours `model` and reports language."""

    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=2, max_connection_time=60)

    def _post(self, mock_model_cls, **fields):
        import io
        from fastapi.testclient import TestClient

        transcriber = MagicMock()
        transcriber.transcribe.return_value = (iter([_make_segment()]), _make_transcription_info())
        mock_model_cls.return_value = transcriber
        self.transcriber = transcriber

        client = TestClient(self.server.build_rest_app("faster_whisper"))
        return client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data=fields,
        )

    @patch("whisper_live.server.WhisperModel")
    def test_stock_model_name_is_loaded(self, mock_model_cls):
        resp = self._post(mock_model_cls, model="medium")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_model_cls.call_args[0][0], "medium")

    @patch("whisper_live.server.WhisperModel")
    def test_whisper_1_loads_default_model(self, mock_model_cls):
        self.server.default_model = "base"
        resp = self._post(mock_model_cls, model="whisper-1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_model_cls.call_args[0][0], "base")

    @patch("whisper_live.server.WhisperModel")
    def test_repeated_requests_reuse_cached_model(self, mock_model_cls):
        import io
        from fastapi.testclient import TestClient

        transcriber = MagicMock()
        transcriber.transcribe.side_effect = lambda *a, **k: (
            iter([_make_segment()]), _make_transcription_info()
        )
        mock_model_cls.return_value = transcriber

        client = TestClient(self.server.build_rest_app("faster_whisper"))
        for _ in range(3):
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
                data={"model": "small"},
            )
            self.assertEqual(resp.status_code, 200)
        mock_model_cls.assert_called_once()

    @patch("whisper_live.server.WhisperModel")
    def test_json_response_includes_language(self, mock_model_cls):
        resp = self._post(mock_model_cls, response_format="json")
        body = resp.json()
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["language"], "en")
        self.assertAlmostEqual(body["language_probability"], 0.97)

    @patch("whisper_live.server.WhisperModel")
    def test_verbose_json_includes_language_probability(self, mock_model_cls):
        resp = self._post(mock_model_cls, response_format="verbose_json")
        body = resp.json()
        self.assertEqual(body["language"], "en")
        self.assertAlmostEqual(body["language_probability"], 0.97)


class TestStreamLanguageEvent(unittest.TestCase):
    """The SSE stream must announce the detected language before any segment."""

    def _stream(self, mock_model_cls, language_probability=0.91):
        import io
        from fastapi.testclient import TestClient

        transcriber = MagicMock()
        transcriber.transcribe.return_value = (
            iter([_make_segment(text=" hello world ")]),
            _make_transcription_info(language="de", language_probability=language_probability),
        )
        mock_model_cls.return_value = transcriber

        server = TranscriptionServer()
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)
        client = TestClient(server.build_rest_app("faster_whisper"))
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true"},
        )
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.split("\n")
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        return events

    @patch("whisper_live.server.WhisperModel")
    def test_first_event_carries_language(self, mock_model_cls):
        events = self._stream(mock_model_cls)
        self.assertEqual(events[0]["language"], "de")
        self.assertAlmostEqual(events[0]["language_probability"], 0.91)

    @patch("whisper_live.server.WhisperModel")
    def test_language_event_precedes_segments(self, mock_model_cls):
        events = self._stream(mock_model_cls)
        self.assertNotIn("text", events[0])
        self.assertEqual(events[1]["text"], "hello world")

    @patch("whisper_live.server.WhisperModel")
    def test_stream_honours_model_field(self, mock_model_cls):
        import io
        from fastapi.testclient import TestClient

        transcriber = MagicMock()
        transcriber.transcribe.return_value = (iter([]), _make_transcription_info())
        mock_model_cls.return_value = transcriber

        server = TranscriptionServer()
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)
        client = TestClient(server.build_rest_app("faster_whisper"))
        client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"\x00" * 100), "audio/wav")},
            data={"stream": "true", "model": "large-v3"},
        )
        self.assertEqual(mock_model_cls.call_args[0][0], "large-v3")


class TestAudioPreprocessor(unittest.TestCase):
    """Tests for the optional audio_preprocessor hook."""

    def setUp(self):
        self.server = TranscriptionServer()
        self.server.backend = BackendType.FASTER_WHISPER
        self.server.client_manager = ClientManager(max_clients=2, max_connection_time=60)

    def _add_client(self, preprocessor):
        import numpy as np

        ws = MagicMock()
        ws.recv.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        client = MagicMock()
        client.audio_preprocessor = preprocessor
        self.server.client_manager.add_client(ws, client)
        return ws, client

    def test_preprocessor_output_reaches_add_frames(self):
        import numpy as np

        gained = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ws, client = self._add_client(lambda frame, rate: gained)
        self.assertTrue(self.server.process_audio_frames(ws))
        np.testing.assert_array_equal(client.add_frames.call_args[0][0], gained)

    def test_preprocessor_receives_frame_and_sample_rate(self):
        import numpy as np

        seen = {}

        def preprocessor(frame, sample_rate):
            seen["frame"] = frame
            seen["sample_rate"] = sample_rate
            return frame

        ws, _ = self._add_client(preprocessor)
        self.server.process_audio_frames(ws)
        self.assertEqual(seen["sample_rate"], TranscriptionServer.RATE)
        np.testing.assert_allclose(seen["frame"], [0.1, 0.2, 0.3], rtol=1e-6)

    def test_no_preprocessor_leaves_frame_untouched(self):
        import numpy as np

        ws, client = self._add_client(None)
        self.server.process_audio_frames(ws)
        np.testing.assert_allclose(client.add_frames.call_args[0][0], [0.1, 0.2, 0.3], rtol=1e-6)

    def test_failing_preprocessor_does_not_break_the_stream(self):
        import numpy as np

        def preprocessor(frame, sample_rate):
            raise RuntimeError("denoiser exploded")

        ws, client = self._add_client(preprocessor)
        self.assertTrue(self.server.process_audio_frames(ws))
        np.testing.assert_allclose(client.add_frames.call_args[0][0], [0.1, 0.2, 0.3], rtol=1e-6)

    def test_preprocessor_runs_before_vad_for_tensorrt(self):
        import numpy as np

        self.server.backend = BackendType.TENSORRT
        self.server.use_vad = True
        denoised = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        ws, client = self._add_client(lambda frame, rate: denoised)
        self.server.vad_detector = MagicMock(return_value=True)
        self.server.process_audio_frames(ws)
        np.testing.assert_array_equal(self.server.vad_detector.call_args[0][0], denoised)

    @patch("whisper_live.server.serve")
    def test_run_stores_preprocessor(self, mock_serve):
        def preprocessor(frame, sample_rate):
            return frame

        server = TranscriptionServer()
        server.run("0.0.0.0", backend="faster_whisper", audio_preprocessor=preprocessor)
        self.assertIs(server.audio_preprocessor, preprocessor)

    def test_initialize_client_attaches_preprocessor(self):
        def preprocessor(frame, sample_rate):
            return frame

        server = TranscriptionServer()
        server.backend = BackendType.FASTER_WHISPER
        server.cache_path = "~/.cache/whisper-live/"
        server.client_manager = ClientManager(max_clients=2, max_connection_time=60)
        ws = MagicMock()
        options = {"uid": "abc", "language": "en", "task": "transcribe", "model": "small"}
        with patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper") as mock_backend:
            server.initialize_client(ws, options, None, None, False, audio_preprocessor=preprocessor)
        client = server.client_manager.get_client(ws)
        self.assertIs(client.audio_preprocessor, preprocessor)


class TestKnownSpeakersOverWebSocket(unittest.TestCase):
    """Tests for known_speakers enrollment sent in the WebSocket handshake."""

    def setUp(self):
        self.server = TranscriptionServer()
        self.diarizer = MagicMock()
        self.diarizer.enroll_speaker.return_value = True

    def _options(self, **extra):
        import base64

        options = {
            "uid": "abc",
            "known_speakers": [
                {"name": "alice", "audio_base64": base64.b64encode(b"RIFFfake").decode()},
            ],
        }
        options.update(extra)
        return options

    def _create_diarizer(self, options):
        import numpy as np

        with patch("whisper_live.diarization.SpeakerDiarizer", return_value=self.diarizer), \
                patch("whisper_live.diarization.load_audio", return_value=np.zeros(16000, dtype="float32")):
            return self.server._create_diarizer(options)

    def test_known_speakers_alone_create_a_diarizer(self):
        diarizer = self._create_diarizer(self._options())
        self.assertIs(diarizer, self.diarizer)

    def test_known_speakers_are_enrolled(self):
        self._create_diarizer(self._options())
        self.diarizer.enroll_speaker.assert_called_once()
        self.assertEqual(self.diarizer.enroll_speaker.call_args[0][0], "alice")

    def test_multiple_known_speakers_enrolled(self):
        import base64

        speakers = [
            {"name": "alice", "audio_base64": base64.b64encode(b"RIFFa").decode()},
            {"name": "bob", "audio_base64": base64.b64encode(b"RIFFb").decode()},
        ]
        self._create_diarizer({"uid": "abc", "known_speakers": speakers})
        enrolled = [call[0][0] for call in self.diarizer.enroll_speaker.call_args_list]
        self.assertEqual(enrolled, ["alice", "bob"])

    def test_no_known_speakers_and_no_diarization_returns_none(self):
        self.assertIsNone(self.server._create_diarizer({"uid": "abc"}))

    def test_diarization_without_known_speakers_enrolls_nothing(self):
        diarizer = self._create_diarizer({"uid": "abc", "enable_diarization": True})
        self.assertIs(diarizer, self.diarizer)
        self.diarizer.enroll_speaker.assert_not_called()

    def test_speaker_without_audio_is_skipped(self):
        self._create_diarizer({"uid": "abc", "known_speakers": [{"name": "alice"}]})
        self.diarizer.enroll_speaker.assert_not_called()

    def test_invalid_base64_does_not_raise(self):
        import numpy as np

        with patch("whisper_live.diarization.load_audio", return_value=np.zeros(16000, dtype="float32")):
            TranscriptionServer._enroll_known_speakers(
                self.diarizer,
                [{"name": "alice", "audio_base64": "not base64 @@@"}],
            )
        self.diarizer.enroll_speaker.assert_not_called()

    def test_too_short_reference_is_reported_not_raised(self):
        import base64
        import numpy as np

        self.diarizer.enroll_speaker.return_value = False
        with patch("whisper_live.diarization.load_audio", return_value=np.zeros(10, dtype="float32")):
            TranscriptionServer._enroll_known_speakers(
                self.diarizer,
                [{"name": "alice", "audio_base64": base64.b64encode(b"RIFFa").decode()}],
            )
        self.diarizer.enroll_speaker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
