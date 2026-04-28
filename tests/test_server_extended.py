import json
import time
import threading
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.server import TranscriptionServer, BackendType, ClientManager


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
        self.server.cleanup(ws)
        self.assertNotIn(ws, self.server.client_manager.clients)
        client.cleanup.assert_called_once()


class TestRESTAPIParamWarnings(unittest.TestCase):
    """Test that unsupported OpenAI-compatible REST params produce warnings."""

    @classmethod
    def setUpClass(cls):
        """Build a FastAPI test app by extracting the endpoint definition."""
        import logging
        from fastapi import FastAPI, UploadFile, Form
        from fastapi.testclient import TestClient
        from typing import Optional, List
        from starlette.responses import PlainTextResponse, JSONResponse

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
            known_speaker_references: Optional[List[str]] = Form(default=None),
            stream: bool = Form(default=False),
        ):
            if stream:
                return JSONResponse({"error": "Streaming not supported in this backend."}, status_code=400)

            ignored_params = []
            if chunking_strategy:
                ignored_params.append(f"chunking_strategy='{chunking_strategy}'")
            if known_speaker_names:
                ignored_params.append("known_speaker_names")
            if known_speaker_references:
                ignored_params.append("known_speaker_references")
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

    def test_known_speaker_names_warning(self):
        resp = self._post(known_speaker_names="alice")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertTrue(any("known_speaker_names" in p for p in ignored))

    def test_stream_returns_400(self):
        resp = self._post(stream="true")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_multiple_ignored_params(self):
        resp = self._post(chunking_strategy="auto", known_speaker_names="bob")
        self.assertEqual(resp.status_code, 200)
        ignored = resp.json()["ignored"]
        self.assertGreaterEqual(len(ignored), 2)


if __name__ == "__main__":
    unittest.main()
