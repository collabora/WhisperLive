import json
import time
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


if __name__ == "__main__":
    unittest.main()
