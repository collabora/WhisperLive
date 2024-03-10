import json
import os
import scipy
import websocket
import copy
import unittest
from unittest.mock import patch, MagicMock
from whisper_live.client import Client, TranscriptionClient, TranscriptionTeeClient
from whisper_live.utils import resample
from pathlib import Path


class BaseTestCase(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    @patch('whisper_live.client.pyaudio.PyAudio')
    def setUp(self, mock_pyaudio, mock_websocket):
        self.mock_pyaudio_instance = MagicMock()
        mock_pyaudio.return_value = self.mock_pyaudio_instance
        self.mock_stream = MagicMock()
        self.mock_pyaudio_instance.open.return_value = self.mock_stream

        self.mock_ws_app = mock_websocket.return_value
        self.mock_ws_app.send = MagicMock()

        self.client = TranscriptionClient(host='localhost', port=9090, lang="en").client

        self.mock_pyaudio = mock_pyaudio
        self.mock_websocket = mock_websocket
        self.mock_audio_packet = b'\x00\x01\x02\x03'

    def tearDown(self):
        self.client.close_websocket()
        self.mock_pyaudio.stop()
        self.mock_websocket.stop()
        del self.client

class TestClientWebSocketCommunication(BaseTestCase):
    def test_websocket_communication(self):
        expected_url = 'ws://localhost:9090'
        self.mock_websocket.assert_called()
        self.assertEqual(self.mock_websocket.call_args[0][0], expected_url)


class TestClientCallbacks(BaseTestCase):
    def test_on_open(self):
        expected_message = json.dumps({
            "uid": self.client.uid,
            "language": self.client.language,
            "task": self.client.task,
            "model": self.client.model,
            "use_vad": True
        })
        self.client.on_open(self.mock_ws_app)
        self.mock_ws_app.send.assert_called_with(expected_message)

    def test_on_message(self):
        message = json.dumps(
            {
                "uid": self.client.uid,
                "message": "SERVER_READY",
                "backend": "faster_whisper"
            }
        )
        self.client.on_message(self.mock_ws_app, message)

        message = json.dumps({
            "uid": self.client.uid,
            "segments": [
                {"start": 0, "end": 1, "text": "Test transcript"},
                {"start": 1, "end": 2, "text": "Test transcript 2"},
                {"start": 2, "end": 3, "text": "Test transcript 3"}
            ]
        })
        self.client.on_message(self.mock_ws_app, message)

        # Assert that the transcript was updated correctly
        self.assertEqual(len(self.client.transcript), 2)
        self.assertEqual(self.client.transcript[1]['text'], "Test transcript 2")

    def test_on_close(self):
        close_status_code = 1000
        close_msg = "Normal closure"
        self.client.on_close(self.mock_ws_app, close_status_code, close_msg)

        self.assertFalse(self.client.recording)
        self.assertFalse(self.client.server_error)
        self.assertFalse(self.client.waiting)

    def test_on_error(self):
        error_message = "Test Error"
        self.client.on_error(self.mock_ws_app, error_message)

        self.assertTrue(self.client.server_error)
        self.assertEqual(self.client.error_message, error_message)


class TestAudioResampling(unittest.TestCase):
    def test_resample_audio(self):
        original_audio = "assets/jfk.flac"
        expected_sr = 16000
        resampled_audio = resample(original_audio, expected_sr)

        sr, _ = scipy.io.wavfile.read(resampled_audio)
        self.assertEqual(sr, expected_sr)

        os.remove(resampled_audio)


class TestSendingAudioPacket(BaseTestCase):
    def test_send_packet(self):
        self.client.send_packet_to_server(self.mock_audio_packet)
        self.client.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

class TestTee(BaseTestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    @patch('whisper_live.client.pyaudio.PyAudio')
    def setUp(self, mock_audio, mock_websocket):
        super().setUp()
        self.client2 = Client(host='localhost', port=9090, lang="es", translate=False, srt_file_path="transcript.srt")
        self.client3 = Client(host='localhost', port=9090, lang="es", translate=True, srt_file_path="translation.srt")
        # need a separate mock for each websocket
        self.client3.client_socket = copy.deepcopy(self.client3.client_socket)
        self.tee = TranscriptionTeeClient([self.client2, self.client3])

    def tearDown(self):
        self.tee.close_all_clients()
        del self.tee
        super().tearDown()

    def test_invalid_constructor(self):
        with self.assertRaises(Exception) as context:
            TranscriptionTeeClient([])

    def test_multicast_unconditional(self):
        self.tee.multicast_packet(self.mock_audio_packet, True)
        for client in self.tee.clients:
            client.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

    def test_multicast_conditional(self):
        self.client2.recording = False
        self.client3.recording = True
        self.tee.multicast_packet(self.mock_audio_packet, False)
        self.client2.client_socket.send.assert_not_called()
        self.client3.client_socket.send.assert_called_with(self.mock_audio_packet, websocket.ABNF.OPCODE_BINARY)

    def test_close_all(self):
        self.tee.close_all_clients()
        for client in self.tee.clients:
            client.client_socket.close.assert_called()

    def test_write_all_srt(self):
        for client in self.tee.clients:
            client.server_backend = "faster_whisper"
        self.tee.write_all_clients_srt()
        self.assertTrue(Path("transcript.srt").is_file())
        self.assertTrue(Path("translation.srt").is_file())
