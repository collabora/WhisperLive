import json
import os
import scipy
import websocket
import unittest
from unittest.mock import patch, MagicMock
from whisper_live.client import TranscriptionClient, resample


class TestClientWebSocketCommunication(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    def test_websocket_communication(self, mock_websocket):
        mock_ws_instance = MagicMock()
        mock_websocket.return_value = mock_ws_instance
        expected_url = 'ws://localhost:9090'

        client = TranscriptionClient(host='localhost', port=9090).client

        mock_websocket.assert_called()
        self.assertEqual(mock_websocket.call_args[0][0], expected_url)

        client.close_websocket()


class TestClientCallbacks(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    def setUp(self, mock_websocket):
        self.mock_ws_app = mock_websocket.return_value
        self.mock_ws_app.send = MagicMock()
        self.client = TranscriptionClient(host='localhost', port=9090, lang="en").client
    
    def tearDown(self):
        self.client.close_websocket()
        del self.client

    def test_on_open(self):
        expected_message = json.dumps({
            "uid": self.client.uid,
            "language": self.client.language,
            "task": self.client.task,
            "model": self.client.model,
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


class TestSendingAudioPacket(unittest.TestCase):
    @patch('whisper_live.client.websocket.WebSocketApp')
    def setUp(self, mock_websocket):
        self.transcription_client = TranscriptionClient("localhost", "9090")
        self.client = self.transcription_client.client
        self.client.client_socket = mock_websocket.return_value
    
    def tearDown(self):
        self.client.close_websocket()
        del self.client

    def test_send_packet(self):
        mock_audio_packet = b'\x00\x01\x02\x03'
        self.client.send_packet_to_server(mock_audio_packet)
        self.client.client_socket.send.assert_called_with(mock_audio_packet, websocket.ABNF.OPCODE_BINARY)
