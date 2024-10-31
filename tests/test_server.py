import subprocess
import time
import json
import unittest
from unittest import mock

import numpy as np
import jiwer

from websockets.exceptions import ConnectionClosed
from whisper_live.server import TranscriptionServer, BackendType, ClientManager
from whisper_live.client import Client, TranscriptionClient, TranscriptionTeeClient
from whisper.normalizers import EnglishTextNormalizer


class TestTranscriptionServerInitialization(unittest.TestCase):
    def test_initialization(self):
        server = TranscriptionServer()
        server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        self.assertEqual(server.client_manager.max_clients, 4)
        self.assertEqual(server.client_manager.max_connection_time, 600)
        self.assertDictEqual(server.client_manager.clients, {})
        self.assertDictEqual(server.client_manager.start_times, {})


class TestGetWaitTime(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        self.server.client_manager.start_times = {
            'client1': time.time() - 120,
            'client2': time.time() - 300
        }
        self.server.client_manager.max_connection_time = 600

    def test_get_wait_time(self):
        expected_wait_time = (600 - (time.time() - self.server.client_manager.start_times['client2'])) / 60
        print(self.server.client_manager.get_wait_time(), expected_wait_time)
        self.assertAlmostEqual(self.server.client_manager.get_wait_time(), expected_wait_time, places=2)


class TestServerConnection(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()

    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_connection(self, mock_websocket):
        mock_websocket.recv.return_value = json.dumps({
            'uid': 'test_client',
            'language': 'en',
            'task': 'transcribe',
            'model': 'tiny.en'
        })
        self.server.recv_audio(mock_websocket, BackendType("faster_whisper"))

    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_recv_audio_exception_handling(self, mock_websocket):
        mock_websocket.recv.side_effect = [json.dumps({
            'uid': 'test_client',
            'language': 'en',
            'task': 'transcribe',
            'model': 'tiny.en'
        }),  np.array([1, 2, 3]).tobytes()]

        with self.assertLogs(level="ERROR"):
            self.server.recv_audio(mock_websocket, BackendType("faster_whisper"))

        self.assertNotIn(mock_websocket, self.server.client_manager.clients)


class TestServerInferenceAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_pyaudio_patch = mock.patch('pyaudio.PyAudio')
        cls.mock_pyaudio = cls.mock_pyaudio_patch.start()
        cls.mock_pyaudio.return_value.open.return_value = mock.MagicMock()
        
        cls.server_process = subprocess.Popen(["python", "run_server.py"])
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        cls.server_process.terminate()
        cls.server_process.wait()

    def setUp(self):
        self.normalizer = EnglishTextNormalizer()

    def check_prediction(self, srt_path):
        gt = "And so my fellow Americans, ask not, what your country can do for you. Ask what you can do for your country!"
        with open(srt_path, "r") as f:
            lines = f.readlines()
            prediction = " ".join([line.strip() for line in lines[2::4]])
        prediction_normalized = self.normalizer(prediction)
        gt_normalized = self.normalizer(gt)

        # calculate WER
        wer_score = jiwer.wer(gt_normalized, prediction_normalized)
        self.assertLess(wer_score, 0.05)

    def test_inference(self):
        client = TranscriptionClient(
            "localhost", "9090", model="base.en", lang="en",
        )
        client("assets/jfk.flac")
        self.check_prediction("output.srt")

    def test_simultaneous_inference(self):
        client1 = Client(
            "localhost", "9090", model="base.en", lang="en", srt_file_path="transcript1.srt")
        client2 = Client(
            "localhost", "9090", model="base.en", lang="en", srt_file_path="transcript2.srt")
        tee = TranscriptionTeeClient([client1, client2])
        tee("assets/jfk.flac")
        self.check_prediction("transcript1.srt")
        self.check_prediction("transcript2.srt")


class TestExceptionHandling(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()

    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_connection_closed_exception(self, mock_websocket):
        mock_websocket.recv.side_effect = ConnectionClosed(1001, "testing connection closed", rcvd_then_sent=mock.Mock())

        with self.assertLogs(level="INFO") as log:
            self.server.recv_audio(mock_websocket, BackendType("faster_whisper"))
            self.assertTrue(any("Connection closed by client" in message for message in log.output))

    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_json_decode_exception(self, mock_websocket):
        mock_websocket.recv.return_value = "invalid json"

        with self.assertLogs(level="ERROR") as log:
            self.server.recv_audio(mock_websocket, BackendType("faster_whisper"))
            self.assertTrue(any("Failed to decode JSON from client" in message for message in log.output))

    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_unexpected_exception_handling(self, mock_websocket):
        mock_websocket.recv.side_effect = RuntimeError("Unexpected error")

        with self.assertLogs(level="ERROR") as log:
            self.server.recv_audio(mock_websocket, BackendType("faster_whisper"))
            for message in log.output:
                print(message)
            print()
            self.assertTrue(any("Unexpected error" in message for message in log.output))
