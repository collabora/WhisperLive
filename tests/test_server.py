import subprocess
import time
import json
import unittest
from unittest import mock

import numpy as np
import evaluate
from whisper_live.server import TranscriptionServer
from whisper_live.client import TranscriptionClient
from whisper.normalizers import EnglishTextNormalizer


class TestTranscriptionServerInitialization(unittest.TestCase):
    def test_initialization(self):
        server = TranscriptionServer()
        self.assertEqual(server.max_clients, 4)
        self.assertEqual(server.max_connection_time, 600)
        self.assertDictEqual(server.clients, {})
        self.assertDictEqual(server.websockets, {})
        self.assertDictEqual(server.clients_start_time, {})


class TestGetWaitTime(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.clients_start_time = {
            'client1': time.time() - 120,
            'client2': time.time() - 300
        }
        self.server.max_connection_time = 600

    def test_get_wait_time(self):
        expected_wait_time = (600 - (time.time() - self.server.clients_start_time['client2'])) / 60
        print(self.server.get_wait_time(), expected_wait_time)
        self.assertAlmostEqual(self.server.get_wait_time(), expected_wait_time, places=2)

    
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
        self.server.recv_audio(mock_websocket, "faster_whisper")

    
    @mock.patch('websockets.WebSocketCommonProtocol')
    def test_recv_audio_exception_handling(self, mock_websocket):
        mock_websocket.recv.side_effect = [json.dumps({
            'uid': 'test_client',
            'language': 'en',
            'task': 'transcribe',
            'model': 'tiny.en'
        }),  np.array([1, 2, 3]).tobytes()]  
        
        with self.assertLogs(level="ERROR"):
            self.server.recv_audio(mock_websocket, "faster_whisper")
        
        self.assertNotIn(mock_websocket, self.server.clients)


class TestServerInferenceAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_process = subprocess.Popen(["python", "run_server.py"])  # Adjust the command as needed
        time.sleep(2)
    
    @classmethod
    def tearDownClass(cls):
        cls.server_process.terminate()
        cls.server_process.wait()
    
    def setUp(self):
        self.metric = evaluate.load("wer")
        self.normalizer = EnglishTextNormalizer()
        self.client  = TranscriptionClient(
            "localhost", "9090", model="base.en", lang="en",
        )
    
    def test_inference(self):
        gt = "And so my fellow Americans, ask not, what your country can do for you. Ask what you can do for your country!"
        self.client("assets/jfk.flac")
        with open("output.srt", "r") as f:
            lines = f.readlines()
            prediction = " ".join([l.strip() for l in lines[2::4]])
        prediction_normalized = self.normalizer(prediction)
        gt_normalized = self.normalizer(gt)

        # calculate WER
        wer = self.metric.compute(
            predictions=[prediction_normalized],
            references=[gt_normalized]
        )
        self.assertLess(wer, 0.05)
