import json
import logging
import threading
import time
import queue
from typing import Dict, Any, Optional
import torch
import threading
from transformers import M2M100ForConditionalGeneration
from whisper_live.backend.tokenization_small100 import SMALL100Tokenizer

from whisper_live.backend.base import ServeClientBase


class ServeClientTranslation(ServeClientBase):
    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    
    def __init__(
        self,
        client_uid,
        websocket,
        translation_queue,
        target_language="fr", 
        send_last_n_segments=10,
        model_name="alirezamsh/small100"
    ):
        """
        Initialize the translation client.
        
        Args:
            client_uid (str): Unique identifier for the client
            websocket: WebSocket connection to the client
            translation_queue (queue.Queue): Queue containing completed segments to translate
            target_language (str): Target language code (default: "fr" for French)
            send_last_n_segments (int): Number of recent translated segments to send
            model_name (str): Translation model name to use
        """
        super().__init__(client_uid, websocket, send_last_n_segments)
        self.translation_queue = translation_queue
        self.target_language = target_language
        self.model_name = model_name
        self.translated_segments = []
        self.translation_model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.load_translation_model()
        
    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Loading translation model on device: {self.device}")
            
            self.translation_model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            self.tokenizer = SMALL100Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.tgt_lang = self.target_language
            
            self.model_loaded = True
            logging.info(f"Translation model loaded successfully. Target language: {self.target_language}")
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            self.translation_model = None
            self.tokenizer = None
            self.model_loaded = False
    
    def translate_text(self, text: str) -> str:
        """
        Translate a single text segment.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text or original text if translation fails
        """
        if not self.model_loaded or not text.strip():
            return text
            
        try:
            # Encode input and move to device
            encoded_input = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.translation_model.generate(**encoded_input)
            
            # Decode output
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return output[0] if output else text
            
        except Exception as e:
            logging.error(f"Translation failed for text '{text}': {e}")
            return text
    
    def process_translation_queue(self):
        """
        Process segments from the translation queue.
        Continuously reads from the queue until None is received (exit signal).
        """
        logging.info(f"Starting translation processing for client {self.client_uid}")
        
        while not self.exit:
            try:
                # Get segment from queue with timeout
                segment = self.translation_queue.get(timeout=1.0)
                
                # Check for exit signal
                if segment is None:
                    logging.info(f"Received exit signal for translation client {self.client_uid}")
                    break
                    
                # Only translate completed segments
                if not segment.get("completed", False):
                    self.translation_queue.task_done()
                    continue
                    
                # Translate the segment
                original_text = segment.get("text", "")
                translated_text = self.translate_text(original_text)
                
                # Create translated segment
                translated_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text,
                    "completed": segment.get("completed", False),
                    "target_language": self.target_language
                }
                
                self.translated_segments.append(translated_segment)
                segments_to_send = self.prepare_translated_segments()
                self.send_translation_to_client(segments_to_send)
                
                self.translation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")
                continue
        
        logging.info(f"Translation processing ended for client {self.client_uid}")
    
    def prepare_translated_segments(self):
        """
        Prepare the last n translated segments to send to client.
        
        Returns:
            list: List of recent translated segments
        """
        if len(self.translated_segments) >= self.send_last_n_segments:
            return self.translated_segments[-self.send_last_n_segments:]
        return self.translated_segments[:]
    
    def send_translation_to_client(self, translated_segments):
        """
        Send translated segments to the client via WebSocket.
        
        Args:
            translated_segments (list): List of translated segments to send
        """
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "translated_segments": translated_segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending translation data to client: {e}")
    
    def speech_to_text(self):
        """
        Override parent method to handle translation processing.
        This method will be called when the translation thread starts.
        """
        self.process_translation_queue()
    
    def set_target_language(self, language: str):
        """
        Change the target language for translation.
        
        Args:
            language (str): New target language code
        """
        self.target_language = language
        if self.tokenizer:
            self.tokenizer.tgt_lang = language
            logging.info(f"Target language changed to: {language}")
    
    def cleanup(self):
        """Clean up translation resources."""
        logging.info(f"Cleaning up translation resources for client {self.client_uid}")
        self.exit = True
        
        try:
            self.translation_queue.put(None, timeout=1.0)
        except:
            pass
        
        self.translated_segments.clear()
        
        if self.translation_model:
            del self.translation_model
            self.translation_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
