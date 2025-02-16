import logging
import requests
import asyncio
from googletrans import Translator
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logging.basicConfig(level=logging.INFO)

class TranslatorAPI:
    """
    Класс для перевода с использованием бесплатного API LibreTranslate.
    """
    def __init__(self, source_language: str, target_language: str):
        self.source_language = source_language
        self.target_language = target_language
        self.translator = Translator()

    def translate(self, text: str, target_language: str = None, source_language: str = None) -> str:
        if target_language is not None:
            self.target_language = target_language
        if source_language is not None:
            self.source_language = source_language
        try:
            result = self.translator.translate(text, src=self.source_language, dest=self.target_language)
            return result.text
        except Exception as e:
            logging.error(f"[GoogleTranslatorAPI] Ошибка перевода: {e}")
            return text


class TranslatorNN:
    """
    Класс для перевода с использованием быстрой нейронной модели из библиотеки transformers.
    """
    def __init__(self, source_language: str, target_language: str):
        if pipeline is None:
            raise ImportError("Не установлен пакет transformers")
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
        try:
            self.translator = pipeline("translation", model=model_name)
        except Exception as e:
            logging.error(f"[TranslatorNN] Ошибка загрузки модели {model_name}: {e}")
            raise

    def translate(self, text: str) -> str:
        try:
            result = self.translator(text)
            return result[0]["translation_text"]
        except Exception as e:
            logging.error(f"[TranslatorNN] Ошибка перевода: {e}")
            return text