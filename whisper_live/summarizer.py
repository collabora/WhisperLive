from transformers import pipeline
from whisper_live.translator import TranslatorAPI


class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", source_language="ru", target_language="ru"):
        # Инициализация пайплайна для саммаризации
        self.summarizer = pipeline("summarization", model=model_name)
        self.translator = TranslatorAPI("en", "en")
        self.target_language = target_language
        self.source_language = source_language

    def summarize(self, text, max_length=130, min_length=30):
        """
        Делает саммаризацию текста.

        :param text: Исходный текст для саммаризации.
        :param max_length: Максимальная длина итоговой саммаризированной версии.
        :param min_length: Минимальная длина итоговой саммаризированной версии.
        :return: Строка саммаризированного текста.
        """
        text = self.translator.translate(
            text, source_language=self.source_language, target_language="en")
        result = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False)
        text = self.translator.translate(
            result[0]['summary_text'], source_language="en",  target_language=self.target_language)
        return text


# Пример использования
if __name__ == "__main__":
    text = (
        """Мы форкнули решение с сервисом и расширением для браузера
		Мы хотим:

		1. Модифицировать сервис уменьшив нагрузку на него. Сейчас каждый пользователь требует на себя отдельную модель, мы хотим реорганизовать это чтобы использовать одну STT на конференцию
		2. Улучшить UI/UX расширения
		3. Добавить TTS. Сначала с помощью встроенных функций браузера, потом попробовать кастомное решение
		4. Расширить функционал перевода. Сейчас доступен перевод только на английский.
		5. К концу хакатона планируем предоставить сервисное решение развертываемое через докер и расширение для хрома"""
    )
    summarizer = TextSummarizer()
    summary = summarizer.summarize(text)
    print("Саммаризация:")
    print(summary)
