import librosa
import os

import openvino_genai as ov_genai
import huggingface_hub as hf_hub


class WhisperOpenVINO(object):
    def __init__(self, model_id="OpenVINO/whisper-tiny-fp16-ov", device="CPU", language="en", task="transcribe"):
        model_path = model_id.split('/')[-1]
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "openvino_whisper_models")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, model_path)
        if not os.path.exists(model_path):
            hf_hub.snapshot_download(model_id, local_dir=model_path)
        self.model = ov_genai.WhisperPipeline(str(model_path), device=device)
        self.language = language
        self.task = task

    def transcribe(self, input_audio):
        outputs = self.model.generate(input_audio, return_timestamps=True, language=self.language, task=self.task)
        outputs = [seg for seg in outputs.chunks]
        return outputs
