import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
import numpy as np
import torch.nn.functional as F
from whisper.tokenizer import get_tokenizer
from whisper_live.tensorrt_utils import (mel_filters, load_audio_wav_format, pad_or_trim, load_audio)

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / 'encoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        dtype = config['builder_config']['precision']
        n_mels = config['builder_config']['n_mels']
        num_languages = config['builder_config']['num_languages']

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f'whisper_encoder_{self.dtype}_tp1_rank0.engine'

        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)

        inputs = OrderedDict()
        inputs['x'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features


class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_config(self, engine_dir):
        config_path = engine_dir / 'decoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config['plugin_config'])
        decoder_config.update(config['builder_config'])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        dtype = self.decoder_config['precision']
        serialize_path = engine_dir / f'whisper_decoder_{dtype}_tp1_rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_heads'],
            num_kv_heads=self.decoder_config['num_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            num_layers=self.decoder_config['num_layers'],
            gpt_attention_plugin=self.decoder_config['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['remove_input_padding'],
            cross_attention=self.decoder_config['cross_attention'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            has_token_type_embedding=self.
            decoder_config['has_token_type_embedding'],
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 eot_id,
                 max_new_tokens=40,
                 num_beams=1):
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device='cuda')

        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [encoder_outputs.shape[0], 1,
             encoder_outputs.shape[1]]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(end_id=eot_id,
                                         pad_id=eot_id,
                                         num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1])

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):

    def __init__(self, engine_dir, assets_dir=None, device=None, is_multilingual=False,
                 language="en", task="transcribe"):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir,
                                       runtime_mapping,
                                       debug_mode=False)
        self.n_mels = self.encoder.n_mels
        # self.tokenizer = get_tokenizer(num_languages=self.encoder.num_languages,
        #                                tokenizer_dir=assets_dir)
        self.device = device
        self.tokenizer = get_tokenizer(
            is_multilingual,
            num_languages=self.encoder.num_languages,
            language=language,
            task=task,
        )
        self.filters = mel_filters(self.device, self.encoder.n_mels, assets_dir)

    def log_mel_spectrogram(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        padding: int = 0,
        return_duration=True
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 and 128 are supported

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (80 or 128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                if audio.endswith('.wav'):
                    audio, _ = load_audio_wav_format(audio)
                else:
                    audio = load_audio(audio)
            assert isinstance(audio, np.ndarray), f"Unsupported audio type: {type(audio)}"
            duration = audio.shape[-1] / SAMPLE_RATE
            audio = pad_or_trim(audio, N_SAMPLES)
            audio = audio.astype(np.float32)
            audio = torch.from_numpy(audio)

        if self.device is not None:
            audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self.filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if return_duration:
            return log_spec, duration
        else:
            return log_spec

    def process_batch(
            self,
            mel,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1):
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=set(self.tokenizer.special_tokens.keys()))

        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        encoder_output = self.encoder.get_audio_features(mel)
        output_ids = self.decoder.generate(decoder_input_ids,
                                           encoder_output,
                                           self.tokenizer.eot,
                                           max_new_tokens=96,
                                           num_beams=num_beams)
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts

    def transcribe(
            self,
            mel,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            dtype='float16',
            batch_size=1,
            num_beams=1,
            ):
        mel = mel.type(str_dtype_to_torch(dtype))
        mel = mel.unsqueeze(0)
        predictions = self.process_batch(mel, text_prefix, num_beams)
        prediction = predictions[0]

        # remove all special tokens in the prediction
        prediction = re.sub(r'<\|.*?\|>', '', prediction)
        return prediction.strip()


def decode_wav_file(
        model,
        mel,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype='float16',
        batch_size=1,
        num_beams=1,
        normalizer=None,
        mel_filters_dir=None):

    mel = mel.type(str_dtype_to_torch(dtype))
    mel = mel.unsqueeze(0)
    # repeat the mel spectrogram to match the batch size
    mel = mel.repeat(batch_size, 1, 1)
    predictions = model.process_batch(mel, text_prefix, num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)

    return prediction.strip()
