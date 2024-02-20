# WhisperLive

<h2 align="center">
  <a href="https://www.youtube.com/watch?v=0PHWCApIcCI"><img
src="https://img.youtube.com/vi/0PHWCApIcCI/0.jpg" style="background-color:rgba(0,0,0,0);" height=300 alt="WhisperLive"></a>
  <br><br>A nearly-live implementation of OpenAI's Whisper.
<br><br>
</h2>

This project is a real-time transcription application that uses the OpenAI Whisper model
to convert speech input into text output. It can be used to transcribe both live audio
input from microphone and pre-recorded audio files.

## Installation
- Install PyAudio and ffmpeg
```bash
 bash scripts/setup.sh
```

- Install whisper-live from pip
```bash
 pip install whisper-live
```

### Setting up NVIDIA/TensorRT-LLM for TensorRT backend
- Please follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) for setup of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and for building Whisper-TensorRT engine.

## Getting Started
The server supports two backends `faster_whisper` and `tensorrt`. If running `tensorrt` backend follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md)

### Running the Server
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) backend
```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper
  
# running with custom model
python3 run_server.py --port 9090 \
                      --backend faster_whisper
                      -fw "/path/to/custom/faster/whisper/model"
```

- 🔥Faster-whisper models can be [downloaded here](https://huggingface.co/ctranslate2-4you).  The easiest option is to download the "int8" for cpu and "float16" for gpu versions of the size of the Whisper model you want and use.  These are the default quantizations specified in this program.  Advanced users can download other quantizations, but make sure that the cpu/gpu you intend to run it on [supports the specific quantization](https://opennmt.net/CTranslate2/quantization.html#quantize-on-model-conversion).  If you choose an unsupported quantization, faster-whisper gracefully will convert to the best supported quantization at runtime, but this will result in increased latency and slightly reduce quality.  Moreover, to actually run at the specified quantization you would have to change the default quantizations in which the model is loaded (within ```server.py```).  The easiest solution, therefore, is to download the "int8" version for cpu-usage and float16 version for gpu usage.  You can also download and open this [html file](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/blob/main/src/User_Manual/transcribe.html) to learn more about the different faster-whisper quantizations.  You'll have to experiment.  Enjoy!🔥
- Note that the faster-whisper backend only supports CUDA 11.8, but CUDA 12+ should be coming soon.

- TensorRT backend. Currently, we recommend to only use the docker setup for TensorRT. Follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) which works as expected. Make sure to build your TensorRT Engines before running the server with TensorRT backend.
```bash
# Run English only model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small_en

# Run Multilingual model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small \
                      -m
```


### Running the Client
- Initializing the client:
```python
from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="small"
)
```
It connects to the server running on localhost at port 9090. Using a multilingual model, language for the transcription will be automatically detected. You can also use the language option to specify the target language for the transcription, in this case, English ("en"). The translate option should be set to `True` if we want to translate from the source language to English and `False` if we want to transcribe in the source language.

- Trancribe an audio file:
```python
client("tests/jfk.wav")
```

- To transcribe from microphone:
```python
client()
```

- To transcribe from a HLS stream:
```python
client(hls_url="http://as-hls-ww-live.akamaized.net/pool_904/live/ww/bbc_1xtra/bbc_1xtra.isml/bbc_1xtra-audio%3d96000.norewind.m3u8") 
```

## Browser Extensions
- Run the server with your desired backend as shown [here](https://github.com/collabora/WhisperLive?tab=readme-ov-file#running-the-server).
- Transcribe audio directly from your browser using our Chrome or Firefox extensions. Refer to [Audio-Transcription-Chrome](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme) and [Audio-Transcription-Firefox](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Firefox#readme) for setup instructions.

## Whisper Live Server in Docker
- GPU
  - Faster-Whisper
  ```bash
  docker build . -t whisper-live -f docker/Dockerfile.gpu
  docker run -it --gpus all -p 9090:9090 whisper-live:latest
  ```

  - TensorRT. Follow [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) in order to setup docker and use TensorRT backend. We provide a pre-built docker image which has TensorRT-LLM built and ready to use.

- CPU
```bash
docker build . -t whisper-live -f docker/Dockerfile.cpu
docker run -it -p 9090:9090 whisper-live:latest
```
**Note**: By default we use "small" model size. To build docker image for a different model size, change the size in server.py and then build the docker image.

## Future Work
- [ ] Add translation to other languages on top of transcription.
- [x] TensorRT backend for Whisper.

## Contact

We are available to help you with both Open Source and proprietary AI projects. You can reach us via the Collabora website or [vineet.suryan@collabora.com](mailto:vineet.suryan@collabora.com) and [marcus.edel@collabora.com](mailto:marcus.edel@collabora.com).

## Citations
```bibtex
@article{Whisper
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  publisher = {arXiv},
  year = {2022},
}
```

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  email = {hello@silero.ai}
}
