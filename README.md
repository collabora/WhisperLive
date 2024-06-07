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
#### Controlling OpenMP Threads
To control the number of threads used by OpenMP, you can set the `OMP_NUM_THREADS` environment variable. This is useful for managing CPU resources and ensuring consistent performance. If not specified, `OMP_NUM_THREADS` is set to `1` by default. You can change this by using the `--omp_num_threads` argument:
```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper \
                      --omp_num_threads 4
```

#### Single model mode
By default, when running the server without specifying a model, the server will instantiate a new whisper model for every client connection. This has the advantage, that the server can use different model sizes, based on the client's requested model size. On the other hand, it also means you have to wait for the model to be loaded upon client connection and you will have increased (V)RAM usage.

When serving a custom TensorRT model using the `-trt` or a custom faster_whisper model using the `-fw` option, the server will instead only instantiate the custom model once and then reuse it for all client connections.

If you don't want this, set `--no_single_model`.


### Running the Client
- Initializing the client with below parameters:
  - `lang`: Language of the input audio, applicable only if using a multilingual model.
  - `translate`: If set to `True` then translate from any language to `en`.
  - `model`: Whisper model size.
  - `use_vad`: Whether to use `Voice Activity Detection` on the server.
  - `save_output_recording`: Set to True to save the microphone input as a `.wav` file during live transcription. This option is helpful for recording sessions for later playback or analysis. Defaults to `False`. 
  - `output_recording_filename`: Specifies the `.wav` file path where the microphone input will be saved if `save_output_recording` is set to `True`.
```python
from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="small",
  use_vad=False,
  save_output_recording=True,                         # Only used for microphone input, False by Default
  output_recording_filename="./output_recording.wav"  # Only used for microphone input
)
```
It connects to the server running on localhost at port 9090. Using a multilingual model, language for the transcription will be automatically detected. You can also use the language option to specify the target language for the transcription, in this case, English ("en"). The translate option should be set to `True` if we want to translate from the source language to English and `False` if we want to transcribe in the source language.

- Transcribe an audio file:
```python
client("tests/jfk.wav")
```

- To transcribe from microphone:
```python
client()
```

- To transcribe from a RTSP stream:
```python
client(rtsp_url="rtsp://admin:admin@192.168.0.1/rtsp")
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
  docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
  ```

  - TensorRT. 
  ```bash
  docker run -p 9090:9090 --runtime=nvidia --gpus all --entrypoint /bin/bash -it ghcr.io/collabora/whisperlive-tensorrt

  # Build tiny.en engine
  bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples small.en

  # Run server with tiny.en
  python3 run_server.py --port 9090 \
                        --backend tensorrt \
                        --trt_model_path "/app/TensorRT-LLM-examples/whisper/whisper_small_en"
  ```

- CPU
```bash
docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
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
