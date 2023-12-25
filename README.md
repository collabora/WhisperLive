# whisper-live
A nearly-live implementation of OpenAI's Whisper.

This project is a real-time transcription application that uses the OpenAI Whisper model to convert speech input into text output. It can be used to transcribe both live audio input from microphone and pre-recorded audio files.

Unlike traditional speech recognition systems that rely on continuous audio streaming, we use [voice activity detection (VAD)](https://github.com/snakers4/silero-vad) to detect the presence of speech and only send the audio data to whisper when speech is detected. This helps to reduce the amount of data sent to the whisper model and improves the accuracy of the transcription output.

## Installation
- Install PyAudio and ffmpeg
```bash
 bash setup.sh
```

- Install whisper-live from pip
```bash
 pip install whisper-live
```

## Getting Started
- Run the server
```python
 from whisper_live.server import TranscriptionServer
 server = TranscriptionServer()
 server.run("0.0.0.0", 9090)
```

- On the client side
    - To transcribe an audio file:
    ```python
      from whisper_live.client import TranscriptionClient
      client = TranscriptionClient(
        "localhost",
        9090,
        is_multilingual=False,
        lang="en",
        translate=False,
        model_size="small"
      )

      client("tests/jfk.wav")
    ```
    This command transcribes the specified audio file (audio.wav) using the Whisper model. It connects to the server running on localhost at port 9090. It can also enable the multilingual feature, allowing transcription in multiple languages. The language option specifies the target language for transcription, in this case, English ("en"). The translate option should be set to `True` if we want to translate from the source language to English and `False` if we want to transcribe in the source language.

    - To transcribe from microphone:
    ```python
      from whisper_live.client import TranscriptionClient
      client = TranscriptionClient(
        "localhost",
        9090,
        is_multilingual=True,
        lang="hi",
        translate=True,
        model_size="small"
      )
      client()
    ```
    This command captures audio from the microphone and sends it to the server for transcription. It uses the multilingual option with `hi` as the selected language, enabling the multilingual feature and specifying the target language and task. We use whisper `small` by default but can be changed to any other option based on the requirements and the hardware running the server.

    - To transcribe from a HLS stream:
    ```python
      client = TranscriptionClient(host, port, is_multilingual=True, lang="en", translate=False) 
      client(hls_url="http://as-hls-ww-live.akamaized.net/pool_904/live/ww/bbc_1xtra/bbc_1xtra.isml/bbc_1xtra-audio%3d96000.norewind.m3u8") 
    ```
    This command streams audio into the server from a HLS stream. It uses the same options as the previous command, enabling the multilingual feature and specifying the target language and task.

## Transcribe audio from browser
- Run the server
```python
 from whisper_live.server import TranscriptionServer
 server = TranscriptionServer()
 server.run("0.0.0.0", 9090)
```
This would start the websocket server on port ```9090```.

### Chrome Extension
- Refer to [Audio-Transcription-Chrome](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme) to use Chrome extension.

### Firefox Extension
- Refer to [Audio-Transcription-Firefox](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Firefox#readme) to use Mozilla Firefox extension.

## Whisper Live Server in Docker
- GPU
```bash
 docker build . -t whisper-live -f docker/Dockerfile.gpu
 docker run -it --gpus all -p 9090:9090 whisper-live:latest
```

- CPU
```bash
 docker build . -t whisper-live -f docker/Dockerfile.cpu
 docker run -it -p 9090:9090 whisper-live:latest
```
**Note**: By default we use "small" model size. To build docker image for a different model size, change the size in server.py and then build the docker image.

## Future Work
- [ ] Add translation to other languages on top of transcription.
- [ ] TensorRT backend for Whisper.

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
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
