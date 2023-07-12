# whisper-live
A nearly-live implementation of OpenAI's Whisper.

This project is a real-time transcription application that uses the OpenAI Whisper model to convert speech input into text output. It can be used to transcribe both live audio input from microphone and pre-recorded audio files.

Unlike traditional speech recognition systems that rely on continuous audio streaming, we use [voice activity detection (VAD)](https://github.com/snakers4/silero-vad) to detect the presence of speech and only send the audio data to whisper when speech is detected. This helps to reduce the amount of data sent to the API and improves the accuracy of the transcription output.

## Installation
- Install PyAudio and ffmpeg
```bash
 bash setup.sh
```

- To install client requirements
```bash
 pip install -r requirements/client.txt
```

- To install server requirements
```bash
 pip install -r requirements/server.txt
```

## Getting Started
- Run the server
```bash
 python server.py
```

- On the client side
    - To transcribe an audio file:
    ```bash
     python client.py --audio "audio.wav" --host "localhost" --port "9090"
    ```

    - To transcribe from microphone:
    ```bash
     python client.py --host "localhost" --port "9090"
    ```

## Transcribe audio from browser
- Run the server
```bash
 python server.py
```
This would start the websocket server on port ```9090```.

### Chrome Extension
- Refer to [Audio-Transcription-Chrome](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme) to use Chrome extension.

### Firefox Extension
- Refer to [Audio-Transcription-Firefox](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Firefox#readme) to use Mozilla Firefox extension.

## Whisper Live Server in Docker
- Build docker container
```bash
 docker build . -t whisper-live
```

- Run docker container
```bash
 docker run -it --gpus all -p 9090:9090 whisper-live:latest
```

## Future Work
- [ ] Update Documentation.
- [x] Keep only a single server implementation i.e. websockets and get rid of the socket implementation in ```server.py```. Also, update ```client.py``` to websockets-client implemenation.
- [ ] Add translation to other languages on top of transcription.

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
