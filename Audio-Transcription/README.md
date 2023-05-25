# Audio Transcription

Audio Transcription is a Chrome extension that allows users to capture any audio playing on the current tab and transcribe it using OpenAI-whisper in real time. Users will have the option to do voice activity detection as well to not send audio to server when there is no speech.

## Loading the Extension
- Open the Google Chrome browser.
- Type chrome://extensions in the address bar and press Enter.
- Enable the Developer mode toggle switch located in the top right corner.
- Clone this repository
- Click the Load unpacked button.
- Browse to the location where you cloned the repository files and select the ```Audio Transcription``` folder.
- The extension should now be loaded and visible on the extensions page.


## Real time transcription with OpenAI-whisper
This Chrome extension allows you to send audio from your browser to a server for transcribing the audio in real time. It can also incorporate voice activity detection on the client side to detect when speech is present, and it continuously receives transcriptions of the spoken content from the server. You can select from the options menu if you want to run the speech recognition.


## Running the Whisper-live server
For a detailed overview of how to run the server to leverage real time transcriptions with OpenAI-whisper, use [whisper-live](https://github.com/collabora/whisper-live) to setup your own server.


## Options
Several options are able to be changed in the extension:
- 'Mute tabs that are being captured' allows the extension to force any tabs currently being captured to be muted on the system's audio output, but still have its audio captured and encoded to the resulting file.
- 'Maximum capture time' changes the amount of time the extension will capture audio for before timing out, and has a limit to prevent exceeding Chrome's memory limit.
- 'Output file format' allows users to choose whether the resulting file will be encoded into .wav or .mp3
- 'MP3 Quality' is only applicable for .mp3 encodings, and will change the bitrate of the encode. (Low: 96 kbps, Medium: 192 kbps, High: 320 kbps)
- 'Enable Voice Activity detection' allows users to run a voice activity detection model before sending audio to a server hosting the OpenAI-whisper model for transcriptions. 


## Implementation Details

### Capturing Audio
To capture the audio in the current tab, I used the chrome `tabCapture` API to obtain a `MediaStream` object of the current tab. Next I used the `MediaStream` object to initialize a recorder that will encode the stream into a .wav file using the `Recorder.js` library.

### Audio Transcription
We use OpenAI-whisper model to process the audio continuously and send the transcription back to the client. We apply a few optimizations on top of OpenAI's implementation to improve performance and run it faster in a real-time manner. To this end, we used [faster-whisper](https://github.com/guillaumekln/faster-whisper) which is 4x faster than OpenAI's implementation.

### Voice Activity Detection
For VAD, we use [silero-vad](https://github.com/snakers4/silero-vad) which is both efficient and accurate for detecting voice activity. It takes around ```1ms``` to process a single audio chunk of ```30ms```. We use the ONNX model in the browser to only send the audio to the server when there is a voice activity.

### Tab Management
To allow audio capture on multiple tabs simultaneously, I stored the `tabId` of each tab being captured into the `sessionStorage` object. When a `stopCapture` command is issued, the extension will check whether the current tab is the same as the tab that the capture was started on, and only stop the specific instance of the capture on the current tab.


### Audio Playback During Capture
By default, using `tabCapture` will mute the audio on the current tab in order for the capture to take place. To allow audio to continue playing during the capture, I created an `Audio` object which has its source linked to the ongoing stream that is being captured. In the options menu, users will have the option to keep the tab muted or unmuted during the capture.


## Limitations
This extension requires an internet connection to stream audio and receive transcriptions. The accuracy of the transcriptions may vary depending on the audio quality and the performance of the server-side transcription service. The extension may consume additional system resources while running, especially when streaming audio.

## License
This extension is provided as-is, without any warranty or guarantee of its performance or suitability for any particular purpose. The developers of this extension shall not be held responsible for any damages or losses incurred while using this extension.
This extension uses LAME MP3 encoder, licensed LGPL.
Everything else is under the MIT License.

