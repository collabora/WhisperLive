# Audio-Transcription-iOS

This is an iOS client for [WhisperLive](https://github.com/collabora/WhisperLive), a real-time speech-to-text server based on OpenAI Whisper.  
The app streams microphone audio to a WhisperLive server via WebSocket and displays live transcription results in real time.

> ⚠️ This client is designed to work specifically with the [WhisperLive Python WebSocket server](https://github.com/collabora/WhisperLive?tab=readme-ov-file#running-the-server).  
> Make sure the server is running and reachable from your iOS device.

## Features

- Real-time microphone capture with AVAudioEngine
- Streaming to WhisperLive backend using WebSocket
- Displays transcription as segments arrive
- Start / Pause / Resume / Stop recording with SwiftUI interface
- Final transcription view on stop

## Requirements

- iOS 15.0+
- Swift 5.8+
- AVFoundation (for microphone)
- Working WhisperLive WebSocket server

## Getting Started

1. Clone the repository (your fork):

    ```bash
    git clone https://github.com/yourusername/whisperlive.git
    cd whisperlive/Audio-Transcription-iOS
    ```

2. Open the `.xcodeproj` or `.xcodeworkspace` in Xcode

3. Add the following to your `Info.plist`:

    ```xml
    <key>NSMicrophoneUsageDescription</key>
    <string>This app requires microphone access for transcription.</string>
    ```

4. Run the app on a physical device (recommended)

## Folder Structure
Audio-Transcription-iOS/
├── AudioViewModel.swift
├── AudioStreamer.swift
├── AudioWebSocket.swift
├── RecordingView.swift
├── WhisperLive_iOS_ClientApp.swift
├── Info.plist
├── README.md


## License

MIT  
This iOS client is provided as an open-source example to complement WhisperLive's real-time transcription ecosystem.


