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

This directory contains the Swift source files for the iOS client, but it does not include a generated `.xcodeproj` or `.xcodeworkspace`. Create a new Xcode project and add these files to it.

1. Clone the repository (your fork):

    ```bash
    git clone https://github.com/yourusername/whisperlive.git
    cd whisperlive/Audio-Transcription-iOS
    ```

2. In Xcode, choose **File ▸ New ▸ Project…**, then create an iOS **App** project with SwiftUI.

3. Add the Swift files from this directory to the new app target:

    - `AudioStream.swift`
    - `AudioWebSocket.swift`
    - `ContentView.swift`
    - `RecordingViewModel.swift`
    - `WhisperLive_iOS_ClientApp.swift`

4. Use `WhisperLive-iOS-Client-Info.plist` as a reference for your app's `Info.plist`, or add the microphone usage description manually:

    ```xml
    <key>NSMicrophoneUsageDescription</key>
    <string>This app requires microphone access for transcription.</string>
    ```

5. Run the app on a physical device (recommended)

## Running on a Physical Device (with Free Apple ID)

You can run this app on a real iPhone without a paid Apple Developer account. Follow these steps:

### 1. Register a Free Apple ID in Xcode

1. Open Xcode ▸ Settings… (or Preferences) ▸ **Accounts**
2. Click the **+** button ▸ Select **Apple ID**
3. Sign in with your Apple ID (a free one is fine)
4. A "Personal Team" will be created automatically

> ✅ You can deploy up to 3 apps on a physical device using a free Apple ID with a 7-day provisioning profile.

---

### 2. Set Up Signing in Your Project

1. In Xcode, select your **project** in the Project Navigator
2. Go to **TARGETS ▸ YourAppName ▸ Signing & Capabilities**
3. Set **Team** to your Personal Team
4. Set a unique **Bundle Identifier** (e.g., `com.yourname.whisperlive`)
5. Make sure **Automatically manage signing** is checked
6. If a red warning appears, click **"Resolve Issues"**

---

### 3. Connect and Trust Your iPhone

1. Connect your iPhone via USB
2. When prompted, tap **“Trust This Computer”** on your iPhone
3. Make sure your iPhone appears in Xcode's device list

---

### 4. Enable Developer Mode on iPhone

1. Press the **Build (▶︎)** button in Xcode
2. Your iPhone will ask to enable **Developer Mode**
3. On iPhone, go to:  
   **Settings ▸ Privacy & Security ▸ Developer Mode**
4. Enable it and restart the device if required

---

Now you can run and debug the app on your real device!



## Folder Structure
```
Audio-Transcription-iOS/
├── AudioStream.swift
├── AudioWebSocket.swift
├── ContentView.swift
├── RecordingViewModel.swift
├── WhisperLive-iOS-Client-Info.plist
├── WhisperLive_iOS_ClientApp.swift
├── README.md
```

## License

MIT  
This iOS client is provided as an open-source example to complement WhisperLive's real-time transcription ecosystem.


