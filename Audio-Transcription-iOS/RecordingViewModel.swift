//
//  RecordingViewModel.swift
//  Lecture2Quiz
//
//  Created by ParkMazorika on 4/27/25.
//

import AVFoundation
import Combine

/// Represents a segment of transcribed audio with start/end timestamps and completion flag.
struct TranscriptionSegment: Identifiable, Equatable {
    var id = UUID()
    var start: Double
    var end: Double
    var text: String
    var completed: Bool
}

/// ViewModel responsible for managing audio recording and transcription logic.
class AudioViewModel: ObservableObject {
    @Published var isRecording = false            // Indicates if recording is active
    @Published var isPaused = false               // Indicates if recording is currently paused
    @Published var timeLabel = "00:00"            // Timer label formatted as mm:ss
    @Published var transcriptionList: [String] = []  // Live transcription output
    @Published var isLoading = false              // True while waiting for server response
    @Published var finalScript: String = ""       // Final script from completed segments

    private var timer: Timer?
    private var elapsedTime: Int = 0

    private var audioStreamer: AudioStreamer?     // Handles audio capture and streaming
    private var audioWebSocket: AudioWebSocket?   // Manages WebSocket communication

    private var segments: [TranscriptionSegment] = []  // Stores all transcription segments

    init() {}

    /// Starts audio recording and initializes WebSocket + AVAudioEngine.
    func startRecording() {
        let audioAPIUrl = "your server url"
        audioWebSocket = AudioWebSocket(host: audioAPIUrl, port: 443)
        audioStreamer = AudioStreamer(webSocket: audioWebSocket!)

        isLoading = true

        // Handle server transcription message
        audioWebSocket?.onTranscriptionReceived = { [weak self] text in
            self?.handleRawTranscriptionJSON(text)
        }

        // When server sends SERVER_READY
        audioWebSocket?.onServerReady = { [weak self] in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.isLoading = false
                self.isRecording = true
                self.isPaused = false
                self.timeLabel = "00:00"
                self.elapsedTime = 0
                self.startTimer()
                self.audioStreamer?.startStreaming()
            }
        }
    }

    /// Pauses the recording and stops the timer.
    func pauseRecording() {
        isPaused = true
        audioStreamer?.pauseStreaming()
        timer?.invalidate()
    }

    /// Resumes recording and restarts the timer.
    func resumeRecording() {
        isPaused = false
        audioStreamer?.resumeStreaming()
        startTimer()
    }

    /// Stops recording and finalizes connection to server.
    func stopRecording() {
        isRecording = false
        isPaused = false
        timer?.invalidate()

        audioStreamer?.stopStreaming()
        audioWebSocket?.sendEndOfAudio()
        audioWebSocket?.onTranscriptionReceived = nil
        audioWebSocket?.closeConnection()
    }

    /// Starts the recording timer (1-second interval).
    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.elapsedTime += 1
            let minutes = self.elapsedTime / 60
            let seconds = self.elapsedTime % 60
            self.timeLabel = String(format: "%02d:%02d", minutes, seconds)
        }
    }

    /// Finalizes the transcription by joining all completed segments into one string.
    func finalizeTranscription() {
        isLoading = false
        let completedText = segments
            .filter { $0.completed }
            .map { $0.text.trimmingCharacters(in: .whitespaces) }
            .joined(separator: " ")
        finalScript = completedText
        print("Final transcript:\n\(finalScript)")
    }

    /// Handles incoming JSON from the server and updates UI state.
    /// Supports both full JSON and raw string cases.
    func handleRawTranscriptionJSON(_ jsonString: String) {
        let trimmed = jsonString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else { return }

        if trimmed.hasPrefix("{") {
            // Parse JSON containing segment list
            do {
                if let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let segmentDicts = dict["segments"] as? [[String: Any]] {

                    for item in segmentDicts {
                        guard let startStr = item["start"] as? String,
                              let endStr = item["end"] as? String,
                              let text = item["text"] as? String,
                              let completed = item["completed"] as? Bool,
                              let start = Double(startStr),
                              let end = Double(endStr) else { continue }

                        let newSegment = TranscriptionSegment(start: start, end: end, text: text, completed: completed)

                        // Overwrite if already exists, else append
                        if let index = self.segments.firstIndex(where: { $0.start == start }) {
                            self.segments[index] = newSegment
                        } else {
                            self.segments.append(newSegment)
                        }
                    }

                    // Update the UI
                    DispatchQueue.main.async {
                        let completedTexts = self.segments
                            .filter { $0.completed }
                            .sorted(by: { $0.start < $1.start })
                            .map { $0.text.trimmingCharacters(in: .whitespaces) }

                        let pendingText = self.segments
                            .filter { !$0.completed }
                            .sorted(by: { $0.start < $1.start })
                            .map { $0.text.trimmingCharacters(in: .whitespaces) }
                            .last ?? ""

                        self.transcriptionList = completedTexts + (pendingText.isEmpty ? [] : [pendingText])
                        self.finalScript = self.transcriptionList.joined(separator: " ")
                    }
                }
            } catch {
                print("JSON parsing error: \(error)")
            }
        } else {
            // Handle raw text line
            DispatchQueue.main.async {
                if self.transcriptionList.last != trimmed {
                    self.transcriptionList.append(trimmed)
                    self.finalScript = self.transcriptionList.joined(separator: " ")
                }
            }
        }
    }
}
