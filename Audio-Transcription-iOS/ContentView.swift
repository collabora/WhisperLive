//
//  ContentView.swift
//  WhisperLive_iOS_Client
//
//  Created by ParkMazorika on 6/17/25.
//

import SwiftUI

/// A standalone view for recording and real-time transcription display.
struct RecordingView: View {
    var onDismiss: () -> Void
    @StateObject private var recordingViewModel = AudioViewModel()
    @State private var showSubmitView = false

    var body: some View {
        VStack(spacing: 0) {
            // Stop button (only visible when recording)
            HStack {
                Spacer()
                if recordingViewModel.isRecording {
                    Button("Stop Recording") {
                        recordingViewModel.stopRecording()
                        recordingViewModel.finalizeTranscription()
                        showSubmitView = true
                    }
                    .font(.headline)
                    .padding()
                    .foregroundColor(.gray)
                }
            }

            // Transcription display
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(recordingViewModel.transcriptionList.indices, id: \.self) { index in
                        Text(recordingViewModel.transcriptionList[index])
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                            .font(.system(size: 14, weight: .semibold))
                    }
                }
                .padding(.horizontal)
            }

            Divider().padding(.top, 8)

            // Timer and Record/Pause/Resume button
            VStack(spacing: 16) {
                Text(recordingViewModel.timeLabel)
                    .font(.system(size: 40))

                Button(action: {
                    if recordingViewModel.isRecording {
                        recordingViewModel.isPaused
                            ? recordingViewModel.resumeRecording()
                            : recordingViewModel.pauseRecording()
                    } else {
                        recordingViewModel.startRecording()
                    }
                }) {
                    Image(systemName: recordingViewModel.isRecording
                          ? (recordingViewModel.isPaused ? "play.circle.fill" : "pause.circle.fill")
                          : "mic.circle.fill")
                        .font(.system(size: 50))
                        .foregroundStyle(.black)
                }
            }
            .padding(.bottom, 40)
        }
        .padding(.top)
        .background(Color(.systemBackground))
        .overlay(
            Group {
                if recordingViewModel.isLoading {
                    ZStack {
                        Color.black.opacity(0.4).ignoresSafeArea()
                        ProgressView("Processing...")
                            .padding()
                            .background(Color.white)
                            .cornerRadius(10)
                    }
                }
            }
        )
        .sheet(isPresented: $showSubmitView) {
            //anotherView
        }
    }
}

#Preview("Recording View") {
    RecordingView {
        // Dummy dismiss handler
        print("RecordingView dismissed")
    }
}
