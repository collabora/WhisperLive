//  AudioStream.swift
//  Lecture2Quiz
//
//  Created by ParkMazorika on 4/27/25.
//

import AVFoundation

/// Streams audio input to a WebSocket after converting and normalizing.
class AudioStreamer {
    private let engine = AVAudioEngine()
    private let inputNode: AVAudioInputNode
    private var inputFormat: AVAudioFormat?
    private var isPaused: Bool = false
    private var audioWebSocket: AudioWebSocket?
    private var partialBuffer = Data()
    private var isStreaming: Bool = false

    private var bufferSize: AVAudioFrameCount = 1600  // ~100ms of audio
    private var sampleRate: Double = 16000
    private var channels: UInt32 = 1

    private var converter: AVAudioConverter?

    init(webSocket: AudioWebSocket) {
        self.inputNode = engine.inputNode
        self.audioWebSocket = webSocket

        let inputFormat = inputNode.outputFormat(forBus: 0)
        print("Input format: \(inputFormat)")

        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16000,
            channels: 1,
            interleaved: true
        )!

        self.converter = AVAudioConverter(from: inputFormat, to: outputFormat)
        self.inputFormat = outputFormat
    }

    /// Configures the audio session for recording.
    func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.allowBluetooth, .defaultToSpeaker])
            try session.setPreferredSampleRate(48000)
            try session.setPreferredInputNumberOfChannels(1)
            try session.setMode(.videoChat)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            sampleRate = session.sampleRate
            channels = UInt32(session.inputNumberOfChannels)
            print("Sample rate: \(sampleRate)")
            print("Input channels: \(channels)")
        } catch {
            print("Failed to configure audio session: \(error.localizedDescription)")
        }
    }

    /// Starts capturing and streaming audio data.
    func startStreaming() {
        guard !isStreaming else {
            print("Already streaming.")
            return
        }

        configureAudioSession()

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 48000,
            channels: channels,
            interleaved: true
        )

        guard let hardwareFormat = format else {
            print("Failed to create audio format.")
            return
        }

        self.inputFormat = hardwareFormat

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: hardwareFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }

        do {
            try engine.start()
            isStreaming = true
            print("AVAudioEngine started.")
        } catch {
            print("Failed to start AVAudioEngine: \(error.localizedDescription)")
        }
    }

    /// Converts and sends the audio buffer to the server via WebSocket.
    func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let converter = self.converter else {
            print("Audio converter is nil.")
            return
        }

        if let floatChannelData = buffer.floatChannelData {
            let frameLength = Int(buffer.frameLength)
            let channelData = Array(UnsafeBufferPointer(start: floatChannelData.pointee, count: frameLength))
            let rms = sqrt(channelData.map { $0 * $0 }.reduce(0, +) / Float(frameLength))
            print("Audio RMS: \(rms)")
            if rms < 0.001 {
                print("Warning: Input volume is too low.")
            }
        }

        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16000,
            channels: 1,
            interleaved: true
        )!

        guard let newBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 1600) else {
            print("Failed to allocate PCM buffer.")
            return
        }

        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }

        var error: NSError?
        converter.convert(to: newBuffer, error: &error, withInputFrom: inputBlock)

        if let error = error {
            print("Audio conversion failed: \(error.localizedDescription)")
            return
        }

        print("Converted buffer frameLength: \(newBuffer.frameLength), sampleRate: \(newBuffer.format.sampleRate)")

        if let audioData = convertToFloat32BytesLikePython(newBuffer) {
            var completeData = partialBuffer + audioData
            let chunkSize = 4096

            while completeData.count >= chunkSize {
                let chunk = completeData.prefix(chunkSize)
                audioWebSocket?.sendDataToServer(chunk)
                print("Sent 4096 bytes of audio.")
                completeData.removeFirst(chunkSize)
            }

            partialBuffer = completeData
        }
    }

    /// Converts the audio buffer to Float32 Data with RMS normalization and soft clipping.
    func convertToFloat32BytesLikePython(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let int16ChannelData = buffer.int16ChannelData else {
            print("int16ChannelData is nil.")
            return nil
        }

        let frameLength = Int(buffer.frameLength)
        let channelPointer = int16ChannelData.pointee

        var floatArray = [Float32](repeating: 0, count: frameLength)
        for i in 0..<frameLength {
            let int16Value = channelPointer[i]
            floatArray[i] = Float32(Int16(littleEndian: int16Value)) / 32768.0
        }

        let rms = sqrt(floatArray.map { $0 * $0 }.reduce(0, +) / Float(frameLength))
        let targetRMS: Float32 = 0.25
        let gain = targetRMS / max(rms, 0.00001)

        print("Original RMS: \(rms), applied gain: \(gain)")

        for i in 0..<frameLength {
            let scaled = floatArray[i] * gain
            let clipped = tanh(scaled * 3.0)
            floatArray[i] = clipped
        }

        let floatData = Data(bytes: floatArray, count: frameLength * MemoryLayout<Float32>.size)

        if let minVal = floatArray.min(), let maxVal = floatArray.max() {
            print("Float32 value range after normalization: \(minVal)...\(maxVal)")
        }

        print("Converted to Float32 data: \(floatData.count) bytes")
        return floatData
    }

    /// Pauses audio streaming by removing the input tap.
    func pauseStreaming() {
        guard !isPaused else { return }
        inputNode.removeTap(onBus: 0)
        isPaused = true
        print("Audio streaming paused.")
    }

    /// Resumes audio streaming by reinstalling the input tap.
    func resumeStreaming() {
        guard isPaused else { return }
        guard let inputFormat = inputFormat else {
            print("inputFormat is nil.")
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
        isPaused = false
        print("Audio streaming resumed.")
    }

    /// Stops the AVAudioEngine and resets streaming state.
    func stopStreaming() {
        guard isStreaming else {
            print("Already stopped.")
            return
        }

        inputNode.removeTap(onBus: 0)
        engine.stop()
        isStreaming = false
        print("AVAudioEngine stopped.")
    }
}
