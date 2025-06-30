//
//  RecordingViewModel.swift
//  Lecture2Quiz
//
//  Created by ParkMazorika on 4/27/25.
//


import Foundation

/// WebSocket client that connects to a transcription server and handles streaming, JSON messages, and retries.
class AudioWebSocket: NSObject, URLSessionWebSocketDelegate {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    private let host: String
    private let port: Int
    private var retryCount = 0
    private let maxRetries = 3
    private var uid: String
    private let modelSize: String
    private var pingTimer: Timer?
    private var processedTexts = Set<String>()

    var onServerReady: (() -> Void)?
    var onTranscriptionReceived: ((String) -> Void)?

    init(host: String, port: Int, modelSize: String = "medium") {
        self.host = host
        self.port = port
        self.uid = UUID().uuidString
        self.modelSize = modelSize
        super.init()

        self.urlSession = URLSession(
            configuration: .default,
            delegate: self,
            delegateQueue: .main
        )
        connect()
    }

    /// Establishes a WebSocket connection with the configured server.
    private func connect() {
        guard retryCount <= maxRetries else {
            print("Maximum reconnect attempts exceeded.")
            return
        }

        let socketURL = port == 443 || port == 80
            ? "wss://\(host)"
            : "wss://\(host):\(port)"

        guard let url = URL(string: socketURL) else {
            print("Invalid URL: \(socketURL)")
            return
        }

        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        print("Attempting WebSocket connection: \(socketURL)")

        listen()
        sendInitialJSON()
        startPing()
    }

    /// Sends the initial JSON payload to identify and configure the session.
    private func sendInitialJSON() {
        let jsonPayload: [String: Any] = [
            "uid": uid,
            "language": "en",
            "task": "transcribe",
            "model": modelSize,
            "use_vad": true,
            "max_clients": 4,
            "max_connection_time": 600
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: jsonPayload, options: [])
            let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
            print("Sending config JSON: \(jsonString)")

            webSocketTask?.send(.string(jsonString)) { [weak self] error in
                if let error = error {
                    print("Failed to send config JSON: \(error.localizedDescription)")
                    self?.reconnect()
                } else {
                    print("Config JSON sent successfully.")
                }
            }
        } catch {
            print("JSON serialization error: \(error.localizedDescription)")
        }
    }

    /// Sends audio data to the server.
    func sendDataToServer(_ data: Data) {
        guard isConnected else {
            print("Not connected - skipping data send.")
            reconnect()
            return
        }

        webSocketTask?.send(.data(data)) { [weak self] error in
            if let error = error {
                print("Failed to send audio data: \(error.localizedDescription)")
                self?.reconnect()
            } else {
                print("Sent audio data: \(data.count) bytes")
            }
        }
    }

    /// Returns true if the WebSocket is currently connected.
    internal var isConnected: Bool {
        webSocketTask?.state == .running
    }

    /// Attempts reconnection with exponential backoff.
    private func reconnect() {
        retryCount += 1
        stopPing()
        let delay = min(5.0, pow(2.0, Double(retryCount)))

        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            print("Reconnecting... (\(self?.retryCount ?? 0)/\(self?.maxRetries ?? 0))")
            self?.connect()
        }
    }

    /// Starts listening for incoming messages from the server.
    private func listen() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleMessage(message)
                self?.listen()
            case .failure(let error):
                print("Receive error: \(error.localizedDescription)")
                self?.reconnect()
            }
        }
    }

    /// Handles incoming WebSocket messages (text or binary).
    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .data(let data):
            print("Received binary data: \(data.count) bytes")

        case .string(let text):
            print("Received text message: \(text)")

            guard let data = text.data(using: .utf8) else { return }

            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let status = json["status"] as? String {
                        handleStatusMessage(status: status, message: json["message"] as? String)
                        return
                    }

                    if let message = json["message"] as? String, message == "SERVER_READY" {
                        print("Server is ready.")
                        onServerReady?()
                        return
                    }

                    if let segments = json["segments"] as? [[String: Any]] {
                        let wrapped = ["segments": segments]
                        let segmentData = try JSONSerialization.data(withJSONObject: wrapped, options: [])
                        let segmentString = String(data: segmentData, encoding: .utf8)!
                        onTranscriptionReceived?(segmentString)
                        print("Transcription segments forwarded.")
                    }
                }
            } catch {
                print("JSON parsing error: \(error.localizedDescription)")
            }

        @unknown default:
            print("Unknown message type received.")
        }
    }

    /// Handles status message JSON from the server.
    private func handleStatusMessage(status: String, message: String?) {
        switch status {
        case "WAIT":
            print("Waiting: \(message ?? "")")
        case "ERROR":
            print("Error: \(message ?? "")")
        case "WARNING":
            print("Warning: \(message ?? "")")
        default:
            print("\(status): \(message ?? "")")
        }
    }

    /// Sends the "END_OF_AUDIO" signal to the server.
    func sendEndOfAudio() {
        guard isConnected else {
            print("Not connected - skipping END_OF_AUDIO.")
            return
        }

        webSocketTask?.send(.string("END_OF_AUDIO")) { error in
            if let error = error {
                print("Failed to send END_OF_AUDIO: \(error.localizedDescription)")
            } else {
                print("END_OF_AUDIO sent.")
            }
        }
    }

    /// Gracefully closes the WebSocket connection.
    func closeConnection() {
        stopPing()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        retryCount = maxRetries
        print("WebSocket closed.")
    }

    /// Starts periodic ping to keep the WebSocket alive.
    private func startPing() {
        stopPing()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 15.0, repeats: true) { [weak self] _ in
            self?.webSocketTask?.sendPing { error in
                if let error = error {
                    print("Ping failed: \(error.localizedDescription)")
                } else {
                    print("Ping sent successfully.")
                }
            }
        }
        RunLoop.main.add(pingTimer!, forMode: .common)
    }

    /// Stops the periodic ping timer.
    private func stopPing() {
        pingTimer?.invalidate()
        pingTimer = nil
    }

    /// Called when the WebSocket is closed by the server.
    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                    reason: Data?) {
        let reasonString = String(data: reason ?? Data(), encoding: .utf8) ?? "No reason"
        print("WebSocket closed - code: \(closeCode.rawValue), reason: \(reasonString)")
        stopPing()
        reconnect()
    }
}
