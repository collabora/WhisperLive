//
//  WhisperLive_iOS_ClientApp.swift
//  WhisperLive_iOS_Client
//
//  Created by 바견규 on 6/17/25.
//

import SwiftUI

@main
struct WhisperLive_iOS_ClientApp: App {
    var body: some Scene {
        WindowGroup {
            RecordingView {
                // Handle dismiss action here, or leave it empty for now
                print("RecordingView dismissed")
            }
        }
    }
}
