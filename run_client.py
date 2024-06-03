from whisper_live.client import TranscriptionClient

if __name__ == "__main__":
    client = TranscriptionClient("localhost", 9090, lang="lt", translate=False, model="isLucid/faster-whisper-large-v2",
                                 use_vad=True)
    client(hls_url="https://stream-live.lrt.lt/lituanica/stream04/streamPlaylist.m3u8")
