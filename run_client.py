from whisper_live.client import TranscriptionClient

# ec2-18-197-140-116.eu-central-1.compute.amazonaws.com

if __name__ == "__main__":
    client = TranscriptionClient("ec2-18-157-73-128.eu-central-1.compute.amazonaws.com", 9090, lang="lt",
                                 translate=False, model="large-v2", use_vad=True)
    client(hls_url="https://stream-live.lrt.lt/lituanica/stream04/streamPlaylist.m3u8")
