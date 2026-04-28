# WhisperLive SDKs

Client SDKs for integrating with the WhisperLive transcription API.

## Available SDKs

### JavaScript / TypeScript

```bash
# Copy sdks/javascript/whisperlive.ts into your project
```

```typescript
import { WhisperLiveClient, WhisperLiveWebSocket } from './whisperlive';

// REST API client
const client = new WhisperLiveClient('http://localhost:8000', { apiKey: 'your-key' });
const result = await client.transcribe(audioFile);
console.log(result.text);

// SSE streaming
for await (const segment of client.transcribeStream(audioFile)) {
  console.log(segment.text);
}

// WebSocket real-time
const ws = new WhisperLiveWebSocket('localhost', 9090, {
  model: 'small',
  wordTimestamps: true,
  enableDiarization: true,
});
ws.connect((segments) => {
  segments.forEach(seg => console.log(seg.text));
});
ws.sendAudio(audioBuffer);
```

### Go

```bash
go get github.com/collabora/WhisperLive/sdks/go
```

```go
package main

import (
    "fmt"
    wl "github.com/collabora/WhisperLive/sdks/go"
)

func main() {
    client := wl.NewClient("http://localhost:8000", &wl.Config{APIKey: "your-key"})
    
    result, err := client.Transcribe("audio.wav", nil)
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Text)
    
    health, _ := client.Health()
    fmt.Printf("Server: %s (%d/%d clients)\n", health.Status, health.Clients, health.MaxClients)
}
```

### Python

The Python client is built into WhisperLive itself:

```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    host="localhost",
    port=9090,
    model="small",
    word_timestamps=True,
    enable_diarization=True,
    smart_formatting=True,
    pii_redaction="all",
    profanity_filter="partial",
)
client()
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/audio/intelligence` | POST | Transcribe + NLP analysis |
| `/v1/models` | GET | List loaded models |
| `/v1/plugins` | GET | List registered plugins |
| `/health` | GET | Server health check |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/openapi.json` | GET | OpenAPI specification |
| `ws://{host}:{port}` | WebSocket | Real-time streaming transcription |
