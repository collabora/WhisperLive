/**
 * WhisperLive JavaScript/TypeScript SDK
 *
 * Lightweight client for the WhisperLive REST API and WebSocket streaming.
 *
 * Usage:
 *   import { WhisperLiveClient } from './whisperlive';
 *
 *   const client = new WhisperLiveClient('http://localhost:8000', { apiKey: 'your-key' });
 *   const result = await client.transcribe(audioFile);
 *   console.log(result.text);
 */

export interface TranscriptionOptions {
  model?: string;
  language?: string;
  prompt?: string;
  responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
  temperature?: number;
  timestampGranularities?: string[];
  hotwords?: string;
  stream?: boolean;
  callbackUrl?: string;
  multichannel?: boolean;
  channelLabels?: string;
}

export interface TranscriptionResult {
  text: string;
  language?: string;
  language_probability?: number;
}

export interface IntelligenceOptions {
  model?: string;
  language?: string;
  sentiment?: boolean;
  topics?: boolean;
  entities?: boolean;
  summary?: boolean;
  summarySentences?: number;
  topicCount?: number;
}

export interface HealthResponse {
  status: string;
  clients: number;
  max_clients: number;
}

export interface WhisperLiveConfig {
  apiKey?: string;
  timeout?: number;
}

export class WhisperLiveClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(baseUrl: string, config: WhisperLiveConfig = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {};
    if (this.apiKey) {
      h['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return h;
  }

  /**
   * Transcribe an audio file.
   */
  async transcribe(
    file: File | Blob,
    options: TranscriptionOptions = {}
  ): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append('file', file);
    form.append('model', options.model || 'whisper-1');
    if (options.language) form.append('language', options.language);
    if (options.prompt) form.append('prompt', options.prompt);
    form.append('response_format', options.responseFormat || 'json');
    form.append('temperature', String(options.temperature ?? 0.0));
    if (options.hotwords) form.append('hotwords', options.hotwords);
    if (options.stream) form.append('stream', 'true');
    if (options.callbackUrl) form.append('callback_url', options.callbackUrl);
    if (options.multichannel) form.append('multichannel', 'true');
    if (options.channelLabels) form.append('channel_labels', options.channelLabels);
    if (options.timestampGranularities) {
      for (const g of options.timestampGranularities) {
        form.append('timestamp_granularities', g);
      }
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const resp = await fetch(`${this.baseUrl}/v1/audio/transcriptions`, {
        method: 'POST',
        headers: this.headers(),
        body: form,
        signal: controller.signal,
      });

      if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`Transcription failed (${resp.status}): ${body}`);
      }

      return await resp.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Stream transcription results via SSE.
   */
  async *transcribeStream(
    file: File | Blob,
    options: Omit<TranscriptionOptions, 'stream'> = {}
  ): AsyncGenerator<Record<string, unknown>> {
    const form = new FormData();
    form.append('file', file);
    form.append('model', options.model || 'whisper-1');
    form.append('stream', 'true');
    if (options.language) form.append('language', options.language);
    if (options.hotwords) form.append('hotwords', options.hotwords);

    const resp = await fetch(`${this.baseUrl}/v1/audio/transcriptions`, {
      method: 'POST',
      headers: this.headers(),
      body: form,
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`Stream failed (${resp.status})`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data);
          } catch {
            // skip malformed SSE events
          }
        }
      }
    }
  }

  /**
   * Analyze transcript with audio intelligence.
   */
  async analyze(
    file: File | Blob,
    options: IntelligenceOptions = {}
  ): Promise<Record<string, unknown>> {
    const form = new FormData();
    form.append('file', file);
    form.append('model', options.model || 'whisper-1');
    if (options.language) form.append('language', options.language);
    if (options.sentiment !== undefined) form.append('sentiment', String(options.sentiment));
    if (options.topics !== undefined) form.append('topics', String(options.topics));
    if (options.entities !== undefined) form.append('entities', String(options.entities));
    if (options.summary !== undefined) form.append('summary', String(options.summary));

    const resp = await fetch(`${this.baseUrl}/v1/audio/intelligence`, {
      method: 'POST',
      headers: this.headers(),
      body: form,
    });

    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`Intelligence failed (${resp.status}): ${body}`);
    }

    return await resp.json();
  }

  /**
   * Check server health.
   */
  async health(): Promise<HealthResponse> {
    const resp = await fetch(`${this.baseUrl}/health`, {
      headers: this.headers(),
    });
    return await resp.json();
  }

  /**
   * List loaded models.
   */
  async listModels(): Promise<Record<string, unknown>[]> {
    const resp = await fetch(`${this.baseUrl}/v1/models`, {
      headers: this.headers(),
    });
    const data = await resp.json();
    return data.models;
  }
}

/**
 * WebSocket client for real-time transcription streaming.
 */
export class WhisperLiveWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private config: Record<string, unknown>;

  constructor(
    host: string,
    port: number,
    config: {
      language?: string;
      model?: string;
      useVad?: boolean;
      wordTimestamps?: boolean;
      hotwords?: string;
      enableDiarization?: boolean;
      smartFormatting?: boolean;
      piiRedaction?: string | boolean;
      profanityFilter?: string | boolean;
      apiKey?: string;
    } = {}
  ) {
    const protocol = 'ws';
    const tokenParam = config.apiKey ? `?token=${config.apiKey}` : '';
    this.url = `${protocol}://${host}:${port}${tokenParam}`;
    this.config = {
      uid: crypto.randomUUID(),
      language: config.language || null,
      task: 'transcribe',
      model: config.model || 'small',
      use_vad: config.useVad ?? true,
      word_timestamps: config.wordTimestamps ?? false,
      hotwords: config.hotwords || null,
      enable_diarization: config.enableDiarization ?? false,
      smart_formatting: config.smartFormatting ?? false,
      pii_redaction: config.piiRedaction || null,
      profanity_filter: config.profanityFilter || null,
    };
  }

  connect(
    onSegments: (segments: Record<string, unknown>[]) => void,
    onError?: (error: Event) => void
  ): void {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.ws?.send(JSON.stringify(this.config));
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.segments) {
          onSegments(data.segments);
        }
      } catch {
        // ignore non-JSON messages
      }
    };

    this.ws.onerror = (event) => {
      if (onError) onError(event);
    };
  }

  sendAudio(audioData: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }

  close(): void {
    if (this.ws) {
      this.ws.send(new TextEncoder().encode('END_OF_AUDIO'));
      this.ws.close();
      this.ws = null;
    }
  }
}
