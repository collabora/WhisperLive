// Package whisperlive provides a Go client for the WhisperLive REST API.
//
// Usage:
//
//	client := whisperlive.NewClient("http://localhost:8000", &whisperlive.Config{APIKey: "your-key"})
//	result, err := client.Transcribe("audio.wav", nil)
//	fmt.Println(result.Text)
package whisperlive

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Config holds client configuration.
type Config struct {
	APIKey  string
	Timeout time.Duration
}

// Client is the WhisperLive API client.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// TranscriptionResult holds a transcription response.
type TranscriptionResult struct {
	Text                string  `json:"text"`
	Language            string  `json:"language,omitempty"`
	LanguageProbability float64 `json:"language_probability,omitempty"`
}

// TranscriptionOptions holds optional parameters for transcription.
type TranscriptionOptions struct {
	Model          string
	Language       string
	Prompt         string
	ResponseFormat string
	Temperature    float64
	Hotwords       string
}

// HealthResponse holds the health check response.
type HealthResponse struct {
	Status     string `json:"status"`
	Clients    int    `json:"clients"`
	MaxClients int    `json:"max_clients"`
}

// NewClient creates a new WhisperLive API client.
func NewClient(baseURL string, config *Config) *Client {
	timeout := 30 * time.Second
	apiKey := ""
	if config != nil {
		if config.Timeout > 0 {
			timeout = config.Timeout
		}
		apiKey = config.APIKey
	}
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) setAuth(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

// Transcribe sends an audio file for transcription.
func (c *Client) Transcribe(audioPath string, opts *TranscriptionOptions) (*TranscriptionResult, error) {
	f, err := os.Open(audioPath)
	if err != nil {
		return nil, fmt.Errorf("open audio file: %w", err)
	}
	defer f.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return nil, fmt.Errorf("create form file: %w", err)
	}
	if _, err := io.Copy(part, f); err != nil {
		return nil, fmt.Errorf("copy audio data: %w", err)
	}

	model := "whisper-1"
	respFmt := "json"
	temp := "0.0"
	if opts != nil {
		if opts.Model != "" {
			model = opts.Model
		}
		if opts.ResponseFormat != "" {
			respFmt = opts.ResponseFormat
		}
		if opts.Temperature != 0 {
			temp = fmt.Sprintf("%.1f", opts.Temperature)
		}
		if opts.Language != "" {
			writer.WriteField("language", opts.Language)
		}
		if opts.Prompt != "" {
			writer.WriteField("prompt", opts.Prompt)
		}
		if opts.Hotwords != "" {
			writer.WriteField("hotwords", opts.Hotwords)
		}
	}
	writer.WriteField("model", model)
	writer.WriteField("response_format", respFmt)
	writer.WriteField("temperature", temp)
	writer.Close()

	req, err := http.NewRequest("POST", c.baseURL+"/v1/audio/transcriptions", &buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("transcription failed (%d): %s", resp.StatusCode, string(body))
	}

	var result TranscriptionResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

// Health checks the server health status.
func (c *Client) Health() (*HealthResponse, error) {
	req, err := http.NewRequest("GET", c.baseURL+"/health", nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	var result HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}
