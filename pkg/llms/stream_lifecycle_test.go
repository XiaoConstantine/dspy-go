package llms

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/require"
)

// TestStreamCancelClosesStalledRequest verifies the lifecycle at the HTTP
// boundary: Cancel must reach a request stalled after Do has returned, and the
// producer must then close its output channel without requiring a receiver.
func TestStreamCancelClosesStalledRequest(t *testing.T) {
	tests := []struct {
		name string
		new  func(string) (core.LLM, error)
	}{
		{
			name: "openai",
			new: func(url string) (core.LLM, error) {
				return NewOpenAILLM(
					core.ModelOpenAIGPT4o,
					WithAPIKey("test-key"),
					WithOpenAIBaseURL(url),
				)
			},
		},
		{
			name: "anthropic",
			new: func(url string) (core.LLM, error) {
				return NewAnthropicLLMFromConfig(context.Background(), core.ProviderConfig{
					APIKey: "test-key",
					Endpoint: &core.EndpointConfig{
						BaseURL: url,
					},
				}, core.ModelID("claude-sonnet-4-5"))
			},
		},
		{
			name: "gemini",
			new: func(url string) (core.LLM, error) {
				llm, err := NewGeminiLLM("key", core.ModelGoogleGeminiFlash)
				if err == nil {
					llm.GetEndpointConfig().BaseURL = url
				}
				return llm, err
			},
		},
		{name: "llamacpp", new: func(url string) (core.LLM, error) { return NewLlamacppLLM(url) }},
		{
			name: "ollama-native",
			new: func(url string) (core.LLM, error) {
				return NewOllamaLLM("test", WithBaseURL(url), WithNativeAPI())
			},
		},
		{
			name: "ollama-openai",
			new: func(url string) (core.LLM, error) {
				return NewOllamaLLM("test", WithBaseURL(url), WithOpenAIAPI())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestStarted := make(chan struct{})
			requestCanceled := make(chan struct{})
			releaseRequest := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				close(requestStarted)
				select {
				case <-r.Context().Done():
					close(requestCanceled)
				case <-releaseRequest:
				}
			}))
			defer func() {
				close(releaseRequest)
				server.CloseClientConnections()
				server.Close()
			}()

			llm, err := tt.new(server.URL)
			require.NoError(t, err)
			stream, err := llm.StreamGenerate(context.Background(), "prompt")
			require.NoError(t, err)
			requireChannelSignal(t, requestStarted, "HTTP request did not start")

			stream.Cancel()
			requireChannelSignal(t, requestCanceled, "Cancel did not cancel HTTP request")
			requireStreamClosed(t, stream.ChunkChannel)
		})
	}
}

func TestSendStreamChunkReturnsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result := make(chan bool, 1)
	go func() {
		result <- sendStreamChunk(ctx, make(chan core.StreamChunk), core.StreamChunk{Content: "blocked"})
	}()

	select {
	case sent := <-result:
		require.False(t, sent)
	case <-time.After(2 * time.Second):
		t.Fatal("stream send remained blocked after cancellation")
	}
}

// TestStreamCancelUnblocksPendingSend verifies the cancellation lifecycle after
// each provider has emitted a frame while its output channel remains unread.
func TestStreamCancelUnblocksPendingSend(t *testing.T) {
	const (
		openAIFrame    = "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"blocked\"},\"finish_reason\":null}]}\n\n"
		anthropicFrame = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"blocked\"}}\n\n"
	)

	newAnthropic := func(url string) (*AnthropicLLM, error) {
		return NewAnthropicLLMFromConfig(context.Background(), core.ProviderConfig{
			APIKey: "test-key",
			Endpoint: &core.EndpointConfig{
				BaseURL: url,
			},
		}, core.ModelID("claude-sonnet-4-5"))
	}

	tests := []struct {
		name  string
		frame string
		start func(string) (*core.StreamResponse, error)
	}{
		{
			name:  "openai",
			frame: openAIFrame,
			start: func(url string) (*core.StreamResponse, error) {
				llm, err := NewOpenAILLM(
					core.ModelOpenAIGPT4o,
					WithAPIKey("test-key"),
					WithOpenAIBaseURL(url),
				)
				if err != nil {
					return nil, err
				}
				return llm.StreamGenerate(context.Background(), "prompt")
			},
		},
		{
			name:  "anthropic-text",
			frame: anthropicFrame,
			start: func(url string) (*core.StreamResponse, error) {
				llm, err := newAnthropic(url)
				if err != nil {
					return nil, err
				}
				return llm.StreamGenerate(context.Background(), "prompt")
			},
		},
		{
			name:  "anthropic-content",
			frame: anthropicFrame,
			start: func(url string) (*core.StreamResponse, error) {
				llm, err := newAnthropic(url)
				if err != nil {
					return nil, err
				}
				return llm.StreamGenerateWithContent(context.Background(), []core.ContentBlock{
					core.NewTextBlock("prompt"),
				})
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			frameWritten := make(chan struct{})
			requestCanceled := make(chan struct{})
			releaseRequest := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tt.frame))
				w.(http.Flusher).Flush()
				close(frameWritten)
				select {
				case <-r.Context().Done():
					close(requestCanceled)
				case <-releaseRequest:
				}
			}))
			defer func() {
				close(releaseRequest)
				server.CloseClientConnections()
				server.Close()
			}()

			stream, err := tt.start(server.URL)
			require.NoError(t, err)
			requireChannelSignal(t, frameWritten, "provider frame was not written")

			// Deliberately leave ChunkChannel unread until cancellation.
			stream.Cancel()

			requireChannelSignal(t, requestCanceled, "Cancel did not cancel HTTP request")
			requireStreamClosed(t, stream.ChunkChannel)
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type cancelReadCloser struct {
	ctx     context.Context
	started chan struct{}
	once    sync.Once
}

func (b *cancelReadCloser) Read(_ []byte) (int, error) {
	b.once.Do(func() { close(b.started) })
	<-b.ctx.Done()
	return 0, errors.New("forced read failure after cancellation")
}

func (b *cancelReadCloser) Close() error { return nil }

func TestLlamacppCancelDoesNotEmitReadError(t *testing.T) {
	llm, err := NewLlamacppLLM("http://llamacpp.invalid")
	require.NoError(t, err)

	// Keep a receiver waiting while cancellation unblocks Scanner. Without an
	// explicit context check, the producer can randomly choose to send the read
	// error instead of observing the canceled context.
	for range 64 {
		readStarted := make(chan struct{})
		llm.GetHTTPClient().Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body: &cancelReadCloser{
					ctx:     req.Context(),
					started: readStarted,
				},
				Request: req,
			}, nil
		})

		stream, err := llm.StreamGenerate(context.Background(), "prompt")
		require.NoError(t, err)
		requireChannelSignal(t, readStarted, "stream body was not read")

		type receiveResult struct {
			chunk core.StreamChunk
			ok    bool
		}
		received := make(chan receiveResult, 1)
		go func() {
			chunk, ok := <-stream.ChunkChannel
			received <- receiveResult{chunk: chunk, ok: ok}
		}()

		stream.Cancel()
		select {
		case result := <-received:
			if result.ok {
				require.NoError(t, result.chunk.Error, "explicit cancellation must not surface as a stream read failure")
				t.Fatalf("received an unexpected chunk after cancellation: %+v", result.chunk)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("stream did not close after cancellation")
		}
	}
}

func requireChannelSignal(t *testing.T, ch <-chan struct{}, message string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatal(message)
	}
}

func requireStreamClosed(t *testing.T, chunks <-chan core.StreamChunk) {
	t.Helper()
	select {
	case _, ok := <-chunks:
		require.False(t, ok, "stream channel remained open")
	case <-time.After(2 * time.Second):
		t.Fatal("stream channel did not close")
	}
}
