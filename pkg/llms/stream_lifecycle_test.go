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
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				close(requestStarted)
				<-r.Context().Done()
				close(requestCanceled)
			}))
			defer server.Close()

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
