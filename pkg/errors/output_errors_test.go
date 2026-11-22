package errors

import (
	"context"
	stderrors "errors"
	"fmt"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutputErrorsMatchingAndRetryability(t *testing.T) {
	t.Run("refusal error", func(t *testing.T) {
		err := &RefusalError{Reason: "cannot comply"}
		assert.True(t, stderrors.Is(err, ErrRefused))
		assert.True(t, IsOutputError(err))
		assert.False(t, Retryable(err))
		assert.Contains(t, err.Error(), "cannot comply")

		var got *RefusalError
		require.True(t, stderrors.As(fmt.Errorf("wrapped: %w", err), &got))
		assert.Equal(t, "cannot comply", got.Reason)
	})

	t.Run("content filter error", func(t *testing.T) {
		err := &ContentFilterError{Raw: "partial"}
		assert.True(t, stderrors.Is(err, ErrContentFiltered))
		assert.True(t, IsOutputError(err))
		assert.False(t, Retryable(err))
	})

	t.Run("truncated error", func(t *testing.T) {
		err := &TruncatedError{PartialFields: map[string]any{"answer": "cut"}, Raw: "<answer>cut"}
		assert.True(t, stderrors.Is(err, ErrTruncated))
		assert.True(t, IsOutputError(err))
		assert.False(t, Retryable(err))
	})

	t.Run("parse error", func(t *testing.T) {
		err := &ParseError{
			Adapter:       "xml",
			MissingFields: []string{"answer"},
			InvalidFields: map[string]string{"score": "expected integer"},
			Raw:           "<score>abc</score>",
		}
		assert.True(t, stderrors.Is(err, ErrParseFailed))
		assert.True(t, IsOutputError(err))
		assert.False(t, Retryable(err))
		assert.Contains(t, err.Error(), "missing fields: answer")
		assert.Contains(t, err.Error(), "score (expected integer)")
	})

	t.Run("stream parse error unwraps cause", func(t *testing.T) {
		cause := &TruncatedError{PartialFields: map[string]any{"answer": "par"}, Raw: "par"}
		err := &StreamParseError{
			PartialFields: map[string]any{"answer": "par"},
			FinishReason:  "length",
			Raw:           "par",
			Err:           cause,
		}
		assert.True(t, stderrors.Is(err, ErrTruncated))
		assert.True(t, IsOutputError(err))
		assert.False(t, Retryable(err))
	})

	t.Run("transport vs non-retryable llm-go errors", func(t *testing.T) {
		assert.True(t, Retryable(&llm.Error{Kind: llm.KindRateLimit}))
		assert.True(t, Retryable(&llm.Error{Kind: llm.KindTransport}))
		assert.True(t, Retryable(&llm.Error{Kind: llm.KindProvider}))
		assert.False(t, Retryable(&llm.Error{Kind: llm.KindMalformedResponse}))
		assert.False(t, Retryable(&llm.Error{Kind: llm.KindInvalidRequest}))
		assert.False(t, Retryable(&llm.Error{Kind: llm.KindAuthentication}))
		assert.False(t, Retryable(&llm.Error{Kind: llm.KindUnsupported}))
		assert.False(t, Retryable(context.Canceled))
		assert.False(t, Retryable(context.DeadlineExceeded))

		assert.True(t, Retryable(New(LLMGenerationFailed, "service unavailable")), "a generation failure with no known cause")
		assert.True(t, Retryable(Wrap(&llm.Error{Kind: llm.KindTransport}, LLMGenerationFailed, "generate")))
		assert.False(t, Retryable(Wrap(&llm.Error{Kind: llm.KindAuthentication}, LLMGenerationFailed, "generate")))

		explicitTrue := true
		explicitFalse := false
		assert.True(t, Retryable(&llm.Error{Kind: llm.KindInvalidRequest, Retryable: &explicitTrue}))
		assert.False(t, Retryable(&llm.Error{Kind: llm.KindRateLimit, Retryable: &explicitFalse}))
	})
}
