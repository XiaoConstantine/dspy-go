package errors

import (
	"context"
	stderrors "errors"
	"fmt"
	"maps"
	"sort"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
)

var (
	// ErrRefused indicates that the model refused to fulfill the request.
	ErrRefused = &Error{code: Refused, message: "model refused request"}
	// ErrContentFiltered indicates that the provider withheld output due to a content filter.
	ErrContentFiltered = &Error{code: ContentFiltered, message: "response filtered by content safety policy"}
	// ErrTruncated indicates that generation stopped because the output token limit was reached.
	ErrTruncated = &Error{code: Truncated, message: "response truncated by token limit"}
	// ErrParseFailed indicates that the model's output could not be parsed or validated against the signature.
	ErrParseFailed = &Error{code: ParseFailed, message: "failed to parse structured output"}
)

// RefusalError represents a model refusal returned by a provider.
type RefusalError struct {
	Reason string
}

func (e *RefusalError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if strings.TrimSpace(e.Reason) == "" {
		return ErrRefused.message
	}
	return fmt.Sprintf("%s: %s", ErrRefused.message, e.Reason)
}

func (e *RefusalError) Code() ErrorCode { return Refused }

func (e *RefusalError) Is(target error) bool {
	if target == ErrRefused {
		return true
	}
	if te, ok := target.(*Error); ok {
		return te.code == Refused
	}
	return false
}

// ContentFilterError represents a response stopped by a provider content filter.
type ContentFilterError struct {
	Raw string
}

func (e *ContentFilterError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return ErrContentFiltered.message
}

func (e *ContentFilterError) Code() ErrorCode { return ContentFiltered }

func (e *ContentFilterError) Is(target error) bool {
	if target == ErrContentFiltered {
		return true
	}
	if te, ok := target.(*Error); ok {
		return te.code == ContentFiltered
	}
	return false
}

// TruncatedError represents a response cut off by the token limit (finish_reason=length).
type TruncatedError struct {
	PartialFields map[string]any
	Raw           string
}

func (e *TruncatedError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if len(e.PartialFields) > 0 {
		return fmt.Sprintf("%s (%d partial field(s) extracted)", ErrTruncated.message, len(e.PartialFields))
	}
	return ErrTruncated.message
}

func (e *TruncatedError) Code() ErrorCode { return Truncated }

func (e *TruncatedError) Is(target error) bool {
	if target == ErrTruncated {
		return true
	}
	if te, ok := target.(*Error); ok {
		return te.code == Truncated
	}
	return false
}

// ParseError reports a structured-output parsing or validation failure.
type ParseError struct {
	Adapter       string
	MissingFields []string
	InvalidFields map[string]string
	Raw           string
	Message       string
	Err           error
}

func (e *ParseError) Error() string {
	if e == nil {
		return "<nil>"
	}
	var parts []string
	if e.Adapter != "" {
		parts = append(parts, e.Adapter)
	}
	if e.Message != "" {
		parts = append(parts, e.Message)
	}
	if len(e.MissingFields) > 0 {
		parts = append(parts, fmt.Sprintf("missing fields: %s", strings.Join(e.MissingFields, ", ")))
	}
	if len(e.InvalidFields) > 0 {
		keys := make([]string, 0, len(e.InvalidFields))
		for k := range e.InvalidFields {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		details := make([]string, 0, len(keys))
		for _, k := range keys {
			details = append(details, fmt.Sprintf("%s (%s)", k, e.InvalidFields[k]))
		}
		parts = append(parts, fmt.Sprintf("invalid fields: %s", strings.Join(details, "; ")))
	}
	if e.Err != nil {
		parts = append(parts, e.Err.Error())
	}
	if len(parts) == 0 {
		return ErrParseFailed.message
	}
	return strings.Join(parts, ": ")
}

func (e *ParseError) Code() ErrorCode { return ParseFailed }

func (e *ParseError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func (e *ParseError) Is(target error) bool {
	if target == ErrParseFailed {
		return true
	}
	if te, ok := target.(*Error); ok {
		return te.code == ParseFailed
	}
	return false
}

// StreamParseError reports a failure during streaming after at least one chunk
// was already emitted to the stream handler, preventing a silent fallback replay.
type StreamParseError struct {
	PartialFields map[string]any
	FinishReason  string
	Raw           string
	Err           error
}

func (e *StreamParseError) Error() string {
	if e == nil {
		return "<nil>"
	}
	var parts []string
	parts = append(parts, "stream parse failed after emitting output")
	if e.FinishReason != "" {
		parts = append(parts, fmt.Sprintf("finish_reason=%s", e.FinishReason))
	}
	if e.Err != nil {
		parts = append(parts, e.Err.Error())
	}
	return strings.Join(parts, ": ")
}

func (e *StreamParseError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// ClonePartialFields returns a shallow copy of partial fields, or nil when empty.
func ClonePartialFields(fields map[string]any) map[string]any {
	if len(fields) == 0 {
		return nil
	}
	return maps.Clone(fields)
}

// IsOutputError reports whether err is a model output failure (refusal, content
// filter, truncation, or parse/validation failure) rather than a transport or
// configuration failure.
func IsOutputError(err error) bool {
	if err == nil {
		return false
	}
	var parseErr *ParseError
	var streamParseErr *StreamParseError
	var truncatedErr *TruncatedError
	var refusalErr *RefusalError
	var filterErr *ContentFilterError
	if stderrors.As(err, &parseErr) ||
		stderrors.As(err, &streamParseErr) ||
		stderrors.As(err, &truncatedErr) ||
		stderrors.As(err, &refusalErr) ||
		stderrors.As(err, &filterErr) {
		return true
	}
	return stderrors.Is(err, ErrRefused) ||
		stderrors.Is(err, ErrContentFiltered) ||
		stderrors.Is(err, ErrTruncated) ||
		stderrors.Is(err, ErrParseFailed)
}

// Retryable reports whether retrying the exact same request at the transport
// layer is appropriate. Output errors, refusals, truncations, content filters,
// configuration/input errors, and context cancellations are never retryable.
func Retryable(err error) bool {
	if err == nil {
		return false
	}
	if stderrors.Is(err, context.Canceled) || stderrors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if IsOutputError(err) {
		return false
	}

	var llmErr *llm.Error
	if stderrors.As(err, &llmErr) {
		if llmErr.Retryable != nil {
			return *llmErr.Retryable
		}
		switch llmErr.Kind {
		case llm.KindRateLimit, llm.KindTransport, llm.KindProvider:
			return true
		case llm.KindInvalidRequest, llm.KindAuthentication, llm.KindPermission,
			llm.KindContextLimit, llm.KindUnsupported, llm.KindMalformedResponse:
			return false
		}
	}

	var dspyErr *Error
	if stderrors.As(err, &dspyErr) {
		switch dspyErr.code {
		case RateLimitExceeded, Timeout:
			return true
		case InvalidInput, ValidationFailed, ResourceNotFound, Canceled,
			TokenLimitExceeded, InvalidResponse, ModelNotSupported, ProviderNotFound,
			ConfigurationError, UnsupportedOperation,
			Refused, ContentFiltered, Truncated, ParseFailed:
			return false
		case LLMGenerationFailed:
			// A generation failure is retryable unless it wraps a cause that is not.
			return dspyErr.original == nil || Retryable(dspyErr.original)
		}
	}

	// Unknown external errors (e.g. raw net.Error or transient HTTP failures)
	// remain retryable unless classified otherwise.
	return true
}
