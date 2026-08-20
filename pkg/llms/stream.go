package llms

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

func sendStreamChunk(ctx context.Context, chunks chan<- core.StreamChunk, chunk core.StreamChunk) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

func invalidStreamJSON(provider, model string, err error) error {
	return errors.WithFields(
		errors.Wrap(err, errors.InvalidResponse, "failed to decode streaming response"),
		errors.Fields{"provider": provider, "model": model},
	)
}
