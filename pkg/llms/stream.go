package llms

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func sendStreamChunk(ctx context.Context, chunks chan<- core.StreamChunk, chunk core.StreamChunk) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}
