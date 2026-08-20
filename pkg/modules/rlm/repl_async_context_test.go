package rlm

import (
	"context"
	"fmt"
	"testing"
	"testing/synctest"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

type asyncContextKey struct{}

type contextCapturingSubLLMClient struct {
	values chan string
}

func (c *contextCapturingSubLLMClient) Query(ctx context.Context, _ string) (QueryResponse, error) {
	value, _ := ctx.Value(asyncContextKey{}).(string)
	c.values <- value
	return QueryResponse{Response: value}, nil
}

func (c *contextCapturingSubLLMClient) QueryBatched(context.Context, []string) ([]QueryResponse, error) {
	panic("unexpected QueryBatched call")
}

func TestYaegiREPLAsyncQuerySnapshotsExecutionContext(t *testing.T) {
	testutil.CheckGoroutineLeaks(t)

	for _, tt := range []struct {
		name  string
		start func(*YaegiREPL, context.Context) (*AsyncQueryHandle, error)
	}{
		{
			name: "Go API",
			start: func(repl *YaegiREPL, ctx context.Context) (*AsyncQueryHandle, error) {
				repl.SetContext(ctx)
				return repl.QueryAsync("prompt"), nil
			},
		},
		{
			name: "interpreter API",
			start: func(repl *YaegiREPL, ctx context.Context) (*AsyncQueryHandle, error) {
				if _, err := repl.Execute(ctx, `handleID := QueryAsync("prompt")`); err != nil {
					return nil, err
				}

				repl.asyncMu.RLock()
				defer repl.asyncMu.RUnlock()
				if len(repl.asyncQueries) != 1 {
					return nil, fmt.Errorf("async query count = %d, want 1", len(repl.asyncQueries))
				}
				for _, handle := range repl.asyncQueries {
					return handle, nil
				}
				return nil, fmt.Errorf("async query disappeared")
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				client := &contextCapturingSubLLMClient{values: make(chan string, 1)}
				repl, err := NewYaegiREPL(client)
				require.NoError(t, err)

				original := context.WithValue(context.Background(), asyncContextKey{}, "original")
				replacement := context.WithValue(context.Background(), asyncContextKey{}, "replacement")
				handle, err := tt.start(repl, original)
				require.NoError(t, err)

				// Changing the REPL context after the async operation starts must not
				// change the context observed by that operation.
				repl.SetContext(replacement)
				synctest.Wait()

				require.Equal(t, "original", <-client.values)
				result, err := handle.Wait()
				require.NoError(t, err)
				require.Equal(t, "original", result)
			})
		})
	}
}
