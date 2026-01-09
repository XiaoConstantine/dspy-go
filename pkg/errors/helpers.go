package errors

import (
	"context"
)

// CheckContext returns an error if the context is canceled or timed out.
// This provides a standardized way to check and wrap context errors.
func CheckContext(ctx context.Context, operation string) error {
	if err := ctx.Err(); err != nil {
		return Wrap(err, Canceled, operation+" canceled")
	}
	return nil
}
