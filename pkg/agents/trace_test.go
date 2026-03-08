package agents

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionTraceClone_PreservesStepSliceShape(t *testing.T) {
	t.Run("preserves nil steps", func(t *testing.T) {
		trace := &ExecutionTrace{}

		cloned := trace.Clone()
		require.NotNil(t, cloned)
		assert.Nil(t, cloned.Steps)
	})

	t.Run("preserves empty non-nil steps", func(t *testing.T) {
		trace := &ExecutionTrace{
			Steps: []TraceStep{},
		}

		cloned := trace.Clone()
		require.NotNil(t, cloned)
		require.NotNil(t, cloned.Steps)
		assert.Len(t, cloned.Steps, 0)
	})
}
