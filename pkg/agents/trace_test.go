package agents

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExecutionTraceClone_DeepCopiesNestedState(t *testing.T) {
	trace := &ExecutionTrace{
		Input:           map[string]any{"nested": map[string]any{"value": "input"}},
		Output:          map[string]any{"nested": []any{map[string]any{"value": "output"}}},
		ContextMetadata: map[string]any{"nested": map[string][]string{"values": {"context"}}},
		Steps: []TraceStep{{
			Arguments:          map[string]any{"nested": map[string]any{"value": "argument"}},
			ObservationDetails: map[string]any{"nested": []any{"detail"}},
			TokenUsage:         map[string]int64{"total_tokens": 3},
			Metadata:           map[string]any{"nested": map[string]any{"value": "metadata"}},
		}},
	}

	cloned := trace.Clone()
	cloned.Input["nested"].(map[string]any)["value"] = "changed"
	cloned.Output["nested"].([]any)[0].(map[string]any)["value"] = "changed"
	cloned.ContextMetadata["nested"].(map[string][]string)["values"][0] = "changed"
	cloned.Steps[0].Arguments["nested"].(map[string]any)["value"] = "changed"
	cloned.Steps[0].ObservationDetails["nested"].([]any)[0] = "changed"
	cloned.Steps[0].TokenUsage["total_tokens"] = 9
	cloned.Steps[0].Metadata["nested"].(map[string]any)["value"] = "changed"

	assert.Equal(t, "input", trace.Input["nested"].(map[string]any)["value"])
	assert.Equal(t, "output", trace.Output["nested"].([]any)[0].(map[string]any)["value"])
	assert.Equal(t, "context", trace.ContextMetadata["nested"].(map[string][]string)["values"][0])
	assert.Equal(t, "argument", trace.Steps[0].Arguments["nested"].(map[string]any)["value"])
	assert.Equal(t, "detail", trace.Steps[0].ObservationDetails["nested"].([]any)[0])
	assert.Equal(t, int64(3), trace.Steps[0].TokenUsage["total_tokens"])
	assert.Equal(t, "metadata", trace.Steps[0].Metadata["nested"].(map[string]any)["value"])
}

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
