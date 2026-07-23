package agents

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeToolResult_UsesDualChannelMetadataWhenPresent(t *testing.T) {
	observation := NormalizeToolResult(core.ToolResult{
		Data: "ignored",
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   "model text",
			core.ToolResultDisplayTextMeta: "display text",
			core.ToolResultIsErrorMeta:     true,
			core.ToolResultSyntheticMeta:   true,
			core.ToolResultRedactedMeta:    true,
			core.ToolResultTruncatedMeta:   true,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: map[string]any{"command": "pytest"},
		},
	})

	assert.Equal(t, "model text", observation.ModelText)
	assert.Equal(t, "display text", observation.DisplayText)
	assert.Equal(t, map[string]any{"command": "pytest"}, observation.Details)
	assert.True(t, observation.IsError)
	assert.True(t, observation.Synthetic)
	assert.True(t, observation.Redacted)
	assert.True(t, observation.Truncated)
}

func TestNormalizeToolResult_PreservesEmptyTextOverrides(t *testing.T) {
	observation := NormalizeToolResult(core.ToolResult{
		Data: "sensitive raw output",
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   "",
			core.ToolResultDisplayTextMeta: "",
		},
	})

	assert.Empty(t, observation.ModelText)
	assert.Empty(t, observation.DisplayText)
}

func TestNormalizeToolResult_FallsBackToDisplayAndData(t *testing.T) {
	observation := NormalizeToolResult(core.ToolResult{
		Data: "raw output",
	})
	assert.Equal(t, "raw output", observation.ModelText)
	assert.Equal(t, "raw output", observation.DisplayText)

	observation = NormalizeToolResult(core.ToolResult{
		Data: "raw output",
		Metadata: map[string]any{
			core.ToolResultDisplayTextMeta: "display only",
		},
	})
	assert.Equal(t, "display only", observation.ModelText)
	assert.Equal(t, "display only", observation.DisplayText)
}

func TestNormalizeToolResult_RecognizesCanonicalAndLegacyErrors(t *testing.T) {
	tests := []struct {
		name        string
		metadata    map[string]any
		annotations map[string]any
	}{
		{name: "canonical metadata", metadata: map[string]any{core.ToolResultIsErrorMeta: true}},
		{name: "legacy metadata", metadata: map[string]any{"isError": true}},
		{name: "snake case annotation", annotations: map[string]any{"is_error": true}},
		{name: "camel case annotation", annotations: map[string]any{"isError": true}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			observation := NormalizeToolResult(core.ToolResult{
				Data:        "failed",
				Metadata:    tt.metadata,
				Annotations: tt.annotations,
			})
			assert.True(t, observation.IsError)
		})
	}
}

func TestBlockedToolObservation_IsSynthetic(t *testing.T) {
	observation := BlockedToolObservation("bash", "approval denied")
	require.NotEmpty(t, observation.ModelText)
	assert.Equal(t, observation.ModelText, observation.DisplayText)
	assert.True(t, observation.IsError)
	assert.True(t, observation.Synthetic)
	assert.Equal(t, "approval denied", observation.Details["reason"])
}
