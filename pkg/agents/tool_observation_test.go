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

func TestBlockedToolObservation_IsSynthetic(t *testing.T) {
	observation := BlockedToolObservation("bash", "approval denied")
	require.NotEmpty(t, observation.ModelText)
	assert.Equal(t, observation.ModelText, observation.DisplayText)
	assert.True(t, observation.IsError)
	assert.True(t, observation.Synthetic)
	assert.Equal(t, "approval denied", observation.Details["reason"])
}
