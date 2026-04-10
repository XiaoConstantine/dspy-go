package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

func TestNewToolInfoFromTool_AllowsNilMetadata(t *testing.T) {
	tool := runtimeStubTool{
		name:        "nil-meta",
		description: "nil-meta",
		schema:      models.InputSchema{Type: "object"},
	}

	info := NewToolInfoFromTool(tool)
	require.NotNil(t, info)
	assert.Equal(t, "nil-meta", info.Name)
	assert.Equal(t, "nil-meta", info.Description)
	assert.Equal(t, "1.0.0", info.Version)
	assert.Empty(t, info.Capabilities)
}

func TestNewToolInfoFromTool_UsesMetadataWhenPresent(t *testing.T) {
	tool := runtimeStubTool{
		name:        "meta",
		description: "tool-description",
		schema:      models.InputSchema{Type: "object"},
		metadata: &ToolMetadata{
			Name:         "metadata-name",
			Description:  "metadata-description",
			Version:      "2.0.0",
			Capabilities: []string{"read", "write"},
		},
	}

	info := NewToolInfoFromTool(tool)
	require.NotNil(t, info)
	assert.Equal(t, "meta", info.Name)
	assert.Equal(t, "tool-description", info.Description)
	assert.Equal(t, "2.0.0", info.Version)
	assert.Equal(t, []string{"read", "write"}, info.Capabilities)
}

type runtimeStubTool struct {
	name        string
	description string
	schema      models.InputSchema
	metadata    *ToolMetadata
}

func (t runtimeStubTool) Name() string { return t.name }

func (t runtimeStubTool) Description() string {
	if t.description != "" {
		return t.description
	}
	return t.name
}

func (t runtimeStubTool) Metadata() *ToolMetadata { return t.metadata }

func (t runtimeStubTool) CanHandle(context.Context, string) bool { return false }

func (t runtimeStubTool) Execute(context.Context, map[string]interface{}) (ToolResult, error) {
	return ToolResult{}, nil
}

func (t runtimeStubTool) Validate(map[string]interface{}) error { return nil }

func (t runtimeStubTool) InputSchema() models.InputSchema { return t.schema }
