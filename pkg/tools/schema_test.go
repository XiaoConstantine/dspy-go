package tools

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type schemaTestTool struct {
	name        string
	description string
	schema      models.InputSchema
}

func (m *schemaTestTool) Name() string        { return m.name }
func (m *schemaTestTool) Description() string { return m.description }
func (m *schemaTestTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:        m.name,
		Description: m.description,
		InputSchema: m.schema,
	}
}
func (m *schemaTestTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (m *schemaTestTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{Data: "ok"}, nil
}
func (m *schemaTestTool) Validate(params map[string]interface{}) error { return nil }
func (m *schemaTestTool) InputSchema() models.InputSchema              { return m.schema }

func TestBuildFunctionSchemas(t *testing.T) {
	registry := NewInMemoryToolRegistry()
	err := registry.Register(&schemaTestTool{
		name:        "search",
		description: "Search for information",
		schema: models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"query": {
					Type:        "string",
					Description: "Search query",
					Required:    true,
				},
				"limit": {
					Type:        "integer",
					Description: "Result limit",
					Required:    false,
				},
			},
		},
	})
	require.NoError(t, err)

	functions, err := BuildFunctionSchemas(registry)
	require.NoError(t, err)
	require.Len(t, functions, 1)

	f := functions[0]
	assert.Equal(t, "search", f["name"])
	assert.Equal(t, "Search for information", f["description"])

	parameters, ok := f["parameters"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "object", parameters["type"])

	required, ok := parameters["required"].([]string)
	require.True(t, ok)
	assert.Equal(t, []string{"query"}, required)

	properties, ok := parameters["properties"].(map[string]interface{})
	require.True(t, ok)
	_, hasQuery := properties["query"]
	_, hasLimit := properties["limit"]
	assert.True(t, hasQuery)
	assert.True(t, hasLimit)
}

func TestBuildFunctionSchemas_DefaultsAndNilRegistry(t *testing.T) {
	functions, err := BuildFunctionSchemas(nil)
	require.NoError(t, err)
	assert.Nil(t, functions)

	registry := NewInMemoryToolRegistry()
	err = registry.Register(&schemaTestTool{
		name:        "echo",
		description: "Echo input",
		schema: models.InputSchema{
			Properties: map[string]models.ParameterSchema{},
		},
	})
	require.NoError(t, err)

	functions, err = BuildFunctionSchemas(registry)
	require.NoError(t, err)
	require.Len(t, functions, 1)
	params := functions[0]["parameters"].(map[string]interface{})
	assert.Equal(t, "object", params["type"])
}

func TestBuildFinishFunctionSchema(t *testing.T) {
	custom := BuildFinishFunctionSchema("Use to finish now")
	assert.Equal(t, "Finish", custom["name"])
	assert.Equal(t, "Use to finish now", custom["description"])

	def := BuildFinishFunctionSchema("")
	assert.Equal(t, "Finish", def["name"])
	assert.NotEmpty(t, def["description"])
}
