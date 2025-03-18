package tools

import (
	"context"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Define a minimal interface that has just the methods we need for our tests.
type MCPClientInterface interface {
	ListTools(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error)
	CallTool(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error)
}

// Ensure the mock client implements the necessary interface.
var _ MCPClientInterface = (*MockClient)(nil)

// Complete the MockClient with any other methods we need.
func (m *MockClient) Initialize(ctx context.Context) (*models.InitializeResult, error) {
	return &models.InitializeResult{}, nil
}

func (m *MockClient) Ping(ctx context.Context) error {
	return nil
}

func (m *MockClient) ListResources(ctx context.Context, cursor *models.Cursor) (*models.ListResourcesResult, error) {
	return &models.ListResourcesResult{}, nil
}

func (m *MockClient) ReadResource(ctx context.Context, uri string) (*models.ReadResourceResult, error) {
	return &models.ReadResourceResult{}, nil
}

func (m *MockClient) Subscribe(ctx context.Context, uri string) error {
	return nil
}

func (m *MockClient) Unsubscribe(ctx context.Context, uri string) error {
	return nil
}

func (m *MockClient) ListPrompts(ctx context.Context, cursor *models.Cursor) (*models.ListPromptsResult, error) {
	return &models.ListPromptsResult{}, nil
}

func (m *MockClient) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*models.GetPromptResult, error) {
	return &models.GetPromptResult{}, nil
}

func (m *MockClient) SetLogLevel(ctx context.Context, level models.LogLevel) error {
	return nil
}
