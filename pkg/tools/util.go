package tools

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	pkgErrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	"github.com/XiaoConstantine/mcp-go/pkg/logging"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/XiaoConstantine/mcp-go/pkg/transport"
)

// This improves testability by allowing mocks.
type MCPClientInterface interface {
	ListTools(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error)
	// Add CallTool if the wrapper's Execute needs it indirectly via the interface
	CallTool(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error)
}

// --- Dynamic MCP Tool Wrapper ---
// mcpCoreToolWrapper dynamically wraps an MCP tool to satisfy the core.Tool interface.
type mcpCoreToolWrapper struct {
	name        string
	description string
	schema      models.InputSchema
	// Use the interface type now for flexibility and testing
	mcpClient MCPClientInterface
}

func (t *mcpCoreToolWrapper) Name() string {
	return t.name
}

func (t *mcpCoreToolWrapper) Description() string {
	return t.description
}

func (t *mcpCoreToolWrapper) InputSchema() models.InputSchema {
	return t.schema
}

func (t *mcpCoreToolWrapper) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:        t.name,
		Description: t.description,
		InputSchema: t.schema}
}

func (t *mcpCoreToolWrapper) CanHandle(ctx context.Context, intent string) bool {
	// Simple name matching for now
	return intent == t.name
}

func (t *mcpCoreToolWrapper) Validate(params map[string]interface{}) error {
	// Basic validation - could potentially delegate to MCP server or use schema
	// For now, assume valid if Execute can handle it.
	return nil
}

func (t *mcpCoreToolWrapper) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	if t.mcpClient == nil {
		return core.ToolResult{}, pkgErrors.New(pkgErrors.InvalidWorkflowState, "MCP client is not set for tool wrapper")
	}

	// Call the actual MCP tool via the client interface
	mcpResult, err := t.mcpClient.CallTool(ctx, t.name, params)
	if err != nil {
		// Wrap the error for context
		return core.ToolResult{}, pkgErrors.Wrap(err, pkgErrors.StepExecutionFailed, fmt.Sprintf("failed to execute MCP tool '%s'", t.name))
	}

	// Convert MCP result to core.ToolResult
	// Simple conversion: extract text content. Could be more sophisticated.
	resultData := extractContentText(mcpResult.Content)
	resultAnnotations := make(map[string]interface{})
	if mcpResult.IsError {
		resultAnnotations["mcp_error"] = true
	}

	return core.ToolResult{
		Data:        resultData,
		Annotations: resultAnnotations,
	}, nil
}

// Ensure mcpCoreToolWrapper implements core.Tool.
var _ core.Tool = (*mcpCoreToolWrapper)(nil)

// MCPClientOptions contains configuration options for creating an MCP client.
type MCPClientOptions struct {
	ClientName    string
	ClientVersion string
	Logger        logging.Logger
}

// NewMCPClientFromStdio creates a new MCP client using standard I/O for communication.
// This is useful for connecting to an MCP server launched as a subprocess.
func NewMCPClientFromStdio(reader io.Reader, writer io.Writer, options MCPClientOptions) (*client.Client, error) {
	logger := options.Logger
	if logger == nil {
		logger = logging.NewStdLogger(logging.InfoLevel)
	}

	t := transport.NewStdioTransport(reader, writer, logger)

	clientOptions := []client.Option{
		client.WithLogger(logger),
	}

	if options.ClientName != "" && options.ClientVersion != "" {
		clientOptions = append(clientOptions, client.WithClientInfo(options.ClientName, options.ClientVersion))
	}

	mcpClient := client.NewClient(t, clientOptions...)

	// Initialize the client with a reasonable timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := mcpClient.Initialize(ctx)
	if err != nil {
		return nil, err
	}

	return mcpClient, nil
}

// RegisterMCPTools dynamically discovers tools from an MCP client and registers them
// into a core.ToolRegistry, wrapping them to conform to the core.Tool interface.
func RegisterMCPTools(registry core.ToolRegistry, mcpClient MCPClientInterface) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	toolsResult, err := mcpClient.ListTools(ctx, nil)
	if err != nil {
		// Wrap the error for context
		return pkgErrors.Wrap(err, pkgErrors.ResourceNotFound, "failed to list tools from MCP client")
	}

	for _, mcpTool := range toolsResult.Tools {
		// Create the dynamic wrapper
		wrapper := &mcpCoreToolWrapper{
			name:        mcpTool.Name,
			description: mcpTool.Description,
			schema:      mcpTool.InputSchema,
			mcpClient:   mcpClient,
		}

		if err := registry.Register(wrapper); err != nil {
			// If registration fails (e.g., duplicate), wrap the error
			return pkgErrors.Wrap(err, pkgErrors.InvalidInput, fmt.Sprintf("failed to register discovered MCP tool '%s'", mcpTool.Name))
		}
	}

	return nil
}

// Helper function to extract text content from MCP Content array.
func extractContentText(content []models.Content) string {
	var result strings.Builder

	for _, item := range content {
		if textContent, ok := item.(models.TextContent); ok {
			if result.Len() > 0 {
				result.WriteString("\n") // Add newline between multiple text parts
			}
			result.WriteString(textContent.Text)
		}
		// TODO: Handle other content types (e.g., JSON) if necessary
	}

	return result.String()
}

// Helper function to extract capabilities from description.
// NOTE: This is a very basic heuristic and might not be reliable.
func extractCapabilities(description string) []string {
	capabilities := []string{}

	keywords := []string{"search", "query", "calculate", "fetch", "retrieve",
		"find", "create", "update", "delete", "git", "status",
		"commit", "repository", "branch", "read", "write", "list", "run", "edit"}

	descLower := strings.ToLower(description)
	for _, keyword := range keywords {
		if strings.Contains(descLower, keyword) {
			capabilities = append(capabilities, keyword)
		}
	}

	return capabilities
}

// calculateToolMatchScore determines how well a tool matches an action.
// NOTE: This is a simple heuristic and might not be optimal.
func calculateToolMatchScore(metadata *core.ToolMetadata, action string) float64 {
	score := 0.1 // Base score
	actionLower := strings.ToLower(action)

	// Check if action mentions tool name
	if strings.Contains(actionLower, strings.ToLower(metadata.Name)) {
		score += 0.5
	}

	// Check capabilities (if available in metadata)
	// Ensure metadata.Capabilities is not nil before iterating
	if metadata.Capabilities != nil {
		for _, capability := range metadata.Capabilities {
			if strings.Contains(actionLower, strings.ToLower(capability)) {
				score += 0.3
			}
		}
	}

	return score
}
