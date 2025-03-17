package tools

import (
	"context"
	"io"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	"github.com/XiaoConstantine/mcp-go/pkg/logging"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/XiaoConstantine/mcp-go/pkg/transport"
)

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

// RegisterMCPTools discovers and registers all tools from an MCP server in the provided registry.
// This automatically bridges MCP tools to the local tool interface.
func RegisterMCPTools(registry *Registry, mcpClient *client.Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	toolsResult, err := mcpClient.ListTools(ctx, nil)
	if err != nil {
		return err
	}

	for _, mcpTool := range toolsResult.Tools {
		tool := NewMCPTool(
			mcpTool.Name,
			mcpTool.Description,
			mcpTool.InputSchema,
			mcpClient,
			mcpTool.Name,
		)

		if err := registry.Register(tool); err != nil {
			return err
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
				result.WriteString("\n")
			}
			result.WriteString(textContent.Text)
		}
	}

	return result.String()
}

// Helper function to extract capabilities from description.
func extractCapabilities(description string) []string {
	capabilities := []string{}

	keywords := []string{"search", "query", "calculate", "fetch", "retrieve",
		"find", "create", "update", "delete", "git", "status",
		"commit", "repository", "branch"}

	descLower := strings.ToLower(description)
	for _, keyword := range keywords {
		if strings.Contains(descLower, keyword) {
			capabilities = append(capabilities, keyword)
		}
	}

	return capabilities
}

// calculateToolMatchScore determines how well a tool matches an action.
func calculateToolMatchScore(metadata *core.ToolMetadata, action string) float64 {
	score := 0.1
	actionLower := strings.ToLower(action)

	// Check if action mentions tool name
	if strings.Contains(actionLower, strings.ToLower(metadata.Name)) {
		score += 0.5
	}

	// Check capabilities
	for _, capability := range metadata.Capabilities {
		if strings.Contains(actionLower, strings.ToLower(capability)) {
			score += 0.3
		}
	}

	return score
}
