package main

import (
	"context"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Helper functions shared between examples.
func contains(text, substr string) bool {
	return len(text) >= len(substr) &&
		(text == substr ||
			len(text) > len(substr) &&
				(text[:len(substr)] == substr ||
					text[len(text)-len(substr):] == substr ||
					findInString(text, substr)))
}

func findInString(text, substr string) bool {
	for i := 0; i <= len(text)-len(substr); i++ {
		if text[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// SearchTool - Shared search tool implementation.
type SearchTool struct {
	name string
}

func (s *SearchTool) Name() string {
	return s.name
}

func (s *SearchTool) Description() string {
	return "Advanced search tool that can find information across multiple databases and APIs"
}

func (s *SearchTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         s.name,
		Description:  s.Description(),
		Capabilities: []string{"search", "query", "find", "lookup", "data_access"},
		Version:      "2.1.0",
	}
}

func (s *SearchTool) CanHandle(ctx context.Context, intent string) bool {
	keywords := []string{"search", "find", "query", "lookup", "discover"}
	for _, keyword := range keywords {
		if contains(intent, keyword) {
			return true
		}
	}
	return false
}

func (s *SearchTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	query, ok := params["query"].(string)
	if !ok {
		query = "default search"
	}

	// Simulate search operation
	time.Sleep(50 * time.Millisecond)

	results := []map[string]interface{}{
		{"id": 1, "title": "Result 1 for " + query, "relevance": 0.95},
		{"id": 2, "title": "Result 2 for " + query, "relevance": 0.87},
		{"id": 3, "title": "Result 3 for " + query, "relevance": 0.82},
	}

	return core.ToolResult{
		Data: map[string]interface{}{
			"results": results,
			"count":   len(results),
			"query":   query,
		},
		Metadata: map[string]interface{}{
			"execution_time_ms": 50,
			"source":            "advanced_search_api",
		},
	}, nil
}

func (s *SearchTool) Validate(params map[string]interface{}) error {
	return nil
}

func (s *SearchTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "The search query to execute",
				Required:    true,
			},
			"limit": {
				Type:        "integer",
				Description: "Maximum number of results to return",
			},
		},
	}
}
