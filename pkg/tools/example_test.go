package tools_test

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

func ExampleNewFuncTool() {
	greet := tools.NewFuncTool(
		"greet",
		"Create a greeting",
		models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"name": {
					Type:        "string",
					Description: "Name to greet",
					Required:    true,
				},
			},
		},
		func(ctx context.Context, args map[string]any) (*models.CallToolResult, error) {
			text := fmt.Sprintf("Hello, %s!", args["name"])
			return &models.CallToolResult{
				Content: []models.Content{
					models.TextContent{Type: "text", Text: text},
				},
			}, nil
		},
	)

	args := map[string]any{"name": "Ada"}
	if err := greet.Validate(args); err != nil {
		fmt.Println("validation error:", err)
		return
	}
	result, err := greet.Execute(context.Background(), args)
	if err != nil {
		fmt.Println("execution error:", err)
		return
	}
	fmt.Println(result.Data)

	// Output:
	// Hello, Ada!
}
