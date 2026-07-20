package tools

import (
	"sort"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

const defaultFinishToolDescription = "Call this tool when you have completed the task and have the final answer."

// BuildFunctionSchemas converts tools in a registry into provider-agnostic
// function schema maps suitable for LLM tool-calling APIs.
func BuildFunctionSchemas(registry core.ToolRegistry) ([]map[string]any, error) {
	if registry == nil {
		return nil, nil
	}

	registered := registry.List()
	sort.Slice(registered, func(i, j int) bool {
		return registered[i].Name() < registered[j].Name()
	})

	functions := make([]map[string]any, 0, len(registered))
	for _, tool := range registered {
		schema := tool.InputSchema()

		required := make([]string, 0, len(schema.Properties))
		properties := make(map[string]any, len(schema.Properties))
		for name, paramSchema := range schema.Properties {
			properties[name] = map[string]any{
				"type":        paramSchema.Type,
				"description": paramSchema.Description,
			}
			if paramSchema.Required {
				required = append(required, name)
			}
		}
		sort.Strings(required)

		schemaType := schema.Type
		if schemaType == "" {
			schemaType = "object"
		}

		function := map[string]any{
			"name":        tool.Name(),
			"description": tool.Description(),
			"parameters": map[string]any{
				"type":       schemaType,
				"properties": properties,
				"required":   required,
			},
		}
		functions = append(functions, function)
	}

	return functions, nil
}

// BuildFinishFunctionSchema returns the standard finish sentinel schema used by
// higher-level loops that need an explicit completion tool.
func BuildFinishFunctionSchema(description string) map[string]any {
	if description == "" {
		description = defaultFinishToolDescription
	}

	return map[string]any{
		"name":        "Finish",
		"description": description,
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{
					"type":        "string",
					"description": "The final answer or result of the task",
				},
				"reasoning": map[string]any{
					"type":        "string",
					"description": "Brief explanation of how the answer was derived",
				},
			},
			"required": []string{"answer"},
		},
	}
}
