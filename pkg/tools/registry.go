package tools

import (
	"context"
	"fmt"
	"sync"

	"github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Registry manages a collection of tools and provides methods for registration,
// retrieval, and invocation. It safely handles concurrent access to tools.
type Registry struct {
	tools map[string]Tool
	mu    sync.RWMutex
}

// NewRegistry creates a new empty tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry.
// Returns an error if a tool with the same name already exists.
func (r *Registry) Register(tool Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool with name %q already exists", name)
	}

	r.tools[name] = tool
	return nil
}

// Get retrieves a tool by name.
// Returns an error if the tool doesn't exist.
func (r *Registry) Get(name string) (Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, exists := r.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool with name %q not found", name)
	}

	return tool, nil
}

// List returns all registered tools.
func (r *Registry) List() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}

	return tools
}

// Unregister removes a tool from the registry.
// Returns an error if the tool doesn't exist.
func (r *Registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[name]; !exists {
		return fmt.Errorf("tool with name %q not found", name)
	}

	delete(r.tools, name)
	return nil
}

// Call invokes a tool by name with the given arguments.
// This provides a convenient way to call tools without retrieving them first.
func (r *Registry) Call(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error) {
	tool, err := r.Get(name)
	if err != nil {
		return nil, err
	}

	return tool.Call(ctx, args)
}
