package tools

import (
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// InMemoryToolRegistry provides a basic in-memory implementation of the ToolRegistry interface.
type InMemoryToolRegistry struct {
	mu    sync.RWMutex
	tools map[string]core.Tool
}

// NewInMemoryToolRegistry creates a new, empty InMemoryToolRegistry.
func NewInMemoryToolRegistry() *InMemoryToolRegistry {
	return &InMemoryToolRegistry{
		tools: make(map[string]core.Tool),
	}
}

// Register adds a tool to the registry.
// It returns an error if a tool with the same name already exists.
func (r *InMemoryToolRegistry) Register(tool core.Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if tool == nil {
		return errors.New(errors.InvalidInput, "cannot register a nil tool")
	}

	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		return errors.WithFields(errors.New(errors.InvalidInput, "tool already registered"), errors.Fields{
			"tool_name": name,
		})
	}

	r.tools[name] = tool
	return nil
}

// Get retrieves a tool by its name.
// It returns an error if the tool is not found.
func (r *InMemoryToolRegistry) Get(name string) (core.Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, exists := r.tools[name]
	if !exists {
		return nil, errors.WithFields(errors.New(errors.ResourceNotFound, "tool not found"), errors.Fields{
			"tool_name": name,
		})
	}
	return tool, nil
}

// List returns a slice of all registered tools.
// The order is not guaranteed.
func (r *InMemoryToolRegistry) List() []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	list := make([]core.Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		list = append(list, tool)
	}
	return list
}

// Match finds tools that might match a given intent string.
// This basic implementation checks if the intent contains the tool name (case-insensitive).
// More sophisticated matching (e.g., using descriptions or CanHandle) could be added.
func (r *InMemoryToolRegistry) Match(intent string) []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var matches []core.Tool
	lowerIntent := strings.ToLower(intent)

	for name, tool := range r.tools {
		// Simple substring match on name
		if strings.Contains(lowerIntent, strings.ToLower(name)) {
			matches = append(matches, tool)
			continue
		}
	}
	return matches
}

// Ensure InMemoryToolRegistry implements the interface.
var _ core.ToolRegistry = (*InMemoryToolRegistry)(nil)
