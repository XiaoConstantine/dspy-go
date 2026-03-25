package subagent

import (
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// Registry stores named subagent tools for one-tool-per-subagent composition.
type Registry struct {
	mu    sync.RWMutex
	order []string
	tools map[string]core.Tool
}

func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]core.Tool),
	}
}

func (r *Registry) Register(name string, cfg ToolConfig) error {
	if r == nil {
		return fmt.Errorf("subagent registry is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	toolName := strings.TrimSpace(name)
	if toolName == "" {
		toolName = strings.TrimSpace(cfg.Name)
	}
	if toolName == "" {
		return fmt.Errorf("subagent registry entry requires a name")
	}
	if _, exists := r.tools[toolName]; exists {
		return fmt.Errorf("subagent %q already registered", toolName)
	}

	cfg.Name = toolName
	tool, err := AsTool(cfg)
	if err != nil {
		return err
	}

	r.order = append(r.order, toolName)
	r.tools[toolName] = tool
	return nil
}

func (r *Registry) Tool(name string) (core.Tool, error) {
	if r == nil {
		return nil, fmt.Errorf("subagent registry is nil")
	}
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, ok := r.tools[strings.TrimSpace(name)]
	if !ok {
		return nil, fmt.Errorf("subagent %q not registered", name)
	}
	return cloneTool(tool), nil
}

func (r *Registry) Tools() []core.Tool {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if len(r.order) == 0 {
		return nil
	}

	tools := make([]core.Tool, 0, len(r.order))
	for _, name := range r.order {
		tools = append(tools, cloneTool(r.tools[name]))
	}
	return tools
}

func cloneTool(tool core.Tool) core.Tool {
	if tool == nil {
		return nil
	}
	cloneable, ok := tool.(core.CloneableTool)
	if !ok {
		return tool
	}
	return cloneable.CloneTool()
}
