package core

import (
	"context"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// ModuleHandler represents the actual module process function.
type ModuleHandler func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error)

// AgentHandler represents the actual agent execute function.
type AgentHandler func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

// ToolHandler represents the actual tool execute function.
type ToolHandler func(ctx context.Context, args map[string]interface{}) (ToolResult, error)

// ModuleInterceptor wraps Module.Process() calls with additional functionality.
// It follows the gRPC interceptor pattern: the interceptor can inspect/modify the request,
// call the handler, and inspect/modify the response.
type ModuleInterceptor func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error)

// AgentInterceptor wraps Agent.Execute() calls with additional functionality.
// Similar to ModuleInterceptor but for agent operations.
type AgentInterceptor func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error)

// ToolInterceptor wraps Tool.Execute() calls with additional functionality.
// This allows interception of tool invocations for logging, validation, security, etc.
type ToolInterceptor func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error)

// ModuleInfo contains metadata about a module for interceptor use.
type ModuleInfo struct {
	// ModuleName is the human-readable name of the module instance
	ModuleName string

	// ModuleType is the category/type of the module (e.g., "ChainOfThought", "React")
	ModuleType string

	// Signature contains the input/output structure definition
	Signature Signature

	// Version is the module version for compatibility tracking
	Version string

	// Metadata contains additional module-specific information
	Metadata map[string]interface{}
}

// AgentInfo contains metadata about an agent for interceptor use.
type AgentInfo struct {
	// AgentID is the unique identifier for the agent instance
	AgentID string

	// AgentType is the category/type of the agent (e.g., "ReactiveAgent", "PlannerAgent")
	AgentType string

	// Capabilities lists the tools/capabilities available to this agent
	Capabilities []Tool

	// Version is the agent version for compatibility tracking
	Version string

	// Metadata contains additional agent-specific information
	Metadata map[string]interface{}
}

// ToolInfo contains metadata about a tool for interceptor use.
type ToolInfo struct {
	// Name is the unique identifier for the tool
	Name string

	// Description is a human-readable explanation of the tool's purpose
	Description string

	// InputSchema defines the expected parameter structure
	InputSchema models.InputSchema

	// ToolType is the category/type of the tool (e.g., "MCPTool", "FunctionTool")
	ToolType string

	// Version is the tool version for compatibility tracking
	Version string

	// Capabilities lists the specific capabilities this tool provides
	Capabilities []string

	// Metadata contains additional tool-specific information
	Metadata map[string]interface{}
}

// InterceptorChain manages a collection of interceptors for a specific component type.
type InterceptorChain struct {
	moduleInterceptors []ModuleInterceptor
	agentInterceptors  []AgentInterceptor
	toolInterceptors   []ToolInterceptor
}

// NewInterceptorChain creates a new empty interceptor chain.
func NewInterceptorChain() *InterceptorChain {
	return &InterceptorChain{
		moduleInterceptors: make([]ModuleInterceptor, 0),
		agentInterceptors:  make([]AgentInterceptor, 0),
		toolInterceptors:   make([]ToolInterceptor, 0),
	}
}

// AddModuleInterceptor adds a module interceptor to the chain.
// Interceptors are applied in the order they are added (first added, first executed).
func (ic *InterceptorChain) AddModuleInterceptor(interceptor ModuleInterceptor) *InterceptorChain {
	ic.moduleInterceptors = append(ic.moduleInterceptors, interceptor)
	return ic
}

// AddAgentInterceptor adds an agent interceptor to the chain.
// Interceptors are applied in the order they are added (first added, first executed).
func (ic *InterceptorChain) AddAgentInterceptor(interceptor AgentInterceptor) *InterceptorChain {
	ic.agentInterceptors = append(ic.agentInterceptors, interceptor)
	return ic
}

// AddToolInterceptor adds a tool interceptor to the chain.
// Interceptors are applied in the order they are added (first added, first executed).
func (ic *InterceptorChain) AddToolInterceptor(interceptor ToolInterceptor) *InterceptorChain {
	ic.toolInterceptors = append(ic.toolInterceptors, interceptor)
	return ic
}

// GetModuleInterceptors returns a copy of the module interceptor slice.
func (ic *InterceptorChain) GetModuleInterceptors() []ModuleInterceptor {
	result := make([]ModuleInterceptor, len(ic.moduleInterceptors))
	copy(result, ic.moduleInterceptors)
	return result
}

// GetAgentInterceptors returns a copy of the agent interceptor slice.
func (ic *InterceptorChain) GetAgentInterceptors() []AgentInterceptor {
	result := make([]AgentInterceptor, len(ic.agentInterceptors))
	copy(result, ic.agentInterceptors)
	return result
}

// GetToolInterceptors returns a copy of the tool interceptor slice.
func (ic *InterceptorChain) GetToolInterceptors() []ToolInterceptor {
	result := make([]ToolInterceptor, len(ic.toolInterceptors))
	copy(result, ic.toolInterceptors)
	return result
}

// ChainModuleInterceptors combines multiple module interceptors into a single interceptor.
// The interceptors are applied in order: first interceptor in the slice is the outermost layer.
// If no interceptors are provided, returns a pass-through interceptor.
func ChainModuleInterceptors(interceptors ...ModuleInterceptor) ModuleInterceptor {
	n := len(interceptors)
	if n == 0 {
		// Return a pass-through interceptor that just calls the handler
		return func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	// Build the chain from the inside out
	return func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
		// Create a chain of handlers, starting with the final handler
		var chainer func(int, ModuleHandler) ModuleHandler
		chainer = func(currentIndex int, currentHandler ModuleHandler) ModuleHandler {
			if currentIndex == n {
				return currentHandler
			}
			return func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
				return interceptors[currentIndex](ctx, inputs, info, chainer(currentIndex+1, currentHandler), opts...)
			}
		}
		return chainer(0, handler)(ctx, inputs, opts...)
	}
}

// ChainAgentInterceptors combines multiple agent interceptors into a single interceptor.
// The interceptors are applied in order: first interceptor in the slice is the outermost layer.
// If no interceptors are provided, returns a pass-through interceptor.
func ChainAgentInterceptors(interceptors ...AgentInterceptor) AgentInterceptor {
	n := len(interceptors)
	if n == 0 {
		// Return a pass-through interceptor that just calls the handler
		return func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			return handler(ctx, input)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	// Build the chain from the inside out
	return func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
		// Create a chain of handlers, starting with the final handler
		var chainer func(int, AgentHandler) AgentHandler
		chainer = func(currentIndex int, currentHandler AgentHandler) AgentHandler {
			if currentIndex == n {
				return currentHandler
			}
			return func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
				return interceptors[currentIndex](ctx, input, info, chainer(currentIndex+1, currentHandler))
			}
		}
		return chainer(0, handler)(ctx, input)
	}
}

// ChainToolInterceptors combines multiple tool interceptors into a single interceptor.
// The interceptors are applied in order: first interceptor in the slice is the outermost layer.
// If no interceptors are provided, returns a pass-through interceptor.
func ChainToolInterceptors(interceptors ...ToolInterceptor) ToolInterceptor {
	n := len(interceptors)
	if n == 0 {
		// Return a pass-through interceptor that just calls the handler
		return func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			return handler(ctx, args)
		}
	}

	if n == 1 {
		return interceptors[0]
	}

	// Build the chain from the inside out
	return func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
		// Create a chain of handlers, starting with the final handler
		var chainer func(int, ToolHandler) ToolHandler
		chainer = func(currentIndex int, currentHandler ToolHandler) ToolHandler {
			if currentIndex == n {
				return currentHandler
			}
			return func(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
				return interceptors[currentIndex](ctx, args, info, chainer(currentIndex+1, currentHandler))
			}
		}
		return chainer(0, handler)(ctx, args)
	}
}

// Compose combines multiple interceptor chains into a single chain.
// This is useful for merging different sources of interceptors (e.g., global + module-specific).
func (ic *InterceptorChain) Compose(other *InterceptorChain) *InterceptorChain {
	if other == nil {
		return ic
	}

	result := NewInterceptorChain()

	// Add interceptors from the current chain first
	result.moduleInterceptors = append(result.moduleInterceptors, ic.moduleInterceptors...)
	result.agentInterceptors = append(result.agentInterceptors, ic.agentInterceptors...)
	result.toolInterceptors = append(result.toolInterceptors, ic.toolInterceptors...)

	// Then add interceptors from the other chain
	result.moduleInterceptors = append(result.moduleInterceptors, other.moduleInterceptors...)
	result.agentInterceptors = append(result.agentInterceptors, other.agentInterceptors...)
	result.toolInterceptors = append(result.toolInterceptors, other.toolInterceptors...)

	return result
}

// Clear removes all interceptors from the chain.
func (ic *InterceptorChain) Clear() *InterceptorChain {
	ic.moduleInterceptors = make([]ModuleInterceptor, 0)
	ic.agentInterceptors = make([]AgentInterceptor, 0)
	ic.toolInterceptors = make([]ToolInterceptor, 0)
	return ic
}

// IsEmpty returns true if the chain contains no interceptors.
func (ic *InterceptorChain) IsEmpty() bool {
	return len(ic.moduleInterceptors) == 0 &&
		len(ic.agentInterceptors) == 0 &&
		len(ic.toolInterceptors) == 0
}

// Count returns the total number of interceptors in the chain.
func (ic *InterceptorChain) Count() int {
	return len(ic.moduleInterceptors) + len(ic.agentInterceptors) + len(ic.toolInterceptors)
}

// CountByType returns the count of interceptors by type.
func (ic *InterceptorChain) CountByType() (modules, agents, tools int) {
	return len(ic.moduleInterceptors), len(ic.agentInterceptors), len(ic.toolInterceptors)
}

// NewModuleInfo creates a ModuleInfo with the provided details.
// This is a convenience function to ensure consistent ModuleInfo creation.
func NewModuleInfo(moduleName, moduleType string, signature Signature) *ModuleInfo {
	return &ModuleInfo{
		ModuleName: moduleName,
		ModuleType: moduleType,
		Signature:  signature,
		Version:    "1.0.0", // Default version
		Metadata:   make(map[string]interface{}),
	}
}

// NewAgentInfo creates an AgentInfo with the provided details.
// This is a convenience function to ensure consistent AgentInfo creation.
func NewAgentInfo(agentID, agentType string, capabilities []Tool) *AgentInfo {
	return &AgentInfo{
		AgentID:      agentID,
		AgentType:    agentType,
		Capabilities: capabilities,
		Version:      "1.0.0", // Default version
		Metadata:     make(map[string]interface{}),
	}
}

// NewToolInfo creates a ToolInfo with the provided details.
// This is a convenience function to ensure consistent ToolInfo creation.
func NewToolInfo(name, description, toolType string, inputSchema models.InputSchema) *ToolInfo {
	return &ToolInfo{
		Name:         name,
		Description:  description,
		InputSchema:  inputSchema,
		ToolType:     toolType,
		Version:      "1.0.0", // Default version
		Capabilities: make([]string, 0),
		Metadata:     make(map[string]interface{}),
	}
}

// WithMetadata adds metadata to ModuleInfo.
func (mi *ModuleInfo) WithMetadata(key string, value interface{}) *ModuleInfo {
	if mi.Metadata == nil {
		mi.Metadata = make(map[string]interface{})
	}
	mi.Metadata[key] = value
	return mi
}

// WithVersion sets the version for ModuleInfo.
func (mi *ModuleInfo) WithVersion(version string) *ModuleInfo {
	mi.Version = version
	return mi
}

// WithMetadata adds metadata to AgentInfo.
func (ai *AgentInfo) WithMetadata(key string, value interface{}) *AgentInfo {
	if ai.Metadata == nil {
		ai.Metadata = make(map[string]interface{})
	}
	ai.Metadata[key] = value
	return ai
}

// WithVersion sets the version for AgentInfo.
func (ai *AgentInfo) WithVersion(version string) *AgentInfo {
	ai.Version = version
	return ai
}

// WithMetadata adds metadata to ToolInfo.
func (ti *ToolInfo) WithMetadata(key string, value interface{}) *ToolInfo {
	if ti.Metadata == nil {
		ti.Metadata = make(map[string]interface{})
	}
	ti.Metadata[key] = value
	return ti
}

// WithVersion sets the version for ToolInfo.
func (ti *ToolInfo) WithVersion(version string) *ToolInfo {
	ti.Version = version
	return ti
}

// WithCapabilities sets the capabilities for ToolInfo.
func (ti *ToolInfo) WithCapabilities(capabilities ...string) *ToolInfo {
	ti.Capabilities = append(ti.Capabilities, capabilities...)
	return ti
}
