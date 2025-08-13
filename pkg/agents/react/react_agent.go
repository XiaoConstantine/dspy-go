package react

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
)

// ExecutionMode defines how the agent executes tasks.
type ExecutionMode int

const (
	// ModeReAct uses classic ReAct loop (Thought -> Action -> Observation).
	ModeReAct ExecutionMode = iota
	// ModeReWOO uses Reasoning Without Observation (plan-then-execute).
	ModeReWOO
	// ModeHybrid adaptively switches between ReAct and ReWOO.
	ModeHybrid
)

// PlanningStrategy defines how the agent plans tasks.
type PlanningStrategy int

const (
	// DecompositionFirst breaks down tasks upfront before execution.
	DecompositionFirst PlanningStrategy = iota
	// Interleaved adapts the plan during execution.
	Interleaved
)

// ReActAgentConfig configures the ReAct agent behavior.
type ReActAgentConfig struct {
	// Core settings
	MaxIterations int
	ExecutionMode ExecutionMode
	Timeout       time.Duration

	// Memory settings
	MemoryRetention time.Duration
	ForgetThreshold float64
	EnableMemoryOpt bool

	// Reflection settings
	EnableReflection bool
	ReflectionDepth  int
	ReflectionDelay  time.Duration

	// Planning settings
	PlanningStrategy PlanningStrategy
	MaxPlanDepth     int
	EnablePlanning   bool

	// Tool settings
	ToolTimeout   time.Duration
	ParallelTools bool
	MaxToolRetries int

	// Interceptor settings
	EnableInterceptors bool

	// XML parsing settings
	EnableXMLParsing bool
	XMLConfig        *interceptors.XMLConfig
}

// DefaultReActAgentConfig returns sensible defaults.
func DefaultReActAgentConfig() ReActAgentConfig {
	return ReActAgentConfig{
		MaxIterations:      10,
		ExecutionMode:      ModeReAct,
		Timeout:            5 * time.Minute,
		MemoryRetention:    24 * time.Hour,
		ForgetThreshold:    0.3,
		EnableMemoryOpt:    true,
		EnableReflection:   true,
		ReflectionDepth:    3,
		ReflectionDelay:    100 * time.Millisecond,
		PlanningStrategy:   Interleaved,
		MaxPlanDepth:       5,
		EnablePlanning:     true,
		ToolTimeout:        30 * time.Second,
		ParallelTools:      true,
		MaxToolRetries:     3,
		EnableInterceptors: true,
		EnableXMLParsing:   false, // Disabled by default for backward compatibility
		XMLConfig:          nil,
	}
}

// ReActAgent implements a generic ReAct-based agent with modern patterns.
type ReActAgent struct {
	// Core components
	id           string
	name         string
	module       *modules.ReAct
	toolRegistry *tools.InMemoryToolRegistry
	memory       agents.Memory
	llm          core.LLM

	// Modern patterns
	Reflector *SelfReflector
	Planner   *TaskPlanner
	Optimizer *MemoryOptimizer

	// Configuration
	config ReActAgentConfig

	// Interceptor support
	interceptors []core.AgentInterceptor

	// Capabilities
	capabilities []core.Tool

	// Execution state
	executionHistory []ExecutionRecord
	currentPlan      *Plan

	// Synchronization
	mu sync.RWMutex
}

// ExecutionRecord tracks execution history for reflection.
type ExecutionRecord struct {
	Timestamp   time.Time
	Input       map[string]interface{}
	Output      map[string]interface{}
	Actions     []ActionRecord
	Success     bool
	Error       error
	Reflections []string
}

// ActionRecord tracks individual actions taken.
type ActionRecord struct {
	Thought     string
	Action      string
	Tool        string
	Arguments   map[string]interface{}
	Observation string
	Success     bool
	Duration    time.Duration
}

// NewReActAgent creates a new ReAct agent with modern patterns.
func NewReActAgent(id, name string, opts ...Option) *ReActAgent {
	config := DefaultReActAgentConfig()

	// Apply options
	for _, opt := range opts {
		opt(&config)
	}

	// Initialize memory if not provided
	memory := agents.NewInMemoryStore()

	// Create tool registry
	toolRegistry := tools.NewInMemoryToolRegistry()

	// Create the agent
	agent := &ReActAgent{
		id:           id,
		name:         name,
		toolRegistry: toolRegistry,
		memory:       memory,
		config:       config,
		interceptors: make([]core.AgentInterceptor, 0),
		capabilities: make([]core.Tool, 0),
		executionHistory: make([]ExecutionRecord, 0),
	}

	// Initialize modern pattern components if enabled
	if config.EnableReflection {
		agent.Reflector = NewSelfReflector(config.ReflectionDepth, config.ReflectionDelay)
	}

	if config.EnablePlanning {
		agent.Planner = NewTaskPlanner(config.PlanningStrategy, config.MaxPlanDepth)
	}

	if config.EnableMemoryOpt {
		agent.Optimizer = NewMemoryOptimizer(
			config.MemoryRetention,
			config.ForgetThreshold,
		)
	}

	return agent
}

// Option configures a ReActAgent.
type Option func(*ReActAgentConfig)

// WithExecutionMode sets the execution mode.
func WithExecutionMode(mode ExecutionMode) Option {
	return func(c *ReActAgentConfig) {
		c.ExecutionMode = mode
	}
}

// WithMaxIterations sets the maximum iterations.
func WithMaxIterations(max int) Option {
	return func(c *ReActAgentConfig) {
		c.MaxIterations = max
	}
}

// WithReflection enables or disables reflection.
func WithReflection(enabled bool, depth int) Option {
	return func(c *ReActAgentConfig) {
		c.EnableReflection = enabled
		c.ReflectionDepth = depth
	}
}

// WithPlanning configures planning.
func WithPlanning(strategy PlanningStrategy, maxDepth int) Option {
	return func(c *ReActAgentConfig) {
		c.EnablePlanning = true
		c.PlanningStrategy = strategy
		c.MaxPlanDepth = maxDepth
	}
}

// WithMemoryOptimization enables memory optimization.
func WithMemoryOptimization(retention time.Duration, threshold float64) Option {
	return func(c *ReActAgentConfig) {
		c.EnableMemoryOpt = true
		c.MemoryRetention = retention
		c.ForgetThreshold = threshold
	}
}

// WithTimeout sets the execution timeout.
func WithTimeout(timeout time.Duration) Option {
	return func(c *ReActAgentConfig) {
		c.Timeout = timeout
	}
}

// WithXMLParsing enables XML interceptor-based parsing for tool actions.
// This provides enhanced XML validation, security features, and error handling.
func WithXMLParsing(config interceptors.XMLConfig) Option {
	return func(c *ReActAgentConfig) {
		c.EnableXMLParsing = true
		c.XMLConfig = &config
	}
}

// Initialize sets up the agent with an LLM and creates the ReAct module.
func (r *ReActAgent) Initialize(llm core.LLM, signature core.Signature) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.llm = llm

	// Create ReAct module with the signature
	r.module = modules.NewReAct(signature, r.toolRegistry, r.config.MaxIterations)
	r.module.SetLLM(llm)

	// Enable XML parsing if configured
	if r.config.EnableXMLParsing && r.config.XMLConfig != nil {
		r.module.WithXMLParsing(*r.config.XMLConfig)
	}

	return nil
}

// EnableXMLParsing enables XML interceptor-based parsing on an already initialized agent.
// This allows enabling XML parsing after agent creation.
func (r *ReActAgent) EnableXMLParsing(config interceptors.XMLConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.module == nil {
		return fmt.Errorf("agent not initialized - call Initialize() first")
	}

	r.config.EnableXMLParsing = true
	r.config.XMLConfig = &config
	r.module.WithXMLParsing(config)

	return nil
}

// RegisterTool adds a tool to the agent's registry.
func (r *ReActAgent) RegisterTool(tool core.Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	err := r.toolRegistry.Register(tool)
	if err != nil {
		return fmt.Errorf("failed to register tool: %w", err)
	}

	r.capabilities = append(r.capabilities, tool)
	return nil
}

// Execute runs the agent's task with given input.
func (r *ReActAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Check if we should use interceptors
	if r.config.EnableInterceptors && len(r.interceptors) > 0 {
		return r.ExecuteWithInterceptors(ctx, input, r.interceptors)
	}

	return r.executeInternal(ctx, input)
}

// executeInternal performs the actual execution logic.
func (r *ReActAgent) executeInternal(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	ctx, span := core.StartSpan(ctx, "ReActAgent.Execute")
	defer core.EndSpan(ctx)

	logger := logging.GetLogger()
	logger.Info(ctx, "Starting ReAct agent execution for agent %s", r.id)

	// Apply timeout
	if r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
	}

	// Load relevant memory context
	if r.config.EnableMemoryOpt && r.Optimizer != nil {
		memoryContext, err := r.loadMemoryContext(ctx, input)
		if err != nil {
			logger.Warn(ctx, "Failed to load memory context: %v", err)
		} else if memoryContext != nil {
			input["memory_context"] = memoryContext
		}
	}

	var result map[string]interface{}
	var err error

	// Choose execution mode
	switch r.config.ExecutionMode {
	case ModeReAct:
		result, err = r.executeReAct(ctx, input)
	case ModeReWOO:
		result, err = r.executeReWOO(ctx, input)
	case ModeHybrid:
		result, err = r.executeHybrid(ctx, input)
	default:
		result, err = r.executeReAct(ctx, input)
	}

	// Record execution for reflection
	record := ExecutionRecord{
		Timestamp: time.Now(),
		Input:     input,
		Output:    result,
		Success:   err == nil,
		Error:     err,
	}

	r.mu.Lock()
	r.executionHistory = append(r.executionHistory, record)
	r.mu.Unlock()

	// Perform reflection if enabled
	if r.config.EnableReflection && r.Reflector != nil {
		reflections := r.Reflector.Reflect(ctx, record)
		if len(reflections) > 0 {
			r.updateFromReflections(ctx, reflections)
		}
	}

	// Update memory with results
	if r.config.EnableMemoryOpt && r.Optimizer != nil {
		_ = r.Optimizer.Store(ctx, input, result, err == nil)
	}

	span.WithAnnotation("execution_mode", r.config.ExecutionMode)
	span.WithAnnotation("success", err == nil)

	return result, err
}

// executeReAct performs classic ReAct execution.
func (r *ReActAgent) executeReAct(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	if r.module == nil {
		return nil, fmt.Errorf("agent not initialized")
	}

	// Use the ReAct module directly
	return r.module.Process(ctx, input)
}

// executeReWOO performs plan-then-execute execution.
func (r *ReActAgent) executeReWOO(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()

	// First, create a plan
	if r.Planner == nil {
		return nil, fmt.Errorf("planner not initialized for ReWOO mode")
	}

	plan, err := r.Planner.CreatePlan(ctx, input, r.capabilities)
	if err != nil {
		return nil, fmt.Errorf("failed to create plan: %w", err)
	}

	r.mu.Lock()
	r.currentPlan = plan
	r.mu.Unlock()

	logger.Info(ctx, "Executing plan with %d steps", len(plan.Steps))

	// Execute plan steps
	results := make(map[string]interface{})
	for i, step := range plan.Steps {
		stepResult, err := r.executePlanStep(ctx, step, results)
		if err != nil {
			logger.Error(ctx, "Step %d failed: %v", i, err)
			// In ReWOO mode, we might want to continue or fail fast
			if step.Critical {
				return nil, fmt.Errorf("critical step %d failed: %w", i, err)
			}
		}

		// Store intermediate results
		results[fmt.Sprintf("step_%d", i)] = stepResult
	}

	return results, nil
}

// executeHybrid adaptively chooses between ReAct and ReWOO.
func (r *ReActAgent) executeHybrid(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	// Analyze task complexity to choose mode
	complexity := r.analyzeTaskComplexity(input)

	if complexity > 0.7 {
		// Complex task: use ReWOO for efficiency
		return r.executeReWOO(ctx, input)
	}

	// Simple or interactive task: use ReAct for flexibility
	return r.executeReAct(ctx, input)
}

// analyzeTaskComplexity estimates task complexity.
func (r *ReActAgent) analyzeTaskComplexity(input map[string]interface{}) float64 {
	// Simple heuristic based on input structure
	// In practice, this could use ML or more sophisticated analysis
	complexity := 0.5

	// Check for multi-step indicators
	if task, ok := input["task"].(string); ok {
		if len(task) > 200 {
			complexity += 0.2
		}
		// Check for keywords indicating complexity
		complexKeywords := []string{"analyze", "compare", "synthesize", "evaluate", "multiple"}
		for _, keyword := range complexKeywords {
			if contains(task, keyword) {
				complexity += 0.1
			}
		}
	}

	// Check if we have many tools available
	if len(r.capabilities) > 5 {
		complexity += 0.1
	}

	if complexity > 1.0 {
		complexity = 1.0
	}
	return complexity
}

// executePlanStep executes a single step of a plan.
func (r *ReActAgent) executePlanStep(ctx context.Context, step PlanStep, previousResults map[string]interface{}) (interface{}, error) {
	logger := logging.GetLogger()
	logger.Debug(ctx, "Executing plan step: %s (tool: %s)", step.ID, step.Tool)

	// Create context with timeout if specified
	execCtx := ctx
	if step.Timeout > 0 {
		var cancel context.CancelFunc
		execCtx, cancel = context.WithTimeout(ctx, step.Timeout)
		defer cancel()
	}

	// Prepare arguments by merging step arguments with previous results
	arguments := make(map[string]interface{})

	// Copy step arguments
	for k, v := range step.Arguments {
		arguments[k] = v
	}

	// Add previous results that this step depends on
	for _, depID := range step.DependsOn {
		if result, exists := previousResults[depID]; exists {
			// Use dependency ID as argument key, or a specific key if defined
			arguments[depID+"_result"] = result
		}
	}

	// Add all previous results as context for complex operations
	if len(previousResults) > 0 {
		arguments["previous_results"] = previousResults
	}

	// Execute the tool
	if r.toolRegistry == nil {
		return nil, fmt.Errorf("tool registry not initialized")
	}

	// Find the tool in the registry
	tool, err := r.toolRegistry.Get(step.Tool)
	if err != nil {
		return nil, fmt.Errorf("tool '%s' not found in registry: %w", step.Tool, err)
	}

	// Validate arguments
	if err := tool.Validate(arguments); err != nil {
		return nil, fmt.Errorf("invalid parameters for tool '%s': %w", step.Tool, err)
	}

	// Execute tool
	result, err := tool.Execute(execCtx, arguments)
	if err != nil {
		if step.Critical {
			return nil, fmt.Errorf("critical step '%s' failed: %w", step.ID, err)
		}
		// For non-critical steps, log the error but return a result indicating failure
		logger.Warn(ctx, "Non-critical step '%s' failed: %v", step.ID, err)
		return map[string]interface{}{
			"success": false,
			"error":   err.Error(),
			"step_id": step.ID,
		}, nil
	}

	logger.Debug(ctx, "Plan step '%s' executed successfully", step.ID)
	return result, nil
}

// loadMemoryContext loads relevant memory for the task.
func (r *ReActAgent) loadMemoryContext(ctx context.Context, input map[string]interface{}) (interface{}, error) {
	if r.Optimizer == nil {
		return nil, nil
	}

	return r.Optimizer.Retrieve(ctx, input)
}

// updateFromReflections updates agent behavior based on reflections.
func (r *ReActAgent) updateFromReflections(ctx context.Context, reflections []Reflection) {
	logger := logging.GetLogger()

	for _, reflection := range reflections {
		logger.Debug(ctx, "Applying reflection: %s", reflection.Insight)

		// Update configuration based on reflection type
		switch reflection.Type {
		case ReflectionTypeStrategy:
			// Adjust execution strategy
			if reflection.Confidence > 0.8 {
				r.adjustStrategy(reflection)
			}
		case ReflectionTypePerformance:
			// Adjust performance parameters
			r.adjustPerformance(reflection)
		case ReflectionTypeLearning:
			// Store learning for future use
			r.storeLearning(ctx, reflection)
		}
	}
}

// adjustStrategy adjusts execution strategy based on reflection.
func (r *ReActAgent) adjustStrategy(reflection Reflection) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Example: Switch execution mode based on past performance
	if reflection.Recommendation == "use_rewoo_for_structured_tasks" {
		r.config.ExecutionMode = ModeHybrid
	}
}

// adjustPerformance adjusts performance parameters.
func (r *ReActAgent) adjustPerformance(reflection Reflection) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Example: Adjust iteration limit based on success rate
	if reflection.Recommendation == "increase_max_iterations" {
		newMax := r.config.MaxIterations + 2
		if newMax > 20 {
			newMax = 20
		}
		r.config.MaxIterations = newMax
	}
}

// storeLearning stores learned insights for future use.
func (r *ReActAgent) storeLearning(ctx context.Context, reflection Reflection) {
	if r.memory != nil {
		key := fmt.Sprintf("learning_%s_%d", r.id, time.Now().Unix())
		_ = r.memory.Store(key, reflection.Insight)
	}
}

// GetCapabilities returns the tools/capabilities available to this agent.
func (r *ReActAgent) GetCapabilities() []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.capabilities
}

// GetMemory returns the agent's memory store.
func (r *ReActAgent) GetMemory() agents.Memory {
	return r.memory
}

// ExecuteWithInterceptors runs the agent with interceptor support.
func (r *ReActAgent) ExecuteWithInterceptors(ctx context.Context, input map[string]interface{}, interceptors []core.AgentInterceptor) (map[string]interface{}, error) {
	// Use provided interceptors, or fall back to agent's default interceptors
	if interceptors == nil {
		interceptors = r.interceptors
	}

	// Create agent info for interceptors
	info := core.NewAgentInfo(r.id, "ReActAgent", r.capabilities)

	// Create the base handler that calls the agent's execute method
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		return r.executeInternal(ctx, input)
	}

	// Chain the interceptors
	chainedInterceptor := core.ChainAgentInterceptors(interceptors...)

	// Execute with interceptors
	return chainedInterceptor(ctx, input, info, handler)
}

// SetInterceptors sets the default interceptors for this agent.
func (r *ReActAgent) SetInterceptors(interceptors []core.AgentInterceptor) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.interceptors = make([]core.AgentInterceptor, len(interceptors))
	copy(r.interceptors, interceptors)
}

// GetInterceptors returns the current interceptors.
func (r *ReActAgent) GetInterceptors() []core.AgentInterceptor {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]core.AgentInterceptor, len(r.interceptors))
	copy(result, r.interceptors)
	return result
}

// ClearInterceptors removes all interceptors.
func (r *ReActAgent) ClearInterceptors() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.interceptors = nil
}

// GetAgentID returns the unique identifier for this agent.
func (r *ReActAgent) GetAgentID() string {
	return r.id
}

// GetAgentType returns the type of this agent.
func (r *ReActAgent) GetAgentType() string {
	return "ReActAgent"
}

// GetExecutionHistory returns the execution history for analysis.
func (r *ReActAgent) GetExecutionHistory() []ExecutionRecord {
	r.mu.RLock()
	defer r.mu.RUnlock()

	history := make([]ExecutionRecord, len(r.executionHistory))
	copy(history, r.executionHistory)
	return history
}

// Utility functions

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
