package react

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
	contextmgmt "github.com/XiaoConstantine/dspy-go/pkg/agents/context"
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

	// Native function calling settings
	EnableNativeFunctionCalling bool
	FunctionCallingConfig       *interceptors.FunctionCallingConfig

	// Context Engineering settings (Manus-inspired patterns)
	EnableContextEngineering bool                             `json:"enable_context_engineering"`
	ContextConfig           contextmgmt.Config               `json:"context_config,omitempty"`
	ContextBaseDir          string                           `json:"context_base_dir"`
	AutoTodoManagement      bool                             `json:"auto_todo_management"`
	AutoErrorLearning       bool                             `json:"auto_error_learning"`
	ContextOptLevel         contextmgmt.CompressionPriority  `json:"context_optimization_level"`
	MaxContextTokens        int                              `json:"max_context_tokens"`
	CacheEfficiencyTarget   float64                          `json:"cache_efficiency_target"`

	// ACE (Agentic Context Engineering) settings
	EnableACE   bool       `json:"enable_ace"`
	ACEConfig   ace.Config `json:"ace_config,omitempty"`
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
		EnableInterceptors:          true,
		EnableXMLParsing:            false, // Disabled by default for backward compatibility
		XMLConfig:                   nil,
		EnableNativeFunctionCalling: false, // Disabled by default for backward compatibility
		FunctionCallingConfig:       nil,
		// Context Engineering defaults
		EnableContextEngineering: false, // Opt-in for backward compatibility
		ContextBaseDir:          "./agent_memory",
		AutoTodoManagement:      true,
		AutoErrorLearning:       true,
		ContextOptLevel:         contextmgmt.PriorityMedium,
		MaxContextTokens:        8192,
		CacheEfficiencyTarget:   0.85,
	}
}

// ProductionReActAgentConfig returns a configuration optimized for production cost efficiency.
func ProductionReActAgentConfig() ReActAgentConfig {
	config := DefaultReActAgentConfig()

	// Enable context engineering for maximum cost reduction
	config.EnableContextEngineering = true
	config.ContextConfig = contextmgmt.ProductionConfig()
	config.CacheEfficiencyTarget = 0.95
	config.ContextOptLevel = contextmgmt.PriorityHigh
	config.MaxContextTokens = 16384

	return config
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

	// Context Engineering (Manus-inspired patterns)
	contextManager *contextmgmt.Manager

	// ACE (Agentic Context Engineering)
	aceManager *ace.Manager

	// Configuration
	config ReActAgentConfig

	// Interceptor support
	interceptors []core.AgentInterceptor

	// Capabilities
	capabilities []core.Tool

	// Execution state enhanced with context tracking
	executionHistory []ExecutionRecord
	currentPlan      *Plan
	activeTaskID     string
	contextVersion   int64

	// Performance metrics
	totalExecutions   int64
	contextSavings    float64
	avgProcessingTime time.Duration

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

	// Context Engineering metrics
	ContextVersion      int64                        `json:"context_version,omitempty"`
	ContextResponse     *contextmgmt.ContextResponse `json:"context_response,omitempty"`
	OptimizationsApplied []string                    `json:"optimizations_applied,omitempty"`
	CostSavings         float64                      `json:"cost_savings,omitempty"`
	ProcessingTime      time.Duration                `json:"processing_time,omitempty"`
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
		id:               id,
		name:             name,
		toolRegistry:     toolRegistry,
		memory:           memory,
		config:           config,
		interceptors:     make([]core.AgentInterceptor, 0),
		capabilities:     make([]core.Tool, 0),
		executionHistory: make([]ExecutionRecord, 0),
		contextVersion:   1,
	}

	// Initialize context engineering if enabled
	if config.EnableContextEngineering {
		contextConfig := config.ContextConfig
		if contextConfig.SessionID == "" {
			contextConfig.SessionID = fmt.Sprintf("session_%s_%d", id, time.Now().Unix())
		}

		contextManager, err := contextmgmt.NewManager(
			contextConfig.SessionID,
			id,
			config.ContextBaseDir,
			contextConfig,
		)
		if err != nil {
			// Log error but don't fail agent creation
			logger := logging.GetLogger()
			logger.Warn(context.Background(), "Failed to initialize context manager for agent %s: %v", id, err)
		} else {
			agent.contextManager = contextManager
		}
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

	// Initialize ACE if enabled
	if config.EnableACE {
		var reflector *ace.UnifiedReflector
		if agent.Reflector != nil {
			adapter := NewSelfReflectorACEAdapter(agent.Reflector)
			reflector = ace.NewUnifiedReflector([]ace.Adapter{adapter}, ace.NewSimpleReflector())
		}

		aceManager, err := ace.NewManager(config.ACEConfig, reflector)
		if err != nil {
			logger := logging.GetLogger()
			logger.Warn(context.Background(), "Failed to initialize ACE manager for agent %s: %v", id, err)
		} else {
			agent.aceManager = aceManager
		}
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

// WithNativeFunctionCalling enables native LLM function calling for tool selection.
// This uses the provider's built-in tool calling API instead of text-based parsing,
// which provides more reliable tool selection and eliminates parsing errors.
func WithNativeFunctionCalling(config interceptors.FunctionCallingConfig) Option {
	return func(c *ReActAgentConfig) {
		c.EnableNativeFunctionCalling = true
		c.FunctionCallingConfig = &config
	}
}

// WithContextEngineering enables Manus-inspired context engineering for 10x cost reduction.
func WithContextEngineering(baseDir string, contextConfig contextmgmt.Config) Option {
	return func(c *ReActAgentConfig) {
		c.EnableContextEngineering = true
		c.ContextBaseDir = baseDir
		c.ContextConfig = contextConfig
	}
}

// WithProductionContextOptimization enables production-grade context optimization.
func WithProductionContextOptimization() Option {
	return func(c *ReActAgentConfig) {
		c.EnableContextEngineering = true
		c.ContextConfig = contextmgmt.ProductionConfig()
		c.CacheEfficiencyTarget = 0.95
		c.ContextOptLevel = contextmgmt.PriorityHigh
		c.MaxContextTokens = 16384
		c.AutoTodoManagement = true
		c.AutoErrorLearning = true
	}
}

// WithContextOptimization configures context optimization level.
func WithContextOptimization(level contextmgmt.CompressionPriority, maxTokens int, cacheTarget float64) Option {
	return func(c *ReActAgentConfig) {
		c.ContextOptLevel = level
		c.MaxContextTokens = maxTokens
		c.CacheEfficiencyTarget = cacheTarget
	}
}

// WithACE enables Agentic Context Engineering for self-improving agents.
// The reflector parameter is optional - if nil, a default reflector using the
// agent's SelfReflector will be created automatically.
func WithACE(config ace.Config) Option {
	return func(c *ReActAgentConfig) {
		c.EnableACE = true
		c.ACEConfig = config
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

	// Enable native function calling if configured
	// Note: This takes precedence over XML parsing as they serve similar purposes
	if r.config.EnableNativeFunctionCalling && r.config.FunctionCallingConfig != nil {
		r.module.WithNativeFunctionCallingConfig(*r.config.FunctionCallingConfig)
	} else if r.config.EnableNativeFunctionCalling {
		// Use default config with the agent's tool registry
		r.module.WithNativeFunctionCalling()
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

// EnableNativeFunctionCalling enables native LLM function calling on an already initialized agent.
// This allows enabling function calling after agent creation.
// If config is nil, uses the default configuration with the agent's tool registry.
func (r *ReActAgent) EnableNativeFunctionCalling(config *interceptors.FunctionCallingConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.module == nil {
		return fmt.Errorf("agent not initialized - call Initialize() first")
	}

	r.config.EnableNativeFunctionCalling = true
	if config != nil {
		r.config.FunctionCallingConfig = config
		r.module.WithNativeFunctionCallingConfig(*config)
	} else {
		// Use default config - the module will set up with its own registry
		r.module.WithNativeFunctionCalling()
	}

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

// executeInternal performs the actual execution logic with context engineering.
func (r *ReActAgent) executeInternal(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	startTime := time.Now()
	ctx, span := core.StartSpan(ctx, "ReActAgent.Execute")
	defer core.EndSpan(ctx)

	logger := logging.GetLogger()
	logger.Info(ctx, "Starting ReAct agent execution for agent %s", r.id)

	r.mu.Lock()
	r.totalExecutions++
	r.contextVersion++
	executionID := fmt.Sprintf("exec_%d_%d", r.totalExecutions, time.Now().UnixNano())
	r.mu.Unlock()

	// Apply timeout
	if r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
	}

	// ACE: Start trajectory recording with handle for concurrency safety
	var taskQuery string
	if task, ok := input["task"].(string); ok {
		taskQuery = task
	}
	var aceRecorder *ace.TrajectoryRecorder
	if r.config.EnableACE && r.aceManager != nil {
		aceRecorder = r.aceManager.StartTrajectory(r.id, "react_execution", taskQuery)
	}

	// STEP 1: Build optimized context using Manus patterns (if enabled)
	var contextResponse *contextmgmt.ContextResponse
	var optimizedInput map[string]interface{}
	if r.config.EnableContextEngineering && r.contextManager != nil {
		var err error
		optimizedInput, contextResponse, err = r.buildOptimizedContext(ctx, input, executionID)
		if err != nil {
			logger.Warn(ctx, "Context optimization failed, falling back to basic execution: %v", err)
			optimizedInput = input
		} else {
			logger.Debug(ctx, "Context optimization achieved %.1f%% cost reduction",
				(1.0-contextResponse.CompressionRatio)*100)
		}
	} else {
		optimizedInput = input
	}

	// ACE: Inject learnings into context (prepended for cache efficiency)
	if r.config.EnableACE && r.aceManager != nil {
		if learnings := r.aceManager.GetLearningsContext(); learnings != "" {
			optimizedInput["ace_learnings"] = learnings
		}
	}

	// STEP 2: Load relevant memory context (legacy system)
	if r.config.EnableMemoryOpt && r.Optimizer != nil {
		memoryContext, err := r.loadMemoryContext(ctx, optimizedInput)
		if err != nil {
			logger.Warn(ctx, "Failed to load memory context: %v", err)
		} else if memoryContext != nil {
			optimizedInput["memory_context"] = memoryContext
		}
	}

	// STEP 3: Execute using standard ReAct flow but with optimized context
	var result map[string]interface{}
	var err error

	switch r.config.ExecutionMode {
	case ModeReAct:
		result, err = r.executeReAct(ctx, optimizedInput)
	case ModeReWOO:
		result, err = r.executeReWOO(ctx, optimizedInput)
	case ModeHybrid:
		result, err = r.executeHybrid(ctx, optimizedInput)
	default:
		result, err = r.executeReAct(ctx, optimizedInput)
	}

	// STEP 4: Create enhanced execution record
	processingTime := time.Since(startTime)
	record := r.createEnhancedExecutionRecord(executionID, input, result, err, contextResponse, processingTime)

	// ACE: Record steps from execution record using the recorder handle
	if aceRecorder != nil {
		for _, action := range record.Actions {
			var stepErr error
			if !action.Success {
				stepErr = fmt.Errorf("%s", action.Observation)
			}
			aceRecorder.RecordStep(action.Action, action.Tool, action.Thought, action.Arguments, map[string]any{"observation": action.Observation}, stepErr)
		}
	}

	r.mu.Lock()
	r.executionHistory = append(r.executionHistory, record)
	r.avgProcessingTime = r.updateAvgProcessingTime(processingTime)
	r.mu.Unlock()

	// STEP 5: Update context management systems with results
	if r.config.EnableContextEngineering && r.contextManager != nil {
		r.updateContextSystems(ctx, executionID, input, result, err)
	}

	// STEP 6: Perform reflection with context awareness
	if r.config.EnableReflection && r.Reflector != nil {
		reflections := r.Reflector.Reflect(ctx, record)
		if len(reflections) > 0 {
			r.updateFromReflections(ctx, reflections)
		}
	}

	// ACE: End trajectory with outcome using the recorder handle
	if aceRecorder != nil {
		outcome := ace.OutcomeSuccess
		if err != nil {
			outcome = ace.OutcomeFailure
		} else if !record.Success {
			outcome = ace.OutcomePartial
		}
		r.aceManager.EndTrajectory(aceRecorder, outcome)
	}

	// STEP 7: Update legacy memory system
	if r.config.EnableMemoryOpt && r.Optimizer != nil {
		_ = r.Optimizer.Store(ctx, input, result, err == nil)
	}

	// STEP 8: Update performance metrics
	if contextResponse != nil {
		r.mu.Lock()
		r.contextSavings += contextResponse.CostSavings
		r.mu.Unlock()
	}

	span.WithAnnotation("execution_mode", r.config.ExecutionMode)
	span.WithAnnotation("success", err == nil)
	span.WithAnnotation("context_engineering", r.config.EnableContextEngineering)
	if contextResponse != nil {
		span.WithAnnotation("cache_hit_rate", contextResponse.CacheHitRate)
		span.WithAnnotation("cost_savings", contextResponse.CostSavings)
	}

	logger.Info(ctx, "ReAct execution completed: %s (%.2fms, context savings: $%.6f)",
		executionID, processingTime.Seconds()*1000, r.getContextSavingsOrDefault(contextResponse))

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

// Context Engineering helper methods

// buildOptimizedContext creates highly optimized context using all Manus patterns.
func (r *ReActAgent) buildOptimizedContext(ctx context.Context, input map[string]interface{}, executionID string) (map[string]interface{}, *contextmgmt.ContextResponse, error) {
	logger := logging.GetLogger()

	// Extract observations and task from input
	observations := r.extractObservations(input)
	currentTask := r.extractCurrentTask(input)

	// Update active task tracking (thread-safe)
	r.mu.Lock()
	if currentTask != "" && r.activeTaskID != executionID {
		r.activeTaskID = executionID
		shouldAddTodo := r.config.AutoTodoManagement
		r.mu.Unlock()

		if shouldAddTodo {
			if err := r.contextManager.AddTodo(ctx, currentTask, 7); err != nil {
				logger.Warn(ctx, "Failed to add task to todo system: %v", err)
			}
		}
	} else {
		r.mu.Unlock()
	}

	// Build context request with optimization preferences
	request := contextmgmt.ContextRequest{
		Observations:         observations,
		CurrentTask:          currentTask,
		AdditionalData:       r.extractAdditionalData(input),
		PrioritizeCache:      true, // CRITICAL for cost savings
		CompressionPriority:  r.config.ContextOptLevel,
		AllowDiversification: true,
		IncludeErrors:        r.config.AutoErrorLearning,
		IncludeTodos:         r.config.AutoTodoManagement,
		MaxTokens:            r.config.MaxContextTokens,
		MinCacheEfficiency:   r.config.CacheEfficiencyTarget,
	}

	// Get optimized context
	contextResponse, err := r.contextManager.BuildOptimizedContext(ctx, request)
	if err != nil {
		return input, nil, fmt.Errorf("failed to build optimized context: %w", err)
	}

	// Create enhanced input with optimized context
	optimizedInput := make(map[string]interface{})
	for k, v := range input {
		optimizedInput[k] = v
	}

	// Replace or enhance with optimized context
	optimizedInput["optimized_context"] = contextResponse.Context
	optimizedInput["context_metadata"] = map[string]interface{}{
		"version":         contextResponse.ContextVersion,
		"token_count":     contextResponse.TokenCount,
		"cache_hit_rate":  contextResponse.CacheHitRate,
		"cost_savings":    contextResponse.CostSavings,
		"optimizations":   contextResponse.OptimizationsApplied,
	}

	logger.Debug(ctx, "Built optimized context: %d tokens (%.1f%% cache hit rate)",
		contextResponse.TokenCount, contextResponse.CacheHitRate*100)

	return optimizedInput, contextResponse, nil
}

// updateContextSystems updates context management systems based on execution results.
func (r *ReActAgent) updateContextSystems(ctx context.Context, executionID string, input map[string]interface{}, result map[string]interface{}, err error) {
	logger := logging.GetLogger()

	// Update todo management (thread-safe)
	r.mu.RLock()
	activeTask := r.activeTaskID
	shouldManageTodo := (activeTask == executionID && r.config.AutoTodoManagement)
	r.mu.RUnlock()

	if shouldManageTodo {
		if err == nil {
			// Mark task as completed
			if todoErr := r.contextManager.CompleteTodo(ctx, activeTask); todoErr != nil {
				logger.Warn(ctx, "Failed to complete todo: %v", todoErr)
			}
		} else {
			// Keep task active for retry, but record the error
			r.recordExecutionError(ctx, executionID, err)
		}
	}

	// Record success/failure for learning
	if r.config.AutoErrorLearning {
		if err == nil {
			r.contextManager.RecordSuccess(ctx, "agent_execution", "Successful task completion", map[string]interface{}{
				"execution_id": executionID,
				"input":        input,
				"result":       result,
			})
		} else {
			r.recordExecutionError(ctx, executionID, err)
		}
	}
}

// recordExecutionError records execution errors for learning.
func (r *ReActAgent) recordExecutionError(ctx context.Context, executionID string, err error) {
	// Classify error for better learning
	errorType := r.classifyError(err)
	severity := r.assessErrorSeverity(err)

	r.contextManager.RecordError(ctx, errorType, err.Error(), severity, map[string]interface{}{
		"execution_id": executionID,
		"agent_id":     r.id,
		"timestamp":    time.Now(),
	})
}

// classifyError categorizes errors for pattern recognition.
func (r *ReActAgent) classifyError(err error) string {
	errStr := strings.ToLower(err.Error())

	switch {
	case strings.Contains(errStr, "timeout"):
		return "execution_timeout"
	case strings.Contains(errStr, "tool"):
		return "tool_failure"
	case strings.Contains(errStr, "llm") || strings.Contains(errStr, "model"):
		return "llm_error"
	case strings.Contains(errStr, "context"):
		return "context_error"
	case strings.Contains(errStr, "memory"):
		return "memory_error"
	default:
		return "general_execution_error"
	}
}

// assessErrorSeverity determines error severity for prioritized learning.
func (r *ReActAgent) assessErrorSeverity(err error) contextmgmt.ErrorSeverity {
	errStr := strings.ToLower(err.Error())

	switch {
	case strings.Contains(errStr, "critical") || strings.Contains(errStr, "fatal"):
		return contextmgmt.SeverityCritical
	case strings.Contains(errStr, "timeout") || strings.Contains(errStr, "fail"):
		return contextmgmt.SeverityHigh
	case strings.Contains(errStr, "warn") || strings.Contains(errStr, "retry"):
		return contextmgmt.SeverityMedium
	default:
		return contextmgmt.SeverityLow
	}
}

// createEnhancedExecutionRecord creates a comprehensive execution record.
func (r *ReActAgent) createEnhancedExecutionRecord(executionID string, input map[string]interface{}, result map[string]interface{}, err error, contextResponse *contextmgmt.ContextResponse, processingTime time.Duration) ExecutionRecord {
	// Thread-safe access to contextVersion
	r.mu.RLock()
	contextVersion := r.contextVersion
	r.mu.RUnlock()

	record := ExecutionRecord{
		Timestamp: time.Now(),
		Input:     input,
		Output:    result,
		Success:   err == nil,
		Error:     err,
		ContextVersion: contextVersion,
		ProcessingTime: processingTime,
	}

	if contextResponse != nil {
		record.ContextResponse = contextResponse
		record.OptimizationsApplied = contextResponse.OptimizationsApplied
		record.CostSavings = contextResponse.CostSavings
	}

	return record
}

// extractObservations extracts observations from input.
func (r *ReActAgent) extractObservations(input map[string]interface{}) []string {
	var observations []string

	if obs, ok := input["observations"]; ok {
		switch v := obs.(type) {
		case []string:
			observations = v
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					observations = append(observations, str)
				}
			}
		case string:
			observations = []string{v}
		}
	}

	// Also check for memory context
	if memCtx, ok := input["memory_context"]; ok {
		if str, ok := memCtx.(string); ok {
			observations = append(observations, str)
		}
	}

	return observations
}

// extractCurrentTask extracts current task from input.
func (r *ReActAgent) extractCurrentTask(input map[string]interface{}) string {
	// Try multiple common keys for task description
	taskKeys := []string{"task", "current_task", "objective", "goal", "instruction"}

	for _, key := range taskKeys {
		if task, ok := input[key].(string); ok && task != "" {
			return task
		}
	}

	return ""
}

// extractAdditionalData extracts additional context data.
func (r *ReActAgent) extractAdditionalData(input map[string]interface{}) map[string]interface{} {
	additional := make(map[string]interface{})

	// Copy all non-standard keys as additional data
	excludeKeys := map[string]bool{
		"task": true, "current_task": true, "objective": true, "goal": true,
		"observations": true, "memory_context": true, "optimized_context": true,
		"context_metadata": true,
	}

	for k, v := range input {
		if !excludeKeys[k] {
			additional[k] = v
		}
	}

	return additional
}

// updateAvgProcessingTime updates average processing time with exponential smoothing.
func (r *ReActAgent) updateAvgProcessingTime(newTime time.Duration) time.Duration {
	if r.totalExecutions == 1 {
		return newTime
	}

	// Exponential moving average
	alpha := 0.1
	return time.Duration(float64(r.avgProcessingTime)*alpha + float64(newTime)*(1.0-alpha))
}

// getContextSavingsOrDefault returns context savings or default value.
func (r *ReActAgent) getContextSavingsOrDefault(contextResponse *contextmgmt.ContextResponse) float64 {
	if contextResponse != nil {
		return contextResponse.CostSavings
	}
	return 0.0
}

// GetContextPerformanceMetrics returns comprehensive context management metrics.
func (r *ReActAgent) GetContextPerformanceMetrics() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics := map[string]interface{}{
		"total_executions":      r.totalExecutions,
		"context_version":       r.contextVersion,
		"context_savings":       r.contextSavings,
		"avg_processing_time":   r.avgProcessingTime.Milliseconds(),
		"context_mgmt_enabled":  r.config.EnableContextEngineering,
	}

	// Add context manager metrics if available
	if r.config.EnableContextEngineering && r.contextManager != nil {
		contextMetrics := r.contextManager.GetPerformanceMetrics()
		metrics["context_manager"] = contextMetrics
	}

	return metrics
}

// GetContextHealthStatus returns health status of context management systems.
func (r *ReActAgent) GetContextHealthStatus() map[string]interface{} {
	if !r.config.EnableContextEngineering || r.contextManager == nil {
		return map[string]interface{}{
			"status": "disabled",
			"message": "Context management is disabled",
		}
	}

	return r.contextManager.GetHealthStatus()
}

// Close releases resources held by the agent.
func (r *ReActAgent) Close() error {
	if r.aceManager != nil {
		return r.aceManager.Close()
	}
	return nil
}

// GetACEMetrics returns ACE performance metrics if enabled.
func (r *ReActAgent) GetACEMetrics() map[string]int64 {
	if r.aceManager == nil {
		return nil
	}
	return r.aceManager.GetMetrics()
}

// Utility functions

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}
