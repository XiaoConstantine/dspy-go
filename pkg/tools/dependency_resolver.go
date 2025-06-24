package tools

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// DependencyNode represents a tool with its dependencies.
type DependencyNode struct {
	ToolName     string                 // Name of the tool
	Dependencies []string               // Names of tools this depends on
	Outputs      []string               // Output fields this tool produces
	Inputs       []string               // Required input fields
	Config       map[string]interface{} // Tool-specific configuration
	Priority     int                    // Execution priority (higher = earlier)
}

// ExecutionPlan represents the optimized execution plan for tools.
type ExecutionPlan struct {
	Phases []ExecutionPhase // Phases of execution (tools within a phase can run in parallel)
	Graph  *DependencyGraph // The dependency graph
}

// ExecutionPhase contains tools that can be executed in parallel.
type ExecutionPhase struct {
	Tools       []string // Tools to execute in this phase
	ParallelOk  bool     // Whether tools in this phase can run in parallel
	MaxParallel int      // Maximum number of parallel executions
}

// DependencyGraph manages tool dependencies and execution planning.
type DependencyGraph struct {
	nodes       map[string]*DependencyNode
	edges       map[string][]string // adjacency list: tool -> dependencies
	reverseEdges map[string][]string // reverse adjacency list: tool -> dependents
	mu          sync.RWMutex
}

// NewDependencyGraph creates a new dependency graph.
func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		nodes:       make(map[string]*DependencyNode),
		edges:       make(map[string][]string),
		reverseEdges: make(map[string][]string),
	}
}

// AddNode adds a tool node to the dependency graph.
func (dg *DependencyGraph) AddNode(node *DependencyNode) error {
	dg.mu.Lock()
	defer dg.mu.Unlock()

	if node.ToolName == "" {
		return errors.New(errors.InvalidInput, "tool name cannot be empty")
	}

	// Store the node
	dg.nodes[node.ToolName] = node
	
	// Initialize edges if not present
	if _, exists := dg.edges[node.ToolName]; !exists {
		dg.edges[node.ToolName] = make([]string, 0)
	}
	if _, exists := dg.reverseEdges[node.ToolName]; !exists {
		dg.reverseEdges[node.ToolName] = make([]string, 0)
	}

	// Add dependency edges
	for _, dep := range node.Dependencies {
		dg.edges[node.ToolName] = append(dg.edges[node.ToolName], dep)
		if _, exists := dg.reverseEdges[dep]; !exists {
			dg.reverseEdges[dep] = make([]string, 0)
		}
		dg.reverseEdges[dep] = append(dg.reverseEdges[dep], node.ToolName)
	}

	return nil
}

// RemoveNode removes a tool node from the dependency graph.
func (dg *DependencyGraph) RemoveNode(toolName string) error {
	dg.mu.Lock()
	defer dg.mu.Unlock()

	if _, exists := dg.nodes[toolName]; !exists {
		return errors.WithFields(
			errors.New(errors.ResourceNotFound, "tool not found in graph"),
			errors.Fields{"tool_name": toolName},
		)
	}

	// Remove from nodes
	delete(dg.nodes, toolName)

	// Remove from edges
	delete(dg.edges, toolName)
	delete(dg.reverseEdges, toolName)

	// Remove references from other nodes
	for tool, deps := range dg.edges {
		newDeps := make([]string, 0)
		for _, dep := range deps {
			if dep != toolName {
				newDeps = append(newDeps, dep)
			}
		}
		dg.edges[tool] = newDeps
	}

	for tool, dependents := range dg.reverseEdges {
		newDependents := make([]string, 0)
		for _, dependent := range dependents {
			if dependent != toolName {
				newDependents = append(newDependents, dependent)
			}
		}
		dg.reverseEdges[tool] = newDependents
	}

	return nil
}

// CreateExecutionPlan creates an optimized execution plan for the dependency graph.
func (dg *DependencyGraph) CreateExecutionPlan() (*ExecutionPlan, error) {
	dg.mu.RLock()
	defer dg.mu.RUnlock()

	// Check for cycles
	if hasCycle, cycle := dg.detectCycle(); hasCycle {
		return nil, errors.WithFields(
			errors.New(errors.InvalidInput, "dependency cycle detected"),
			errors.Fields{"cycle": cycle},
		)
	}

	// Topological sort with priority consideration
	phases, err := dg.topologicalSortByPhases()
	if err != nil {
		return nil, err
	}

	plan := &ExecutionPlan{
		Phases: phases,
		Graph:  dg,
	}

	return plan, nil
}

// topologicalSortByPhases performs topological sort and groups tools into execution phases.
func (dg *DependencyGraph) topologicalSortByPhases() ([]ExecutionPhase, error) {
	// Calculate in-degrees
	inDegree := make(map[string]int)
	for tool := range dg.nodes {
		inDegree[tool] = len(dg.edges[tool])
	}

	var phases []ExecutionPhase
	remaining := make(map[string]bool)
	for tool := range dg.nodes {
		remaining[tool] = true
	}

	for len(remaining) > 0 {
		// Find all tools with no dependencies (in-degree = 0)
		var currentPhase []string
		for tool := range remaining {
			if inDegree[tool] == 0 {
				currentPhase = append(currentPhase, tool)
			}
		}

		if len(currentPhase) == 0 {
			// This shouldn't happen if we checked for cycles, but safety check
			return nil, errors.New(errors.Unknown, "circular dependency detected during topological sort")
		}

		// Sort current phase by priority (higher priority first)
		sort.Slice(currentPhase, func(i, j int) bool {
			return dg.nodes[currentPhase[i]].Priority > dg.nodes[currentPhase[j]].Priority
		})

		// Add phase
		phases = append(phases, ExecutionPhase{
			Tools:       currentPhase,
			ParallelOk:  true, // Tools in same phase can run in parallel
			MaxParallel: len(currentPhase), // No limit by default
		})

		// Remove current phase tools and update in-degrees
		for _, tool := range currentPhase {
			delete(remaining, tool)
			
			// Reduce in-degree for dependent tools
			for _, dependent := range dg.reverseEdges[tool] {
				if remaining[dependent] {
					inDegree[dependent]--
				}
			}
		}
	}

	return phases, nil
}

// detectCycle detects if there's a cycle in the dependency graph.
func (dg *DependencyGraph) detectCycle() (bool, []string) {
	color := make(map[string]int) // 0: white, 1: gray, 2: black
	parent := make(map[string]string)
	
	var cycle []string
	
	var dfs func(string) bool
	dfs = func(node string) bool {
		color[node] = 1 // gray
		
		for _, dep := range dg.edges[node] {
			if color[dep] == 1 {
				// Back edge found - cycle detected
				cycle = dg.buildCyclePath(dep, node, parent)
				return true
			}
			if color[dep] == 0 {
				parent[dep] = node
				if dfs(dep) {
					return true
				}
			}
		}
		
		color[node] = 2 // black
		return false
	}
	
	for node := range dg.nodes {
		if color[node] == 0 {
			if dfs(node) {
				return true, cycle
			}
		}
	}
	
	return false, nil
}

// buildCyclePath builds the cycle path from the detected back edge.
func (dg *DependencyGraph) buildCyclePath(start, end string, parent map[string]string) []string {
	var path []string
	current := end
	path = append(path, current)
	
	for current != start && parent[current] != "" {
		current = parent[current]
		path = append(path, current)
	}
	path = append(path, start)
	
	// Reverse to get proper order
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	
	return path
}

// GetNode returns a dependency node by tool name.
func (dg *DependencyGraph) GetNode(toolName string) (*DependencyNode, error) {
	dg.mu.RLock()
	defer dg.mu.RUnlock()
	
	if node, exists := dg.nodes[toolName]; exists {
		return node, nil
	}
	
	return nil, errors.WithFields(
		errors.New(errors.ResourceNotFound, "tool not found in graph"),
		errors.Fields{"tool_name": toolName},
	)
}

// GetDependencies returns the direct dependencies of a tool.
func (dg *DependencyGraph) GetDependencies(toolName string) ([]string, error) {
	dg.mu.RLock()
	defer dg.mu.RUnlock()
	
	if deps, exists := dg.edges[toolName]; exists {
		result := make([]string, len(deps))
		copy(result, deps)
		return result, nil
	}
	
	return nil, errors.WithFields(
		errors.New(errors.ResourceNotFound, "tool not found in graph"),
		errors.Fields{"tool_name": toolName},
	)
}

// GetDependents returns the tools that depend on the given tool.
func (dg *DependencyGraph) GetDependents(toolName string) ([]string, error) {
	dg.mu.RLock()
	defer dg.mu.RUnlock()
	
	if dependents, exists := dg.reverseEdges[toolName]; exists {
		result := make([]string, len(dependents))
		copy(result, dependents)
		return result, nil
	}
	
	return nil, errors.WithFields(
		errors.New(errors.ResourceNotFound, "tool not found in graph"),
		errors.Fields{"tool_name": toolName},
	)
}

// DependencyPipeline extends ToolPipeline with dependency-aware execution.
type DependencyPipeline struct {
	*ToolPipeline
	graph *DependencyGraph
	plan  *ExecutionPlan
}

// NewDependencyPipeline creates a new dependency-aware pipeline.
func NewDependencyPipeline(name string, registry core.ToolRegistry, graph *DependencyGraph, options PipelineOptions) (*DependencyPipeline, error) {
	plan, err := graph.CreateExecutionPlan()
	if err != nil {
		return nil, err
	}
	
	basePipeline := NewToolPipeline(name, registry, options)
	
	return &DependencyPipeline{
		ToolPipeline: basePipeline,
		graph:       graph,
		plan:        plan,
	}, nil
}

// ExecuteWithDependencies executes the pipeline using the dependency graph.
func (dp *DependencyPipeline) ExecuteWithDependencies(ctx context.Context, initialInput map[string]interface{}) (*PipelineResult, error) {
	start := time.Now()
	result := &PipelineResult{
		Results:      make([]core.ToolResult, 0),
		StepMetadata: make(map[string]StepMetadata),
		Cache:        make(map[string]core.ToolResult),
		Success:      true,
	}

	// Set up pipeline timeout
	var cancel context.CancelFunc
	if dp.options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, dp.options.Timeout)
		defer cancel()
	}

	// Store results by tool name for dependency resolution
	toolResults := make(map[string]core.ToolResult)
	toolResults["__initial__"] = core.ToolResult{Data: initialInput}

	// Execute phases in order
	for phaseIdx, phase := range dp.plan.Phases {
		phaseStart := time.Now()
		
		if phase.ParallelOk && len(phase.Tools) > 1 {
			// Execute phase tools in parallel
			err := dp.executePhaseParallel(ctx, phase, toolResults, result, phaseIdx)
			if err != nil && dp.options.FailureStrategy == FailFast {
				result.Success = false
				result.Error = err
				result.Duration = time.Since(start)
				return result, err
			}
		} else {
			// Execute phase tools sequentially
			err := dp.executePhaseSequential(ctx, phase, toolResults, result, phaseIdx)
			if err != nil && dp.options.FailureStrategy == FailFast {
				result.Success = false
				result.Error = err
				result.Duration = time.Since(start)
				return result, err
			}
		}
		
		// Phase completed successfully
		fmt.Printf("Phase %d completed in %v\n", phaseIdx, time.Since(phaseStart))
	}

	result.Duration = time.Since(start)
	return result, nil
}

// executePhaseParallel executes a phase with parallel tool execution.
func (dp *DependencyPipeline) executePhaseParallel(ctx context.Context, phase ExecutionPhase, toolResults map[string]core.ToolResult, result *PipelineResult, phaseIdx int) error {
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstError error
	
	// Channel to limit parallelism if needed
	semaphore := make(chan struct{}, phase.MaxParallel)
	
	for toolIdx, toolName := range phase.Tools {
		wg.Add(1)
		go func(tName string, tIdx int) {
			defer wg.Done()
			
			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			
			stepID := fmt.Sprintf("phase_%d_step_%d_%s", phaseIdx, tIdx, tName)
			stepStart := time.Now()
			
			// Prepare input from dependencies
			input, err := dp.prepareToolInput(tName, toolResults, &mu)
			if err != nil {
				mu.Lock()
				if firstError == nil {
					firstError = err
					result.FailedStep = tName
				}
				mu.Unlock()
				return
			}
			
			// Execute tool
			stepResult, err := dp.executeToolStep(ctx, tName, input)
			
			mu.Lock()
			defer mu.Unlock()
			
			if err != nil {
				if firstError == nil {
					firstError = err
					result.FailedStep = tName
				}
			} else {
				toolResults[tName] = stepResult
				result.Results = append(result.Results, stepResult)
			}
			
			result.StepMetadata[stepID] = StepMetadata{
				ToolName: tName,
				Duration: time.Since(stepStart),
				Success:  err == nil,
			}
		}(toolName, toolIdx)
	}
	
	wg.Wait()
	return firstError
}

// executePhaseSequential executes a phase with sequential tool execution.
func (dp *DependencyPipeline) executePhaseSequential(ctx context.Context, phase ExecutionPhase, toolResults map[string]core.ToolResult, result *PipelineResult, phaseIdx int) error {
	for toolIdx, toolName := range phase.Tools {
		stepID := fmt.Sprintf("phase_%d_step_%d_%s", phaseIdx, toolIdx, toolName)
		stepStart := time.Now()
		
		// Prepare input from dependencies (no mutex needed for sequential execution)
		input, err := dp.prepareToolInput(toolName, toolResults, nil)
		if err != nil {
			result.FailedStep = toolName
			return err
		}
		
		// Execute tool
		stepResult, err := dp.executeToolStep(ctx, toolName, input)
		if err != nil {
			result.FailedStep = toolName
			if dp.options.FailureStrategy == FailFast {
				return err
			}
		} else {
			toolResults[toolName] = stepResult
			result.Results = append(result.Results, stepResult)
		}
		
		result.StepMetadata[stepID] = StepMetadata{
			ToolName: toolName,
			Duration: time.Since(stepStart),
			Success:  err == nil,
		}
	}
	
	return nil
}

// prepareToolInput prepares input for a tool based on its dependencies.
func (dp *DependencyPipeline) prepareToolInput(toolName string, toolResults map[string]core.ToolResult, mu *sync.Mutex) (map[string]interface{}, error) {
	node, err := dp.graph.GetNode(toolName)
	if err != nil {
		return nil, err
	}
	
	input := make(map[string]interface{})
	
	// If no dependencies, use initial input
	if len(node.Dependencies) == 0 {
		if mu != nil {
			mu.Lock()
		}
		initialResult, exists := toolResults["__initial__"]
		if mu != nil {
			mu.Unlock()
		}
		if exists {
			if initialData, ok := initialResult.Data.(map[string]interface{}); ok {
				return initialData, nil
			}
		}
		return input, nil
	}
	
	// Combine outputs from dependencies
	if mu != nil {
		mu.Lock()
	}
	for _, dep := range node.Dependencies {
		if depResult, exists := toolResults[dep]; exists {
			if depData, ok := depResult.Data.(map[string]interface{}); ok {
				// Merge dependency output into input
				for key, value := range depData {
					input[key] = value
				}
			}
		} else {
			if mu != nil {
				mu.Unlock()
			}
			return nil, errors.WithFields(
				errors.New(errors.Unknown, "dependency result not available"),
				errors.Fields{"tool": toolName, "dependency": dep},
			)
		}
	}
	if mu != nil {
		mu.Unlock()
	}
	
	return input, nil
}

// executeToolStep executes a single tool step.
func (dp *DependencyPipeline) executeToolStep(ctx context.Context, toolName string, input map[string]interface{}) (core.ToolResult, error) {
	tool, err := dp.registry.Get(toolName)
	if err != nil {
		return core.ToolResult{}, err
	}
	
	return tool.Execute(ctx, input)
}

// GetExecutionPlan returns the execution plan.
func (dp *DependencyPipeline) GetExecutionPlan() *ExecutionPlan {
	return dp.plan
}