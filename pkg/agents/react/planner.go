package react

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// Plan represents a structured plan for task execution.
type Plan struct {
	ID          string
	Goal        string
	Steps       []PlanStep
	Strategy    PlanningStrategy
	CreatedAt   time.Time
	EstimatedDuration time.Duration
	Dependencies map[string][]string // step dependencies
}

// PlanStep represents a single step in a plan.
type PlanStep struct {
	ID          string
	Description string
	Tool        string
	Arguments   map[string]interface{}
	Expected    string // expected outcome
	Critical    bool   // if true, failure stops execution
	Parallel    bool   // can be executed in parallel
	DependsOn   []string // IDs of steps this depends on
	Timeout     time.Duration
}

// TaskPlanner implements task planning and decomposition.
type TaskPlanner struct {
	strategy     PlanningStrategy
	maxDepth     int
	planCache    map[string]*Plan
	templateLib  *PlanTemplateLibrary
	decomposer   *TaskDecomposer
}

// PlanTemplateLibrary stores reusable plan templates.
type PlanTemplateLibrary struct {
	templates map[string]*PlanTemplate
}

// PlanTemplate is a reusable plan structure.
type PlanTemplate struct {
	Name        string
	TaskPattern string // regex or keyword pattern
	Steps       []PlanStepTemplate
	SuccessRate float64
	LastUsed    time.Time
}

// PlanStepTemplate is a template for a plan step.
type PlanStepTemplate struct {
	Description   string
	ToolType      string
	ArgumentsFunc func(map[string]interface{}) map[string]interface{}
	Critical      bool
}

// TaskDecomposer breaks down complex tasks into subtasks.
type TaskDecomposer struct {
	maxDepth      int
	decomposition map[string][]string // task patterns to subtask patterns
}

// NewTaskPlanner creates a new task planner.
func NewTaskPlanner(strategy PlanningStrategy, maxDepth int) *TaskPlanner {
	return &TaskPlanner{
		strategy:    strategy,
		maxDepth:    maxDepth,
		planCache:   make(map[string]*Plan),
		templateLib: NewPlanTemplateLibrary(),
		decomposer:  NewTaskDecomposer(maxDepth),
	}
}

// NewPlanTemplateLibrary creates a new plan template library with common templates.
func NewPlanTemplateLibrary() *PlanTemplateLibrary {
	lib := &PlanTemplateLibrary{
		templates: make(map[string]*PlanTemplate),
	}

	// Add common plan templates
	lib.AddTemplate(&PlanTemplate{
		Name:        "research_and_summarize",
		TaskPattern: "research|investigate|analyze|summarize",
		Steps: []PlanStepTemplate{
			{
				Description: "Search for relevant information",
				ToolType:    "search",
				Critical:    true,
			},
			{
				Description: "Extract key facts",
				ToolType:    "extract",
				Critical:    false,
			},
			{
				Description: "Synthesize findings",
				ToolType:    "synthesize",
				Critical:    true,
			},
			{
				Description: "Generate summary",
				ToolType:    "summarize",
				Critical:    true,
			},
		},
		SuccessRate: 0.85,
	})

	lib.AddTemplate(&PlanTemplate{
		Name:        "compare_and_evaluate",
		TaskPattern: "compare|evaluate|assess|contrast",
		Steps: []PlanStepTemplate{
			{
				Description: "Gather items to compare",
				ToolType:    "gather",
				Critical:    true,
			},
			{
				Description: "Extract features",
				ToolType:    "extract",
				Critical:    true,
			},
			{
				Description: "Perform comparison",
				ToolType:    "compare",
				Critical:    true,
			},
			{
				Description: "Generate evaluation",
				ToolType:    "evaluate",
				Critical:    true,
			},
		},
		SuccessRate: 0.80,
	})

	lib.AddTemplate(&PlanTemplate{
		Name:        "calculate_and_verify",
		TaskPattern: "calculate|compute|solve|verify",
		Steps: []PlanStepTemplate{
			{
				Description: "Parse problem",
				ToolType:    "parse",
				Critical:    true,
			},
			{
				Description: "Perform calculation",
				ToolType:    "calculate",
				Critical:    true,
			},
			{
				Description: "Verify result",
				ToolType:    "verify",
				Critical:    false,
			},
		},
		SuccessRate: 0.90,
	})

	return lib
}

// NewTaskDecomposer creates a new task decomposer.
func NewTaskDecomposer(maxDepth int) *TaskDecomposer {
	decomposer := &TaskDecomposer{
		maxDepth:      maxDepth,
		decomposition: make(map[string][]string),
	}

	// Add common decomposition patterns
	decomposer.decomposition["analyze_dataset"] = []string{
		"load_data",
		"validate_data",
		"compute_statistics",
		"identify_patterns",
		"generate_insights",
	}

	decomposer.decomposition["write_code"] = []string{
		"understand_requirements",
		"design_architecture",
		"implement_functions",
		"write_tests",
		"refactor_optimize",
	}

	decomposer.decomposition["debug_issue"] = []string{
		"reproduce_issue",
		"identify_root_cause",
		"develop_fix",
		"test_fix",
		"verify_resolution",
	}

	return decomposer
}

// CreatePlan creates an execution plan for a task.
func (tp *TaskPlanner) CreatePlan(ctx context.Context, input map[string]interface{}, tools []core.Tool) (*Plan, error) {
	logger := logging.GetLogger()
	logger.Info(ctx, "Creating plan with strategy: %v", tp.strategy)

	// Extract task from input
	task, ok := input["task"].(string)
	if !ok {
		return nil, fmt.Errorf("no task found in input")
	}

	// Check cache first
	if cachedPlan, exists := tp.planCache[task]; exists {
		logger.Debug(ctx, "Using cached plan for task: %s", task)
		return cachedPlan, nil
	}

	var plan *Plan
	var err error

	switch tp.strategy {
	case DecompositionFirst:
		plan, err = tp.createDecompositionPlan(ctx, task, input, tools)
	case Interleaved:
		plan, err = tp.createInterleavedPlan(ctx, task, input, tools)
	default:
		plan, err = tp.createDecompositionPlan(ctx, task, input, tools)
	}

	if err != nil {
		return nil, err
	}

	// Optimize plan
	plan = tp.optimizePlan(ctx, plan, tools)

	// Cache the plan
	tp.planCache[task] = plan

	logger.Info(ctx, "Created plan with %d steps", len(plan.Steps))
	return plan, nil
}

// createDecompositionPlan creates a plan by decomposing the task upfront.
func (tp *TaskPlanner) createDecompositionPlan(ctx context.Context, task string, input map[string]interface{}, tools []core.Tool) (*Plan, error) {
	logger := logging.GetLogger()

	// First, check if we have a template
	template := tp.templateLib.FindTemplate(task)
	if template != nil {
		logger.Debug(ctx, "Using template: %s", template.Name)
		return tp.createPlanFromTemplate(ctx, task, template, input, tools)
	}

	// Otherwise, decompose the task
	subtasks := tp.decomposer.Decompose(task, tp.maxDepth)

	plan := &Plan{
		ID:           fmt.Sprintf("plan_%d", time.Now().UnixNano()),
		Goal:         task,
		Steps:        make([]PlanStep, 0),
		Strategy:     DecompositionFirst,
		CreatedAt:    time.Now(),
		Dependencies: make(map[string][]string),
	}

	// Create steps for each subtask
	for i, subtask := range subtasks {
		step := tp.createStep(fmt.Sprintf("step_%d", i), subtask, tools)

		// Set dependencies (each step depends on the previous one by default)
		if i > 0 {
			step.DependsOn = []string{fmt.Sprintf("step_%d", i-1)}
			plan.Dependencies[step.ID] = step.DependsOn
		}

		plan.Steps = append(plan.Steps, step)
	}

	// Estimate duration
	plan.EstimatedDuration = time.Duration(len(plan.Steps)) * 30 * time.Second

	return plan, nil
}

// createInterleavedPlan creates an adaptive plan.
func (tp *TaskPlanner) createInterleavedPlan(ctx context.Context, task string, input map[string]interface{}, tools []core.Tool) (*Plan, error) {
	logger := logging.GetLogger()
	logger.Debug(ctx, "Creating interleaved plan for task: %s", task)

	// Start with a minimal plan
	plan := &Plan{
		ID:           fmt.Sprintf("plan_%d", time.Now().UnixNano()),
		Goal:         task,
		Steps:        make([]PlanStep, 0),
		Strategy:     Interleaved,
		CreatedAt:    time.Now(),
		Dependencies: make(map[string][]string),
	}

	// Create initial exploration step
	initialStep := PlanStep{
		ID:          "step_0",
		Description: "Explore task requirements",
		Tool:        tp.selectBestTool("explore", tools),
		Arguments:   map[string]interface{}{"task": task},
		Expected:    "Understanding of task requirements",
		Critical:    true,
		Parallel:    false,
		Timeout:     30 * time.Second,
	}
	plan.Steps = append(plan.Steps, initialStep)

	// Add placeholder for adaptive steps
	adaptiveStep := PlanStep{
		ID:          "step_adaptive",
		Description: "Adaptive execution based on initial results",
		Tool:        "adaptive",
		Arguments:   map[string]interface{}{"mode": "interleaved"},
		Expected:    "Task completion",
		Critical:    true,
		Parallel:    false,
		DependsOn:   []string{"step_0"},
		Timeout:     2 * time.Minute,
	}
	plan.Steps = append(plan.Steps, adaptiveStep)

	plan.EstimatedDuration = 3 * time.Minute // Conservative estimate for adaptive execution

	return plan, nil
}

// createPlanFromTemplate creates a plan from a template.
func (tp *TaskPlanner) createPlanFromTemplate(ctx context.Context, task string, template *PlanTemplate, input map[string]interface{}, tools []core.Tool) (*Plan, error) {
	plan := &Plan{
		ID:           fmt.Sprintf("plan_%d", time.Now().UnixNano()),
		Goal:         task,
		Steps:        make([]PlanStep, 0),
		Strategy:     tp.strategy,
		CreatedAt:    time.Now(),
		Dependencies: make(map[string][]string),
	}

	for i, stepTemplate := range template.Steps {
		step := PlanStep{
			ID:          fmt.Sprintf("step_%d", i),
			Description: stepTemplate.Description,
			Tool:        tp.selectBestTool(stepTemplate.ToolType, tools),
			Critical:    stepTemplate.Critical,
			Parallel:    false, // Will be optimized later
			Timeout:     30 * time.Second,
		}

		// Generate arguments if function provided
		if stepTemplate.ArgumentsFunc != nil {
			step.Arguments = stepTemplate.ArgumentsFunc(input)
		} else {
			step.Arguments = input
		}

		// Set dependencies
		if i > 0 && !step.Parallel {
			step.DependsOn = []string{fmt.Sprintf("step_%d", i-1)}
			plan.Dependencies[step.ID] = step.DependsOn
		}

		plan.Steps = append(plan.Steps, step)
	}

	// Update template usage
	template.LastUsed = time.Now()

	return plan, nil
}

// createStep creates a plan step for a subtask.
func (tp *TaskPlanner) createStep(id, subtask string, tools []core.Tool) PlanStep {
	// Determine the best tool for this subtask
	tool := tp.selectBestTool(subtask, tools)

	return PlanStep{
		ID:          id,
		Description: subtask,
		Tool:        tool,
		Arguments:   map[string]interface{}{"subtask": subtask},
		Expected:    fmt.Sprintf("Completion of: %s", subtask),
		Critical:    tp.isStepCritical(subtask),
		Parallel:    tp.canParallelize(subtask),
		Timeout:     30 * time.Second,
	}
}

// selectBestTool selects the most appropriate tool for a task.
func (tp *TaskPlanner) selectBestTool(taskType string, tools []core.Tool) string {
	// Simple heuristic - in practice, this could use more sophisticated matching
	taskLower := strings.ToLower(taskType)

	for _, tool := range tools {
		toolName := strings.ToLower(tool.Name())
		toolDesc := strings.ToLower(tool.Description())

		// Check if tool name or description matches task type
		if strings.Contains(toolName, taskLower) || strings.Contains(toolDesc, taskLower) {
			return tool.Name()
		}
	}

	// Default to first available tool
	if len(tools) > 0 {
		return tools[0].Name()
	}

	return "default"
}

// isStepCritical determines if a step is critical for task completion.
func (tp *TaskPlanner) isStepCritical(subtask string) bool {
	criticalKeywords := []string{
		"validate", "verify", "essential", "required", "must", "critical",
		"calculate", "compute", "determine", "decide",
	}

	subtaskLower := strings.ToLower(subtask)
	for _, keyword := range criticalKeywords {
		if strings.Contains(subtaskLower, keyword) {
			return true
		}
	}

	return false
}

// canParallelize determines if a step can be executed in parallel.
func (tp *TaskPlanner) canParallelize(subtask string) bool {
	// Steps that typically can't be parallelized
	sequentialKeywords := []string{
		"then", "after", "based on", "using result", "verify", "validate",
	}

	subtaskLower := strings.ToLower(subtask)
	for _, keyword := range sequentialKeywords {
		if strings.Contains(subtaskLower, keyword) {
			return false
		}
	}

	// Steps that typically can be parallelized
	parallelKeywords := []string{
		"gather", "collect", "fetch", "retrieve", "search",
	}

	for _, keyword := range parallelKeywords {
		if strings.Contains(subtaskLower, keyword) {
			return true
		}
	}

	return false
}

// optimizePlan optimizes a plan for efficiency.
func (tp *TaskPlanner) optimizePlan(ctx context.Context, plan *Plan, tools []core.Tool) *Plan {
	logger := logging.GetLogger()
	logger.Debug(ctx, "Optimizing plan with %d steps", len(plan.Steps))

	// Identify parallelization opportunities
	tp.identifyParallelSteps(plan)

	// Optimize tool selection
	tp.optimizeToolSelection(plan, tools)

	// Reorder steps for efficiency
	tp.reorderSteps(plan)

	return plan
}

// identifyParallelSteps identifies steps that can run in parallel.
func (tp *TaskPlanner) identifyParallelSteps(plan *Plan) {
	// Build dependency graph
	dependents := make(map[string][]string)
	for _, step := range plan.Steps {
		for _, dep := range step.DependsOn {
			dependents[dep] = append(dependents[dep], step.ID)
		}
	}

	// Find steps with no dependencies or same dependencies
	for i := range plan.Steps {
		for j := i + 1; j < len(plan.Steps); j++ {
			if tp.canRunInParallel(&plan.Steps[i], &plan.Steps[j], dependents) {
				plan.Steps[i].Parallel = true
				plan.Steps[j].Parallel = true
			}
		}
	}
}

// canRunInParallel checks if two steps can run in parallel.
func (tp *TaskPlanner) canRunInParallel(step1, step2 *PlanStep, dependents map[string][]string) bool {
	// Check if one depends on the other
	for _, dep := range step1.DependsOn {
		if dep == step2.ID {
			return false
		}
	}
	for _, dep := range step2.DependsOn {
		if dep == step1.ID {
			return false
		}
	}

	// Check if they have the same dependencies
	if len(step1.DependsOn) == len(step2.DependsOn) {
		depMap := make(map[string]bool)
		for _, dep := range step1.DependsOn {
			depMap[dep] = true
		}
		for _, dep := range step2.DependsOn {
			if !depMap[dep] {
				return false
			}
		}
		return true
	}

	return false
}

// optimizeToolSelection optimizes tool selection for each step.
func (tp *TaskPlanner) optimizeToolSelection(plan *Plan, tools []core.Tool) {
	// Group steps by tool to identify potential batching opportunities
	toolGroups := make(map[string][]int)
	for i, step := range plan.Steps {
		toolGroups[step.Tool] = append(toolGroups[step.Tool], i)
	}

	// Look for opportunities to batch operations
	for _, indices := range toolGroups {
		if len(indices) > 2 {
			// Consider using a more powerful tool or batching
			for _, idx := range indices {
				plan.Steps[idx].Arguments["batch_mode"] = true
			}
		}
	}
}

// reorderSteps reorders steps for optimal execution.
func (tp *TaskPlanner) reorderSteps(plan *Plan) {
	// Topological sort based on dependencies
	sorted := tp.topologicalSort(plan.Steps)
	plan.Steps = sorted
}

// topologicalSort performs topological sorting of plan steps.
func (tp *TaskPlanner) topologicalSort(steps []PlanStep) []PlanStep {
	// Build adjacency list
	graph := make(map[string][]string)
	inDegree := make(map[string]int)

	for _, step := range steps {
		if _, exists := inDegree[step.ID]; !exists {
			inDegree[step.ID] = 0
		}
		for _, dep := range step.DependsOn {
			graph[dep] = append(graph[dep], step.ID)
			inDegree[step.ID]++
		}
	}

	// Find all nodes with no incoming edges
	queue := make([]string, 0)
	for _, step := range steps {
		if inDegree[step.ID] == 0 {
			queue = append(queue, step.ID)
		}
	}

	// Process queue
	sorted := make([]PlanStep, 0)
	stepMap := make(map[string]PlanStep)
	for _, step := range steps {
		stepMap[step.ID] = step
	}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		sorted = append(sorted, stepMap[current])

		// Reduce in-degree for neighbors
		for _, neighbor := range graph[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// If we couldn't sort all steps, return original order
	if len(sorted) != len(steps) {
		return steps
	}

	return sorted
}

// Decompose breaks down a task into subtasks.
func (td *TaskDecomposer) Decompose(task string, currentDepth int) []string {
	if currentDepth <= 0 {
		return []string{task}
	}

	taskLower := strings.ToLower(task)

	// Check for known decomposition patterns
	for pattern, subtasks := range td.decomposition {
		if strings.Contains(taskLower, pattern) {
			return subtasks
		}
	}

	// Generic decomposition based on task complexity
	return td.genericDecompose(task)
}

// genericDecompose provides generic task decomposition.
func (td *TaskDecomposer) genericDecompose(task string) []string {
	// Simple heuristic-based decomposition
	subtasks := []string{}

	// Add understanding phase
	subtasks = append(subtasks, fmt.Sprintf("Understand: %s", task))

	// Add execution phase
	if strings.Contains(strings.ToLower(task), "and") {
		// Split compound tasks
		parts := strings.Split(task, "and")
		for _, part := range parts {
			subtasks = append(subtasks, fmt.Sprintf("Execute: %s", strings.TrimSpace(part)))
		}
	} else {
		subtasks = append(subtasks, fmt.Sprintf("Execute: %s", task))
	}

	// Add verification phase
	subtasks = append(subtasks, fmt.Sprintf("Verify: %s completed successfully", task))

	return subtasks
}

// AddTemplate adds a plan template to the library.
func (ptl *PlanTemplateLibrary) AddTemplate(template *PlanTemplate) {
	ptl.templates[template.Name] = template
}

// FindTemplate finds a matching template for a task.
func (ptl *PlanTemplateLibrary) FindTemplate(task string) *PlanTemplate {
	taskLower := strings.ToLower(task)

	var bestMatch *PlanTemplate
	var bestScore float64

	for _, template := range ptl.templates {
		// Simple keyword matching - could be enhanced with regex or NLP
		keywords := strings.Split(template.TaskPattern, "|")
		matchScore := 0.0

		for _, keyword := range keywords {
			if strings.Contains(taskLower, keyword) {
				matchScore += 1.0
			}
		}

		// Weight by success rate
		weightedScore := matchScore * template.SuccessRate

		if weightedScore > bestScore {
			bestScore = weightedScore
			bestMatch = template
		}
	}

	if bestScore > 0.5 {
		return bestMatch
	}

	return nil
}

// ValidatePlan checks if a plan is valid and executable.
func (tp *TaskPlanner) ValidatePlan(plan *Plan) error {
	if plan == nil {
		return fmt.Errorf("plan is nil")
	}

	if len(plan.Steps) == 0 {
		return fmt.Errorf("plan has no steps")
	}

	// Check for circular dependencies
	if tp.hasCircularDependencies(plan) {
		return fmt.Errorf("plan has circular dependencies")
	}

	// Check that all dependencies exist
	stepIDs := make(map[string]bool)
	for _, step := range plan.Steps {
		stepIDs[step.ID] = true
	}

	for _, step := range plan.Steps {
		for _, dep := range step.DependsOn {
			if !stepIDs[dep] {
				return fmt.Errorf("step %s depends on non-existent step %s", step.ID, dep)
			}
		}
	}

	return nil
}

// hasCircularDependencies checks for circular dependencies in a plan.
func (tp *TaskPlanner) hasCircularDependencies(plan *Plan) bool {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	var hasCycle func(stepID string) bool
	hasCycle = func(stepID string) bool {
		visited[stepID] = true
		recStack[stepID] = true

		// Find the step
		var currentStep *PlanStep
		for _, step := range plan.Steps {
			if step.ID == stepID {
				currentStep = &step
				break
			}
		}

		if currentStep != nil {
			for _, dep := range currentStep.DependsOn {
				if !visited[dep] {
					if hasCycle(dep) {
						return true
					}
				} else if recStack[dep] {
					return true
				}
			}
		}

		recStack[stepID] = false
		return false
	}

	for _, step := range plan.Steps {
		if !visited[step.ID] {
			if hasCycle(step.ID) {
				return true
			}
		}
	}

	return false
}

// GetPlanMetrics returns metrics about a plan.
func (tp *TaskPlanner) GetPlanMetrics(plan *Plan) map[string]interface{} {
	parallelSteps := 0
	criticalSteps := 0

	for _, step := range plan.Steps {
		if step.Parallel {
			parallelSteps++
		}
		if step.Critical {
			criticalSteps++
		}
	}

	return map[string]interface{}{
		"total_steps":     len(plan.Steps),
		"parallel_steps":  parallelSteps,
		"critical_steps":  criticalSteps,
		"estimated_time":  plan.EstimatedDuration,
		"strategy":        plan.Strategy,
		"parallelization": float64(parallelSteps) / float64(len(plan.Steps)),
	}
}
