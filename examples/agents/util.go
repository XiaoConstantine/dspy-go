package main

import (
	"context"
	"encoding/xml"
	"fmt"
	"sort"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	workflows "github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func CreateClassifierStep() *workflows.Step {
	// Create a signature for classification that captures reasoning and selection
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "input"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "reasoning", Prefix: "reasoning"}},
			{Field: core.Field{Name: "selection", Prefix: "selection"}},
			{Field: core.Field{Name: "classification", Prefix: "classification"}}, // Required by RouterWorkflow
		},
	).WithInstruction(`Analyze the input and select the most appropriate support team.
    First explain your reasoning, then provide your selection.
    You must classify the ticket into exactly one of these categories: "billing", "technical", "account", or "product".
    Do not use any other classification values.`)
	// Create a specialized predict module that formats the response correctly
	predictModule := modules.NewPredict(signature)

	return &workflows.Step{
		ID:     "support_classifier",
		Module: predictModule,
	}
}

func CreateHandlerStep(routeType string, prompt string) *workflows.Step {
	// Create signature for handling tickets
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "input"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "response"}},
		},
	).WithInstruction(prompt)

	return &workflows.Step{
		ID:     fmt.Sprintf("%s_handler", routeType),
		Module: modules.NewPredict(signature),
	}
}

func CreateRouterWorkflow() *workflows.RouterWorkflow {
	routerWorkflow := workflows.NewRouterWorkflow(agents.NewInMemoryStore(), CreateClassifierStep())
	return routerWorkflow
}

func CreateParallelWorkflow(stakeholders []string) (*workflows.ParallelWorkflow, map[string]interface{}, error) {
	// Create a new parallel workflow with in-memory storage
	workflow := workflows.NewParallelWorkflow(agents.NewInMemoryStore(), 3)
	inputs := make(map[string]interface{})

	for i, stakeholder := range stakeholders {
		step := &workflows.Step{
			ID:     fmt.Sprintf("analyze_stakeholder_%d", i),
			Module: NewStakeholderAnalysis(i),
		}

		if err := workflow.AddStep(step); err != nil {
			return nil, nil, fmt.Errorf("Failed to add step: %v", err)
		}
		inputKey := fmt.Sprintf("analyze_stakeholder_%d_stakeholder_info", i)
		logging.GetLogger().Info(context.Background(), "input key: %s", inputKey)

		inputs[inputKey] = stakeholder
	}

	return workflow, inputs, nil

}

func NewStakeholderAnalysis(index int) core.Module {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: fmt.Sprintf("analyze_stakeholder_%d_stakeholder_info", index)}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "analysis", Prefix: "analysis"}},
		},
	).WithInstruction(`Analyze how market changes will impact this stakeholder group.
        Provide specific impacts and recommended actions.
        Format with clear sections and priorities.`)

	return modules.NewPredict(signature)
}

func CreateDataProcessingWorkflow() (*workflows.ChainWorkflow, error) {
	// Create a new chain workflow with in-memory storage
	workflow := workflows.NewChainWorkflow(agents.NewInMemoryStore())

	// Step 1: Extract numerical values
	extractSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "raw_text"}}},
		[]core.OutputField{{Field: core.Field{Name: "extracted_values", Prefix: "extracted_values:"}}},
	).WithInstruction(`Extract only the numerical values and their associated metrics from the text.
        Format each as 'value: metric' on a new line.
		Example: extracted_values:
        92: customer satisfaction
        45%: revenue growth`)
	extractStep := &workflows.Step{
		ID:     "extract_numbers",
		Module: modules.NewPredict(extractSignature),
	}

	// Step 2: Standardize to percentages
	standardizeSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "extracted_values"}}},
		[]core.OutputField{{Field: core.Field{Name: "standardized_values", Prefix: "standardized_values:"}}},
	).WithInstruction(`Convert all numerical values to percentages where possible.
        If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
        Keep one number per line.
	Example format:
	standardized_values:
	92%: customer satisfaction
	87%: employee satisfaction
	`)

	standardizeStep := &workflows.Step{
		ID:     "standardize_values",
		Module: modules.NewPredict(standardizeSignature),
	}

	// Step 3: Sort values
	sortSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "standardized_values"}}},
		[]core.OutputField{{Field: core.Field{Name: "sorted_values", Prefix: "sorted_values:"}}},
	).WithInstruction(`Sort all lines in descending order by numerical value.
                Keep the format 'value: metric' on each line.
		`)
	sortStep := &workflows.Step{
		ID:     "sort_values",
		Module: modules.NewPredict(sortSignature),
	}

	// Step 4: Format as table
	tableSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "sorted_values"}}},
		[]core.OutputField{{Field: core.Field{Name: "markdown_table", Prefix: "markdown_table:"}}},
	).WithInstruction(`Format the sorted values as a markdown table with two columns:
    - First column: Metric
    - Second column: Value
Format example:
| Metric | Value |
|:--|--:|
|--------|-------|
| Customer Satisfaction | 92% |`)
	tableStep := &workflows.Step{
		ID:     "format_table",
		Module: modules.NewPredict(tableSignature),
	}

	// Add steps to workflow
	if err := workflow.AddStep(extractStep); err != nil {
		return nil, fmt.Errorf("failed to add extract step: %w", err)
	}
	if err := workflow.AddStep(standardizeStep); err != nil {
		return nil, fmt.Errorf("failed to add standardize step: %w", err)
	}
	if err := workflow.AddStep(sortStep); err != nil {
		return nil, fmt.Errorf("failed to add sort step: %w", err)
	}
	if err := workflow.AddStep(tableStep); err != nil {
		return nil, fmt.Errorf("failed to add table step: %w", err)
	}

	return workflow, nil
}

// A simple optimizer following the pattern from the Python implementation.
type PromptingOptimizer struct {
	// Metric evaluates solution quality and returns PASS/FAIL with feedback
	Metric      func(ctx context.Context, solution map[string]interface{}) (pass bool, feedback string)
	MaxAttempts int
	Memory      []string // Track previous attempts like Python version
	Logger      *logging.Logger
}

func NewPromptingOptimizer(
	metric func(ctx context.Context, solution map[string]interface{}) (bool, string),
	maxAttempts int,
) *PromptingOptimizer {
	return &PromptingOptimizer{
		Metric:      metric,
		MaxAttempts: maxAttempts,
		Memory:      make([]string, 0),
		Logger:      logging.GetLogger(),
	}
}

func (o *PromptingOptimizer) Compile(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset, // Kept for interface compatibility but not used
	metric core.Metric, // Kept for interface compatibility but not used
) (core.Program, error) {
	ctx, span := core.StartSpan(ctx, "PromptingOptimization")
	defer core.EndSpan(ctx)

	// Initial attempt without context
	o.Logger.Info(ctx, "Making initial attempt...")
	result, err := program.Execute(ctx, map[string]interface{}{
		"task": "Implement MinStack with O(1) operations",
	})
	if err != nil {
		span.WithError(err)
		return program, fmt.Errorf("initial attempt failed: %w", err)
	}

	// Store first attempt
	if solution, ok := result["solution"].(string); ok {
		o.Memory = append(o.Memory, solution)
		o.Logger.Debug(ctx, "Initial solution:\n%s", solution)
	}

	// Improvement loop
	for attempt := 0; attempt < o.MaxAttempts; attempt++ {
		attemptCtx, attemptSpan := core.StartSpan(ctx, fmt.Sprintf("Attempt_%d", attempt))

		// Evaluate current solution
		pass, feedback := o.Metric(attemptCtx, result)
		attemptSpan.WithAnnotation("feedback", feedback)
		attemptSpan.WithAnnotation("pass", pass)

		o.Logger.Info(ctx, "Attempt %d evaluation - Pass: %v, Feedback: %s", attempt, pass, feedback)

		if pass {
			o.Logger.Info(ctx, "Found satisfactory solution on attempt %d", attempt)
			core.EndSpan(attemptCtx)
			return program, nil
		}

		// Build context from memory like Python version
		context := "Previous attempts:\n"
		for _, m := range o.Memory {
			context += fmt.Sprintf("- %s\n", m)
		}
		context += fmt.Sprintf("\nFeedback: %s", feedback)

		// Try again with feedback
		result, err = program.Execute(attemptCtx, map[string]interface{}{
			"task":    "Implement MinStack with O(1) operations",
			"context": context,
		})
		if err != nil {
			core.EndSpan(attemptCtx)
			return program, fmt.Errorf("attempt %d failed: %w", attempt, err)
		}

		// Store this attempt
		if solution, ok := result["solution"].(string); ok {
			o.Memory = append(o.Memory, solution)
			o.Logger.Debug(ctx, "Attempt %d solution:\n%s", attempt, solution)
		}

		core.EndSpan(attemptCtx)
	}

	return program, fmt.Errorf("failed to find satisfactory solution in %d attempts", o.MaxAttempts)
}

// Custom impl for orchestrator worker example
// XMLTaskParser parses tasks from XML format
type XMLTaskParser struct {
	// Configuration for XML parsing
	RequiredFields []string
}

// XMLTask represents the XML structure of a task
type XMLTask struct {
	XMLName       xml.Name    `xml:"task"`
	ID            string      `xml:"id,attr"`
	Type          string      `xml:"type"`
	ProcessorType string      `xml:"processor"`
	Description   string      `xml:"description"`
	Priority      int         `xml:"priority,attr"`
	Dependencies  []string    `xml:"dependencies>task"`
	Metadata      XMLMetadata `xml:"metadata"`
}

type XMLMetadata struct {
	Items []XMLMetadataItem `xml:"item"`
}

type XMLMetadataItem struct {
	Key   string `xml:"key,attr"`
	Value string `xml:",chardata"`
}

func (p *XMLTaskParser) Parse(analyzerOutput map[string]interface{}) ([]agents.Task, error) {
	tasksXML, ok := analyzerOutput["tasks"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid tasks format in analyzer output")
	}

	// Parse XML tasks
	var xmlTasks struct {
		Tasks []XMLTask `xml:"task"`
	}
	if err := xml.Unmarshal([]byte(tasksXML), &xmlTasks); err != nil {
		return nil, fmt.Errorf("failed to parse XML tasks: %w", err)
	}

	// Convert to Task objects
	tasks := make([]agents.Task, len(xmlTasks.Tasks))
	for i, xmlTask := range xmlTasks.Tasks {
		// Validate required fields
		if err := p.validateTask(xmlTask); err != nil {
			return nil, fmt.Errorf("invalid task %s: %w", xmlTask.ID, err)
		}

		// Convert metadata to map
		metadata := make(map[string]interface{})
		for _, item := range xmlTask.Metadata.Items {
			metadata[item.Key] = item.Value
		}

		tasks[i] = agents.Task{
			ID:            xmlTask.ID,
			Type:          xmlTask.Type,
			ProcessorType: xmlTask.ProcessorType,
			Dependencies:  xmlTask.Dependencies,
			Priority:      xmlTask.Priority,
			Metadata:      metadata,
		}
	}

	return tasks, nil
}

func (p *XMLTaskParser) validateTask(task XMLTask) error {
	if task.ID == "" {
		return fmt.Errorf("missing task ID")
	}
	if task.Type == "" {
		return fmt.Errorf("missing task type")
	}
	if task.ProcessorType == "" {
		return fmt.Errorf("missing processor type")
	}
	return nil
}

// DependencyPlanCreator creates execution plans based on task dependencies
type DependencyPlanCreator struct {
	// Optional configuration for planning
	MaxTasksPerPhase int
}

func NewDependencyPlanCreator(maxTasksPerPhase int) *DependencyPlanCreator {
	if maxTasksPerPhase <= 0 {
		maxTasksPerPhase = 10 // Default value
	}
	return &DependencyPlanCreator{
		MaxTasksPerPhase: maxTasksPerPhase,
	}
}

func (p *DependencyPlanCreator) CreatePlan(tasks []agents.Task) ([][]agents.Task, error) {
	// Build dependency graph
	graph := buildDependencyGraph(tasks)

	// Detect cycles
	if err := detectCycles(graph); err != nil {
		return nil, fmt.Errorf("invalid task dependencies: %w", err)
	}

	// Create phases based on dependencies
	phases := [][]agents.Task{}
	remaining := make(map[string]agents.Task)
	completed := make(map[string]bool)

	// Initialize remaining tasks
	for _, task := range tasks {
		remaining[task.ID] = task
	}

	// Create phases until all tasks are allocated
	for len(remaining) > 0 {
		phase := []agents.Task{}

		// Find tasks with satisfied dependencies
		for _, task := range remaining {
			if canExecute(task, completed) {
				phase = append(phase, task)
				delete(remaining, task.ID)

				// Respect max tasks per phase
				if len(phase) >= p.MaxTasksPerPhase {
					break
				}
			}
		}

		// If no tasks can be executed, we have a problem
		if len(phase) == 0 {
			return nil, fmt.Errorf("circular dependency or missing dependency detected")
		}

		// Sort phase by priority
		sort.Slice(phase, func(i, j int) bool {
			return phase[i].Priority < phase[j].Priority
		})

		phases = append(phases, phase)

		// Mark phase tasks as completed
		for _, task := range phase {
			completed[task.ID] = true
		}
	}

	return phases, nil
}

// Helper function to build dependency graph
func buildDependencyGraph(tasks []agents.Task) map[string][]string {
	graph := make(map[string][]string)
	for _, task := range tasks {
		graph[task.ID] = task.Dependencies
	}
	return graph
}

// Helper function to detect cycles in the dependency graph
func detectCycles(graph map[string][]string) error {
	visited := make(map[string]bool)
	path := make(map[string]bool)

	var checkCycle func(string) error
	checkCycle = func(node string) error {
		visited[node] = true
		path[node] = true

		for _, dep := range graph[node] {
			if !visited[dep] {
				if err := checkCycle(dep); err != nil {
					return err
				}
			} else if path[dep] {
				return fmt.Errorf("cycle detected involving task %s", node)
			}
		}

		path[node] = false
		return nil
	}

	for node := range graph {
		if !visited[node] {
			if err := checkCycle(node); err != nil {
				return err
			}
		}
	}

	return nil
}

// Helper function to check if a task can be executed
func canExecute(task agents.Task, completed map[string]bool) bool {
	for _, dep := range task.Dependencies {
		if !completed[dep] {
			return false
		}
	}
	return true
}

// Example processor implementation
type ExampleProcessor struct{}

func (p *ExampleProcessor) Process(ctx context.Context, task agents.Task, context map[string]interface{}) (interface{}, error) {
	// Process the task based on its type and metadata
	switch task.Type {
	case "example_type":
		return p.handleExampleTask(task, context)
	default:
		return nil, fmt.Errorf("unsupported task type: %s", task.Type)
	}
}

func (p *ExampleProcessor) handleExampleTask(task agents.Task, context map[string]interface{}) (interface{}, error) {
	// Implement your task handling logic here
	return task.Type, nil
}
