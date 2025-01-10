package main

import (
	"context"
	"fmt"

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

// Example processor implementation
type ExampleProcessor struct{}

func (p *ExampleProcessor) Process(ctx context.Context, task agents.Task, taskContext map[string]interface{}) (interface{}, error) {
	// Create a logger to help us understand what's happening
	logger := logging.GetLogger()
	logger.Info(ctx, "Processing task: %s (Type: %s)", task.ID, task.Type)

	// Process different task types
	switch task.Type {
	case "analysis":
		return p.handleAnalysisTask(task, taskContext)
	case "decomposition":
		return p.handleDecompositionTask(task, taskContext)
	case "formatting":
		return p.handleFormattingTask(task, taskContext)
	default:
		// Instead of returning an error, let's handle any task type
		return p.handleGenericTask(task, taskContext)
	}
}

func (p *ExampleProcessor) handleAnalysisTask(task agents.Task, taskContext map[string]interface{}) (interface{}, error) {
	// Simulate analysis work
	result := fmt.Sprintf("Completed analysis for: %s", task.Metadata)
	return result, nil
}

func (p *ExampleProcessor) handleDecompositionTask(task agents.Task, taskContext map[string]interface{}) (interface{}, error) {
	// Simulate decomposition work
	result := fmt.Sprintf("Decomposed task: %s", task.Metadata)
	return result, nil
}

func (p *ExampleProcessor) handleFormattingTask(task agents.Task, taskContext map[string]interface{}) (interface{}, error) {
	// Simulate formatting work
	result := fmt.Sprintf("Formatted output for: %s", task.Metadata)
	return result, nil
}

func (p *ExampleProcessor) handleGenericTask(task agents.Task, taskContext map[string]interface{}) (interface{}, error) {
	// Handle any other task type
	result := fmt.Sprintf("Processed task %s: %s", task.ID, task.Metadata)
	return result, nil
}
