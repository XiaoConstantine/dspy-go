package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"

	workflows "github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

func RunChainExample(ctx context.Context, logger *logging.Logger) {
	logger.Info(ctx, "============ Example 1: Chain workflow for structured data extraction and formatting ==============")
	dataWorkflow, err := CreateDataProcessingWorkflow()
	if err != nil {
		logger.Error(ctx, "Failed to create data workflow: %v", err)
	}

	report := `Q3 Performance Summary:Our customer satisfaction score rose to 92 points this quarter.Revenue grew by 45% compared to last year.Market share is now at 23% in our primary market.Customer churn decreased to 5% from 8%.New user acquisition cost is $43 per user.Product adoption rate increased to 78%.Employee satisfaction is at 87 points.Operating margin improved to 34%.`
	result, err := dataWorkflow.Execute(ctx, map[string]interface{}{
		"raw_text": report,
	})
	if err != nil {
		logger.Error(ctx, "Data processing failed: %v", err)
	}
	// Format the output nicely
	if table, ok := result["markdown_table"].(string); ok {
		// clean up any extra whitespace
		lines := strings.Split(strings.TrimSpace(table), "\n")
		for _, line := range lines {
			logger.Info(ctx, strings.TrimSpace(line))
		}
	} else {
		logger.Error(ctx, "invalid table format in result")
	}
	logger.Info(ctx, "================================================================================")
}

func RunParallelExample(ctx context.Context, logger *logging.Logger) {
	logger.Info(ctx, "============ Example 2: Parallelization workflow for stakeholder impact analysis ==============")
	stakeholders := []string{
		`customers:
	       - price sensitive
	       - want better tech
	       - environmental concerns`,

		`employees:
	       - job security worries
	       - need new skills
	       - want clear direction`,

		`investors:
	       - expect growth
	       - want cost control
	       - risk concerns`,

		`suppliers:
	       - capacity constraints
	       - price pressures
	       - tech transitions`,
	}
	workflow, inputs, err := CreateParallelWorkflow(stakeholders)
	logger.Info(ctx, "Inputs: %v", inputs)
	results, err := workflow.Execute(ctx, inputs)
	if err != nil {
		logger.Error(ctx, "Workflow execution failed: %v", err)
	}
	// Print results
	for i := range stakeholders {
		analysisKey := fmt.Sprintf("analyze_stakeholder_%d_analysis", i)
		if analysis, ok := results[analysisKey]; ok {
			fmt.Printf("\n=== Stakeholder Analysis %d ===\n", i+1)
			fmt.Println(analysis)
		}
	}
	logger.Info(ctx, "=================================================")
}

func RunRouteExample(ctx context.Context, logger *logging.Logger) {
	logger.Info(ctx, "============ Example 3: Route workflow for customer support ticket handling ==============")

	supportRoutes := map[string]string{
		"billing": `You are a billing support specialist. Follow these guidelines:
            1. Always start with "Billing Support Response:"
            2. First acknowledge the specific billing issue
            3. Explain any charges or discrepancies clearly
            4. List concrete next steps with timeline
            5. End with payment options if relevant
            
            Keep responses professional but friendly.,
	    Input:`,

		"technical": `You are a technical support engineer. Follow these guidelines:
            1. Always start with "Technical Support Response:"
            2. List exact steps to resolve the issue
            3. Include system requirements if relevant
            4. Provide workarounds for common problems
            5. End with escalation path if needed
            
            Use clear, numbered steps and technical details.
		Input:`,
		"account": `You are an account security specialist. Follow these guidelines:
		1. Always start with "Account Support Response:"
		2. Prioritize account security and verification
	        3. Provide clear steps for account recovery/changes
	        4. Include security tips and warnings
		5. Set clear expectations for resolution time
    
		Maintain a serious, security-focused tone.
		Input:`,
		"product": `You are a product specialist. Follow these guidelines:
    1. Always start with "Product Support Response:"
    2. Focus on feature education and best practices
    3. Include specific examples of usage
    4. Link to relevant documentation sections
    5. Suggest related features that might help
    
    Be educational and encouraging in tone.
		Input:`,
	}
	tickets := []string{
		`Subject: Can't access my account
        Message: Hi, I've been trying to log in for the past hour but keep getting an 'invalid password' error. 
        I'm sure I'm using the right password. Can you help me regain access? This is urgent as I need to 
        submit a report by end of day.
        - John`,
		`Subject: Unexpected charge on my card
    Message: Hello, I just noticed a charge of $49.99 on my credit card from your company, but I thought
    I was on the $29.99 plan. Can you explain this charge and adjust it if it's a mistake?
    Thanks,
    Sarah`,
		`Subject: How to export data?
		Message: I need to export all my project data to Excel. I've looked through the docs but can't
		figure out how to do a bulk export. Is this possible? If so, could you walk me through the steps?:
		Best regards,
		Mike`,
	}
	router := CreateRouterWorkflow()

	for routeType, prompt := range supportRoutes {
		routeStep := CreateHandlerStep(routeType, prompt)
		if err := router.AddStep(routeStep); err != nil {
			logger.Error(ctx, "Failed to add step %s: %v", routeType, err)
			continue
		}
		if err := router.AddRoute(routeType, []*workflows.Step{routeStep}); err != nil {
			logger.Error(ctx, "Failed to add route %s: %v", routeType, err)
		}
	}
	logger.Info(ctx, "Processing support tickets...\n")
	for i, ticket := range tickets {
		logger.Info(ctx, "\nTicket %d:\n", i+1)
		logger.Info(ctx, strings.Repeat("-", 40))
		logger.Info(ctx, ticket)
		logger.Info(ctx, "\nResponse:")
		logger.Info(ctx, strings.Repeat("-", 40))

		response, err := router.Execute(ctx, map[string]interface{}{"input": ticket})
		if err != nil {
			logger.Info(ctx, "Error processing ticket %d: %v", i+1, err)
			continue
		}
		logger.Info(ctx, response["response"].(string))
	}
	logger.Info(ctx, "=================================================")

}

func RunEvalutorOptimizerExample(ctx context.Context, logger *logging.Logger) {
	// Create signature for our task - simpler now that we don't need dataset support
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "task"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "solution", Prefix: "solution"}},
		},
	).WithInstruction(`
        Your task is to implement the requested data structure.
        Consider time and space complexity.
        Provide a complete, correct implementation in Go.
        If context is provided, learn from previous attempts and feedback.
    `)

	// Create predict module with the signature
	predict := modules.NewPredict(signature)

	// Create program that uses predict module
	program := core.NewProgram(
		map[string]core.Module{"predict": predict},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predict.Process(ctx, inputs)
		},
	)

	// Create evaluation metric that matches Python implementation
	metric := func(ctx context.Context, result map[string]interface{}) (bool, string) {
		solution, ok := result["solution"].(string)
		if !ok {
			return false, "No solution provided"
		}

		// Check implementation quality
		issues := make([]string, 0)

		if !strings.Contains(solution, "type MinStack struct") {
			issues = append(issues, "Missing MinStack struct definition")
		}

		if !strings.Contains(solution, "min []int") && !strings.Contains(solution, "minStack []int") {
			issues = append(issues, "Need a way to track minimum values")
		}

		if strings.Contains(solution, "for") || strings.Contains(solution, "range") {
			issues = append(issues, "Operations should be O(1) - avoid loops")
		}

		if !strings.Contains(solution, "func (s *MinStack) GetMin()") {
			issues = append(issues, "Missing GetMin() method")
		}

		// Pass only if no issues found
		if len(issues) == 0 {
			return true, "Implementation meets all requirements"
		}

		return false, strings.Join(issues, ". ")
	}

	ctx = core.WithExecutionState(ctx)
	// Create and use PromptingOptimizer
	optimizer := NewPromptingOptimizer(metric, 5) // Allow 5 attempts

	logger.Info(ctx, "Starting optimization process...")
	optimizedProgram, err := optimizer.Compile(ctx, program, nil, nil)
	if err != nil {
		logger.Error(ctx, "Optimization failed: %v", err)
		return
	}

	// Test the optimized program
	result, err := optimizedProgram.Execute(ctx, map[string]interface{}{
		"task": "Implement MinStack with O(1) operations for push, pop, and getMin",
	})
	if err != nil {
		logger.Error(ctx, "Final execution failed: %v", err)
		return
	}

	if solution, ok := result["solution"].(string); ok {
		logger.Info(ctx, "Final MinStack Implementation:\n%s", solution)
	} else {
		logger.Error(ctx, "No solution in final result")
	}
}

func RunOrchestratorExample(ctx context.Context, logger *logging.Logger) {
	xmlFormat := `Format the tasks section in XML with the following structure:
    <tasks>
        <task id="task_1" type="analysis" processor="example" priority="1">
            <description>Task description here</description>
            <dependencies></dependencies>
            <metadata>
                <item key="resource">cpu</item>
            </metadata>
        </task>
    </tasks>`
	parser := &agents.XMLTaskParser{
		RequiredFields: []string{"id", "type", "processor"},
	}

	planner := agents.NewDependencyPlanCreator(5) // Max 5 tasks per phase

	// Create orchestrator configuration
	config := agents.OrchestrationConfig{
		MaxConcurrent:  3,
		DefaultTimeout: 30 * time.Second,
		RetryConfig: &agents.RetryConfig{
			MaxAttempts:       3,
			BackoffMultiplier: 2.0,
		},
		// Use custom parser and planner
		TaskParser:  parser,
		PlanCreator: planner,
		// Add your custom processors
		CustomProcessors: map[string]agents.TaskProcessor{
			"general": &ExampleProcessor{},
		},
		AnalyzerConfig: agents.AnalyzerConfig{
			FormatInstructions: xmlFormat,
			Considerations: []string{
				"Task dependencies and optimal execution order",
				"Opportunities for parallel execution",
				"Required processor types for each task",
				"Task priorities and resource requirements",
			},
		},
		Options: core.WithGenerateOptions(
			core.WithTemperature(0.3),
			core.WithMaxTokens(8192),
		),
	}

	// Create orchestrator
	orchestrator := agents.NewFlexibleOrchestrator(agents.NewInMemoryStore(), config)

	// The analyzer will return tasks in XML format that our parser understands
	task := "Your high-level task description"
	context := map[string]interface{}{
		"key": "value",
	}

	// Process the task
	ctx = core.WithExecutionState(ctx)
	result, err := orchestrator.Process(ctx, task, context)
	if err != nil {
		logger.Error(ctx, "Orchestration failed: %v", err)
		// Log more details about failed tasks
		for taskID, taskErr := range result.FailedTasks {
			logger.Error(ctx, "Task %s failed: %v", taskID, taskErr)
		}
	}
	// Log successful tasks with details
	for taskID, taskResult := range result.CompletedTasks {
		logger.Info(ctx, "Task %s completed successfully with result: %v", taskID, taskResult)
	}

	// Handle results
	logger.Info(ctx, "Orchestration completed with %d successful tasks and %d failures\n",
		len(result.CompletedTasks), len(result.FailedTasks))
}

func main() {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	fileOutput, err := logging.NewFileOutput(
		filepath.Join(".", "dspy.log"),
		logging.WithRotation(10*1024*1024, 5), // 10MB max size, keep 5 files
		logging.WithJSONFormat(true),          // Use JSON format
	)
	if err != nil {
		fmt.Printf("Failed to create file output: %v\n", err)
		os.Exit(1)
	}
	logger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs:  []logging.Output{output, fileOutput},
	})
	logging.SetLogger(logger)
	apiKey := flag.String("api-key", "", "Anthropic API Key")

	ctx := core.WithExecutionState(context.Background())
	logger.Info(ctx, "Starting application")
	logger.Debug(ctx, "This is a debug message")
	logger.Warn(ctx, "This is a warning message")
	llms.EnsureFactory()
	err = core.ConfigureDefaultLLM(*apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Error(ctx, "Failed to configure LLM: %v", err)
	}
	// RunChainExample(ctx, logger)
	// RunParallelExample(ctx, logger)
	// RunRouteExample(ctx, logger)
	// RunEvalutorOptimizerExample(ctx, logger)
	RunOrchestratorExample(ctx, logger)
}
