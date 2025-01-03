package main

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	workflows "github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

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

func main() {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)
	apiKey := flag.String("api-key", "", "Anthropic API Key")

	ctx := context.Background()
	err := config.ConfigureDefaultLLM(*apiKey, core.ModelAnthropicSonnet)
	if err != nil {
		logger.Error(ctx, "Failed to configure LLM: %v", err)
	}
	// Example 1: Data Processing
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
		routeStep := createHandlerStep(routeType, prompt)
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

func createClassifierStep() *workflows.Step {
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

func createHandlerStep(routeType string, prompt string) *workflows.Step {
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
	routerWorkflow := workflows.NewRouterWorkflow(agents.NewInMemoryStore(), createClassifierStep())
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
