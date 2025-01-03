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
