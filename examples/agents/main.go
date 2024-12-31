package main

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/modules"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	workflows "github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
)

func CreateDataProcessingWorkflow() (*workflows.ChainWorkflow, error) {
	// Create a new chain workflow with in-memory storage
	workflow := workflows.NewChainWorkflow(agents.NewInMemoryStore())

	// Step 1: Extract numerical values
	extractStep := &workflows.Step{
		ID: "extract_numbers",
		// We use Predict here since we want a straightforward transformation
		Module: modules.Predict(
			"raw_text -> extracted_values",
			WithInstruction(`Extract only the numerical values and their associated metrics from the text.
                Format each as 'value: metric' on a new line.
                Example format:
                92: customer satisfaction
                45%: revenue growth`),
		),
		InputFields:  []string{"raw_text"},
		OutputFields: []string{"extracted_values"},
	}

	// Step 2: Standardize to percentages
	standardizeStep := &workflows.Step{
		ID: "standardize_values",
		Module: modules.Predict(
			"extracted_values -> standardized_values",
			dspy.WithInstruction(`Convert all numerical values to percentages where possible.
                If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
                Keep one number per line.`),
		),
		InputFields:  []string{"extracted_values"},
		OutputFields: []string{"standardized_values"},
	}

	// Step 3: Sort values
	sortStep := &workflows.Step{
		ID: "sort_values",
		Module: modules.Predict(
			"standardized_values -> sorted_values",
			dspy.WithInstruction(`Sort all lines in descending order by numerical value.
                Keep the format 'value: metric' on each line.`),
		),
		InputFields:  []string{"standardized_values"},
		OutputFields: []string{"sorted_values"},
	}

	// Step 4: Format as table
	tableStep := &workflows.Step{
		ID: "format_table",
		Module: modules.Predict(
			"sorted_values -> markdown_table",
			dspy.WithInstruction(`Format the sorted data as a markdown table with columns:
                | Metric | Value |
                |:--|--:|`),
		),
		InputFields:  []string{"sorted_values"},
		OutputFields: []string{"markdown_table"},
	}

	// Add steps to workflow
	if err := workflow.AddStep(extractStep); err != nil {
		return nil, fmt.Errorf("failed to add extract step: %w", err)
	}
	// Add remaining steps...

	return workflow, nil
}

func main() {
	ctx := context.Background()

	// Example 1: Data Processing
	dataWorkflow, err := CreateDataProcessingWorkflow()
	if err != nil {
		log.Fatalf("Failed to create data workflow: %v", err)
	}

	report := `Q3 Performance Summary...`
	result, err := dataWorkflow.Execute(ctx, map[string]interface{}{
		"raw_text": report,
	})
	if err != nil {
		log.Fatalf("Data processing failed: %v", err)
	}
	fmt.Println(result["markdown_table"])
}
