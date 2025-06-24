package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Example tools for demonstrating tool chaining

// DataExtractorTool extracts specific fields from input.
type DataExtractorTool struct{}

func (d *DataExtractorTool) Name() string { return "data_extractor" }
func (d *DataExtractorTool) Description() string {
	return "Extracts and structures data from raw input"
}
func (d *DataExtractorTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         "data_extractor",
		Description:  d.Description(),
		Capabilities: []string{"extraction", "parsing", "data_processing"},
		Version:      "1.0.0",
	}
}
func (d *DataExtractorTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (d *DataExtractorTool) Validate(params map[string]interface{}) error      { return nil }
func (d *DataExtractorTool) InputSchema() models.InputSchema                   { return models.InputSchema{} }

func (d *DataExtractorTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	rawData, ok := params["raw_data"].(string)
	if !ok {
		rawData = "sample data"
	}

	// Simulate extraction process
	time.Sleep(50 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"extracted_fields": map[string]interface{}{
				"title":   "Extracted Title",
				"content": rawData,
				"metadata": map[string]interface{}{
					"source":    "data_extractor",
					"timestamp": time.Now().Unix(),
				},
			},
			"confidence": 0.95,
			"status":     "extracted",
		},
		Metadata: map[string]interface{}{
			"processing_time_ms": 50,
			"tool":               "data_extractor",
		},
	}, nil
}

// DataValidatorTool validates extracted data.
type DataValidatorTool struct{}

func (d *DataValidatorTool) Name() string        { return "data_validator" }
func (d *DataValidatorTool) Description() string { return "Validates data integrity and completeness" }
func (d *DataValidatorTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         "data_validator",
		Description:  d.Description(),
		Capabilities: []string{"validation", "quality_check", "data_integrity"},
		Version:      "1.2.0",
	}
}
func (d *DataValidatorTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (d *DataValidatorTool) Validate(params map[string]interface{}) error      { return nil }
func (d *DataValidatorTool) InputSchema() models.InputSchema                   { return models.InputSchema{} }

func (d *DataValidatorTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	confidence, _ := params["confidence"].(float64)

	// Simulate validation process
	time.Sleep(30 * time.Millisecond)

	isValid := confidence > 0.8

	return core.ToolResult{
		Data: map[string]interface{}{
			"validation_result": map[string]interface{}{
				"is_valid":      isValid,
				"quality_score": confidence * 100,
				"issues":        []string{},
			},
			"validated_data":       params,
			"validation_timestamp": time.Now().Unix(),
			"status":               "validated",
		},
		Metadata: map[string]interface{}{
			"processing_time_ms": 30,
			"tool":               "data_validator",
		},
	}, nil
}

// DataEnricherTool enriches data with additional information.
type DataEnricherTool struct{}

func (d *DataEnricherTool) Name() string { return "data_enricher" }
func (d *DataEnricherTool) Description() string {
	return "Enriches data with additional context and metadata"
}
func (d *DataEnricherTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         "data_enricher",
		Description:  d.Description(),
		Capabilities: []string{"enrichment", "augmentation", "context_addition"},
		Version:      "2.1.0",
	}
}
func (d *DataEnricherTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (d *DataEnricherTool) Validate(params map[string]interface{}) error      { return nil }
func (d *DataEnricherTool) InputSchema() models.InputSchema                   { return models.InputSchema{} }

func (d *DataEnricherTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	// Simulate enrichment process
	time.Sleep(40 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"enriched_data": params,
			"additional_context": map[string]interface{}{
				"tags":             []string{"processed", "validated", "enriched"},
				"category":         "enriched_content",
				"enrichment_level": "standard",
			},
			"external_references": []string{
				"ref_001", "ref_002",
			},
			"status": "enriched",
		},
		Metadata: map[string]interface{}{
			"processing_time_ms": 40,
			"tool":               "data_enricher",
		},
	}, nil
}

// DataTransformerTool transforms data into final format.
type DataTransformerTool struct{}

func (d *DataTransformerTool) Name() string { return "data_transformer" }
func (d *DataTransformerTool) Description() string {
	return "Transforms data into the final output format"
}
func (d *DataTransformerTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         "data_transformer",
		Description:  d.Description(),
		Capabilities: []string{"transformation", "formatting", "output_generation"},
		Version:      "3.0.0",
	}
}
func (d *DataTransformerTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (d *DataTransformerTool) Validate(params map[string]interface{}) error      { return nil }
func (d *DataTransformerTool) InputSchema() models.InputSchema                   { return models.InputSchema{} }

func (d *DataTransformerTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	// Simulate transformation process
	time.Sleep(25 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"final_output": map[string]interface{}{
				"processed_data": params,
				"format":         "final",
				"version":        "1.0",
				"generated_at":   time.Now().Format(time.RFC3339),
			},
			"transformation_summary": map[string]interface{}{
				"steps_completed":       4,
				"total_processing_time": "~145ms",
				"final_quality":         "high",
			},
			"status": "completed",
		},
		Metadata: map[string]interface{}{
			"processing_time_ms": 25,
			"tool":               "data_transformer",
		},
	}, nil
}

func setupRegistry() core.ToolRegistry {
	registry := tools.NewInMemoryToolRegistry()

	// Register all tools
	tools := []core.Tool{
		&DataExtractorTool{},
		&DataValidatorTool{},
		&DataEnricherTool{},
		&DataTransformerTool{},
	}

	for _, tool := range tools {
		if err := registry.Register(tool); err != nil {
			log.Fatalf("Failed to register tool %s: %v", tool.Name(), err)
		}
	}

	return registry
}

func demonstrateBasicPipeline(registry core.ToolRegistry) {
	fmt.Println("\n🔧 Basic Pipeline Demonstration")
	fmt.Println("===============================")

	// Create a basic sequential pipeline
	pipeline, err := tools.NewPipelineBuilder("data_processing", registry).
		Step("data_extractor").
		Step("data_validator").
		Step("data_enricher").
		Step("data_transformer").
		FailFast().
		EnableCaching().
		Build()

	if err != nil {
		log.Fatalf("Failed to build pipeline: %v", err)
	}

	ctx := context.Background()
	input := map[string]interface{}{
		"raw_data": "Sample input data for processing pipeline",
	}

	fmt.Printf("Processing input: %s\n", input["raw_data"])

	start := time.Now()
	result, err := pipeline.Execute(ctx, input)
	duration := time.Since(start)

	if err != nil {
		log.Fatalf("Pipeline execution failed: %v", err)
	}

	fmt.Printf("✅ Pipeline completed successfully in %v\n", duration)
	fmt.Printf("📊 Steps executed: %d\n", len(result.Results))
	fmt.Printf("📈 Success: %t\n", result.Success)

	// Show final result
	if len(result.Results) > 0 {
		finalResult := result.Results[len(result.Results)-1]
		if data, ok := finalResult.Data.(map[string]interface{}); ok {
			if status, ok := data["status"].(string); ok {
				fmt.Printf("🎯 Final status: %s\n", status)
			}
		}
	}
}

func demonstrateDataTransformations(registry core.ToolRegistry) {
	fmt.Println("\n🔄 Data Transformation Demonstration")
	fmt.Println("====================================")

	// Create transformer that extracts validation results
	validationTransformer := tools.TransformExtractField("validation_result")

	// Create transformer that adds processing metadata
	enrichmentTransformer := tools.TransformAddConstant(map[string]interface{}{
		"pipeline_id":      "demo_pipeline_001",
		"processing_stage": "enrichment",
	})

	// Chain transformers
	chainedTransformer := tools.TransformChain(
		tools.TransformRename(map[string]string{
			"status": "processing_status",
		}),
		enrichmentTransformer,
		tools.TransformFilter([]string{"validation_result", "pipeline_id", "processing_stage", "processing_status"}),
	)

	pipeline, err := tools.NewPipelineBuilder("transformation_demo", registry).
		Step("data_extractor").
		StepWithTransformer("data_validator", validationTransformer).
		StepWithTransformer("data_enricher", chainedTransformer).
		Build()

	if err != nil {
		log.Fatalf("Failed to build transformation pipeline: %v", err)
	}

	ctx := context.Background()
	input := map[string]interface{}{
		"raw_data": "Data for transformation demonstration",
	}

	fmt.Printf("Processing with transformations...\n")

	result, err := pipeline.Execute(ctx, input)
	if err != nil {
		log.Fatalf("Transformation pipeline failed: %v", err)
	}

	fmt.Printf("✅ Transformation pipeline completed\n")
	fmt.Printf("📊 Data transformations applied: %d\n", len(result.Results))

	// Show transformation effects
	for i, stepResult := range result.Results {
		fmt.Printf("  Step %d result keys: ", i+1)
		if data, ok := stepResult.Data.(map[string]interface{}); ok {
			keys := make([]string, 0, len(data))
			for k := range data {
				keys = append(keys, k)
			}
			fmt.Printf("%v\n", keys)
		}
	}
}

func demonstrateConditionalExecution(registry core.ToolRegistry) {
	fmt.Println("\n⚡ Conditional Execution Demonstration")
	fmt.Println("=====================================")

	pipeline, err := tools.NewPipelineBuilder("conditional_demo", registry).
		Step("data_extractor").
		Step("data_validator").
		ConditionalStep("data_enricher",
			tools.ConditionExists("validation_result"),
			tools.ConditionEquals("status", "validated")).
		ConditionalStep("data_transformer",
			tools.ConditionEquals("status", "enriched")).
		Build()

	if err != nil {
		log.Fatalf("Failed to build conditional pipeline: %v", err)
	}

	ctx := context.Background()
	input := map[string]interface{}{
		"raw_data": "Data for conditional execution test",
	}

	fmt.Printf("Testing conditional execution...\n")

	result, err := pipeline.Execute(ctx, input)
	if err != nil {
		log.Fatalf("Conditional pipeline failed: %v", err)
	}

	fmt.Printf("✅ Conditional pipeline completed\n")
	fmt.Printf("📊 Steps executed: %d (conditional steps may have been skipped)\n", len(result.Results))

	// Show which steps executed
	for stepID, metadata := range result.StepMetadata {
		status := "executed"
		if !metadata.Success {
			status = "failed"
		}
		fmt.Printf("  %s: %s (%s)\n", stepID, metadata.ToolName, status)
	}
}

func demonstrateDependencyResolution(registry core.ToolRegistry) {
	fmt.Println("\n🕸️  Dependency Resolution Demonstration")
	fmt.Println("=======================================")

	// Create dependency graph
	graph := tools.NewDependencyGraph()

	// Define dependencies: extractor -> [validator, enricher] -> transformer
	nodes := []*tools.DependencyNode{
		{
			ToolName:     "data_extractor",
			Dependencies: []string{},
			Outputs:      []string{"extracted_data"},
			Priority:     1,
		},
		{
			ToolName:     "data_validator",
			Dependencies: []string{"data_extractor"},
			Inputs:       []string{"extracted_data"},
			Outputs:      []string{"validation_result"},
			Priority:     2,
		},
		{
			ToolName:     "data_enricher",
			Dependencies: []string{"data_extractor"},
			Inputs:       []string{"extracted_data"},
			Outputs:      []string{"enriched_data"},
			Priority:     3, // Higher priority than validator
		},
		{
			ToolName:     "data_transformer",
			Dependencies: []string{"data_validator", "data_enricher"},
			Inputs:       []string{"validation_result", "enriched_data"},
			Outputs:      []string{"final_output"},
			Priority:     1,
		},
	}

	for _, node := range nodes {
		if err := graph.AddNode(node); err != nil {
			log.Fatalf("Failed to add node %s: %v", node.ToolName, err)
		}
	}

	// Create dependency pipeline
	options := tools.PipelineOptions{
		Timeout:         30 * time.Second,
		FailureStrategy: tools.FailFast,
		CacheResults:    true,
	}

	depPipeline, err := tools.NewDependencyPipeline("dependency_demo", registry, graph, options)
	if err != nil {
		log.Fatalf("Failed to create dependency pipeline: %v", err)
	}

	// Show execution plan
	plan := depPipeline.GetExecutionPlan()
	fmt.Printf("📋 Execution plan created with %d phases:\n", len(plan.Phases))
	for i, phase := range plan.Phases {
		fmt.Printf("  Phase %d: %v (parallel: %t)\n", i, phase.Tools, phase.ParallelOk)
	}

	ctx := context.Background()
	input := map[string]interface{}{
		"raw_data": "Data for dependency resolution test",
	}

	fmt.Printf("\nExecuting dependency-aware pipeline...\n")

	start := time.Now()
	result, err := depPipeline.ExecuteWithDependencies(ctx, input)
	duration := time.Since(start)

	if err != nil {
		log.Fatalf("Dependency pipeline failed: %v", err)
	}

	fmt.Printf("✅ Dependency pipeline completed in %v\n", duration)
	fmt.Printf("📊 Tools executed: %d\n", len(result.Results))
	fmt.Printf("📈 Success: %t\n", result.Success)
}

func demonstrateParallelExecution(registry core.ToolRegistry) {
	fmt.Println("\n⚡ Parallel Execution Demonstration")
	fmt.Println("===================================")

	// Create parallel executor
	executor := tools.NewParallelExecutor(registry, 4)

	// Create multiple independent tasks
	tasks := []*tools.ParallelTask{
		{
			ID:         "extract_task",
			ToolName:   "data_extractor",
			Input:      map[string]interface{}{"raw_data": "Dataset 1"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "validate_task",
			ToolName:   "data_validator",
			Input:      map[string]interface{}{"confidence": 0.9},
			Priority:   2,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "enrich_task",
			ToolName:   "data_enricher",
			Input:      map[string]interface{}{"status": "ready"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "transform_task",
			ToolName:   "data_transformer",
			Input:      map[string]interface{}{"status": "processed"},
			Priority:   3,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
	}

	fmt.Printf("Executing %d tasks in parallel...\n", len(tasks))

	ctx := context.Background()
	start := time.Now()
	results, err := executor.ExecuteParallel(ctx, tasks, &tools.PriorityScheduler{})
	duration := time.Since(start)

	if err != nil {
		log.Fatalf("Parallel execution failed: %v", err)
	}

	fmt.Printf("✅ Parallel execution completed in %v\n", duration)
	fmt.Printf("📊 Tasks completed: %d\n", len(results))

	// Show results
	for _, result := range results {
		status := "✅ success"
		if result.Error != nil {
			status = "❌ failed"
		}
		fmt.Printf("  %s: %s (wait: %v, exec: %v, worker: %d)\n",
			result.TaskID, status, result.WaitTime, result.Duration, result.WorkerID)
	}

	// Show metrics
	metrics := executor.GetMetrics()
	fmt.Printf("\n📈 Executor Metrics:\n")
	fmt.Printf("  Total executions: %d\n", metrics.TotalExecutions)
	fmt.Printf("  Average wait time: %v\n", metrics.AverageWaitTime)
	fmt.Printf("  Average exec time: %v\n", metrics.AverageExecTime)
	fmt.Printf("  Worker utilization: %.2f%%\n", metrics.WorkerUtilization*100)
}

func demonstrateBatchExecution(registry core.ToolRegistry) {
	fmt.Println("\n📦 Batch Execution Demonstration")
	fmt.Println("=================================")

	executor := tools.NewParallelExecutor(registry, 3)
	batchExecutor := tools.NewBatchExecutor(executor, tools.NewFairShareScheduler())

	// Create batch of tool calls
	calls := []tools.ToolCall{
		{
			ToolName: "data_extractor",
			Input:    map[string]interface{}{"raw_data": "Batch item 1"},
			Priority: 1,
			Timeout:  5 * time.Second,
		},
		{
			ToolName: "data_extractor",
			Input:    map[string]interface{}{"raw_data": "Batch item 2"},
			Priority: 1,
			Timeout:  5 * time.Second,
		},
		{
			ToolName: "data_validator",
			Input:    map[string]interface{}{"confidence": 0.85},
			Priority: 2,
			Timeout:  5 * time.Second,
		},
		{
			ToolName: "data_enricher",
			Input:    map[string]interface{}{"status": "ready"},
			Priority: 1,
			Timeout:  5 * time.Second,
		},
		{
			ToolName: "data_transformer",
			Input:    map[string]interface{}{"status": "enriched"},
			Priority: 3,
			Timeout:  5 * time.Second,
		},
	}

	fmt.Printf("Executing batch of %d tool calls...\n", len(calls))

	ctx := context.Background()
	start := time.Now()
	results, err := batchExecutor.ExecuteBatch(ctx, calls)
	duration := time.Since(start)

	if err != nil {
		log.Fatalf("Batch execution failed: %v", err)
	}

	fmt.Printf("✅ Batch execution completed in %v\n", duration)

	// Count successes and failures
	successCount := 0
	for _, result := range results {
		if result.Error == nil {
			successCount++
		}
	}

	fmt.Printf("📊 Success rate: %d/%d (%.1f%%)\n",
		successCount, len(results), float64(successCount)/float64(len(results))*100)
}

func main() {
	fmt.Println("🚀 Tool Chaining and Composition Examples")
	fmt.Println("==========================================")

	// Setup registry with example tools
	registry := setupRegistry()
	fmt.Printf("✅ Registered %d tools\n", len(registry.List()))

	// Run all demonstrations
	demonstrateBasicPipeline(registry)
	demonstrateDataTransformations(registry)
	demonstrateConditionalExecution(registry)
	demonstrateDependencyResolution(registry)
	demonstrateParallelExecution(registry)
	demonstrateBatchExecution(registry)

	fmt.Println("\n🎉 All demonstrations completed successfully!")
	fmt.Println("\nKey Features Demonstrated:")
	fmt.Println("  ✅ Basic sequential pipelines with fluent API")
	fmt.Println("  ✅ Data transformations between pipeline steps")
	fmt.Println("  ✅ Conditional step execution based on previous results")
	fmt.Println("  ✅ Dependency-aware execution with automatic parallelization")
	fmt.Println("  ✅ High-performance parallel task execution")
	fmt.Println("  ✅ Batch processing with intelligent scheduling")
	fmt.Println("  ✅ Comprehensive caching and performance tracking")
	fmt.Println("  ✅ Advanced error handling and retry mechanisms")
}
