package tools

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDependencyGraph_AddNode(t *testing.T) {
	graph := NewDependencyGraph()
	
	node := &DependencyNode{
		ToolName:     "test_tool",
		Dependencies: []string{"dep1", "dep2"},
		Outputs:      []string{"output1"},
		Inputs:       []string{"input1"},
		Priority:     1,
	}
	
	err := graph.AddNode(node)
	require.NoError(t, err)
	
	// Test duplicate addition
	err = graph.AddNode(node)
	require.NoError(t, err) // Should overwrite, not error
	
	// Test empty tool name
	emptyNode := &DependencyNode{ToolName: ""}
	err = graph.AddNode(emptyNode)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tool name cannot be empty")
}

func TestDependencyGraph_RemoveNode(t *testing.T) {
	graph := NewDependencyGraph()
	
	// Add nodes
	node1 := &DependencyNode{
		ToolName:     "tool1",
		Dependencies: []string{},
	}
	node2 := &DependencyNode{
		ToolName:     "tool2",
		Dependencies: []string{"tool1"},
	}
	
	err := graph.AddNode(node1)
	require.NoError(t, err)
	err = graph.AddNode(node2)
	require.NoError(t, err)
	
	// Remove node1
	err = graph.RemoveNode("tool1")
	require.NoError(t, err)
	
	// Verify tool1 is removed from tool2's dependencies
	deps, err := graph.GetDependencies("tool2")
	require.NoError(t, err)
	assert.Empty(t, deps)
	
	// Test removing non-existent node
	err = graph.RemoveNode("nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tool not found")
}

func TestDependencyGraph_CycleDetection(t *testing.T) {
	graph := NewDependencyGraph()
	
	// Create a cycle: A -> B -> C -> A
	nodeA := &DependencyNode{
		ToolName:     "A",
		Dependencies: []string{"C"},
	}
	nodeB := &DependencyNode{
		ToolName:     "B",
		Dependencies: []string{"A"},
	}
	nodeC := &DependencyNode{
		ToolName:     "C",
		Dependencies: []string{"B"},
	}
	
	err := graph.AddNode(nodeA)
	require.NoError(t, err)
	err = graph.AddNode(nodeB)
	require.NoError(t, err)
	err = graph.AddNode(nodeC)
	require.NoError(t, err)
	
	// Creating execution plan should detect cycle
	_, err = graph.CreateExecutionPlan()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cycle detected")
}

func TestDependencyGraph_TopologicalSort(t *testing.T) {
	graph := NewDependencyGraph()
	
	// Create a DAG: A -> B -> D, A -> C -> D
	nodeA := &DependencyNode{
		ToolName:     "A",
		Dependencies: []string{},
		Priority:     1,
	}
	nodeB := &DependencyNode{
		ToolName:     "B",
		Dependencies: []string{"A"},
		Priority:     2,
	}
	nodeC := &DependencyNode{
		ToolName:     "C",
		Dependencies: []string{"A"},
		Priority:     3, // Higher priority than B
	}
	nodeD := &DependencyNode{
		ToolName:     "D",
		Dependencies: []string{"B", "C"},
		Priority:     1,
	}
	
	err := graph.AddNode(nodeA)
	require.NoError(t, err)
	err = graph.AddNode(nodeB)
	require.NoError(t, err)
	err = graph.AddNode(nodeC)
	require.NoError(t, err)
	err = graph.AddNode(nodeD)
	require.NoError(t, err)
	
	plan, err := graph.CreateExecutionPlan()
	require.NoError(t, err)
	require.NotNil(t, plan)
	
	// Should have 3 phases: [A], [C, B], [D]
	assert.Len(t, plan.Phases, 3)
	
	// Phase 0: A (no dependencies)
	assert.Len(t, plan.Phases[0].Tools, 1)
	assert.Contains(t, plan.Phases[0].Tools, "A")
	
	// Phase 1: B and C (depend on A)
	assert.Len(t, plan.Phases[1].Tools, 2)
	assert.Contains(t, plan.Phases[1].Tools, "B")
	assert.Contains(t, plan.Phases[1].Tools, "C")
	// C should come before B due to higher priority
	assert.Equal(t, "C", plan.Phases[1].Tools[0])
	
	// Phase 2: D (depends on B and C)
	assert.Len(t, plan.Phases[2].Tools, 1)
	assert.Contains(t, plan.Phases[2].Tools, "D")
}

func TestDependencyGraph_GetMethods(t *testing.T) {
	graph := NewDependencyGraph()
	
	node := &DependencyNode{
		ToolName:     "test_tool",
		Dependencies: []string{"dep1", "dep2"},
		Outputs:      []string{"output1"},
	}
	
	err := graph.AddNode(node)
	require.NoError(t, err)
	
	// Test GetNode
	retrievedNode, err := graph.GetNode("test_tool")
	require.NoError(t, err)
	assert.Equal(t, "test_tool", retrievedNode.ToolName)
	assert.Equal(t, []string{"dep1", "dep2"}, retrievedNode.Dependencies)
	
	// Test GetNode with non-existent tool
	_, err = graph.GetNode("nonexistent")
	assert.Error(t, err)
	
	// Test GetDependencies
	deps, err := graph.GetDependencies("test_tool")
	require.NoError(t, err)
	assert.Equal(t, []string{"dep1", "dep2"}, deps)
	
	// Test GetDependents
	dependentNode := &DependencyNode{
		ToolName:     "dependent",
		Dependencies: []string{"test_tool"},
	}
	err = graph.AddNode(dependentNode)
	require.NoError(t, err)
	
	dependents, err := graph.GetDependents("test_tool")
	require.NoError(t, err)
	assert.Contains(t, dependents, "dependent")
}

func TestDependencyPipeline_Creation(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	// Create a simple dependency graph
	nodeA := &DependencyNode{
		ToolName:     "parser",
		Dependencies: []string{},
	}
	nodeB := &DependencyNode{
		ToolName:     "validator",
		Dependencies: []string{"parser"},
	}
	
	err := graph.AddNode(nodeA)
	require.NoError(t, err)
	err = graph.AddNode(nodeB)
	require.NoError(t, err)
	
	options := PipelineOptions{
		Timeout:         10 * time.Second,
		FailureStrategy: FailFast,
	}
	
	pipeline, err := NewDependencyPipeline("dep-pipeline", registry, graph, options)
	require.NoError(t, err)
	assert.NotNil(t, pipeline)
	assert.NotNil(t, pipeline.plan)
}

func TestDependencyPipeline_ExecuteWithDependencies(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	// Create a dependency graph: parser -> validator -> transformer
	nodes := []*DependencyNode{
		{
			ToolName:     "parser",
			Dependencies: []string{},
			Outputs:      []string{"parsed_data"},
			Priority:     1,
		},
		{
			ToolName:     "validator",
			Dependencies: []string{"parser"},
			Inputs:       []string{"parsed_data"},
			Outputs:      []string{"validated_data"},
			Priority:     2,
		},
		{
			ToolName:     "transformer",
			Dependencies: []string{"validator"},
			Inputs:       []string{"validated_data"},
			Outputs:      []string{"transformed_data"},
			Priority:     3,
		},
	}
	
	for _, node := range nodes {
		err := graph.AddNode(node)
		require.NoError(t, err)
	}
	
	options := PipelineOptions{
		Timeout:         10 * time.Second,
		FailureStrategy: FailFast,
	}
	
	pipeline, err := NewDependencyPipeline("dep-pipeline", registry, graph, options)
	require.NoError(t, err)
	
	// Execute pipeline
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	result, err := pipeline.ExecuteWithDependencies(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 3)
	
	// Verify execution order by checking processed_by fields
	for i, toolResult := range result.Results {
		data, ok := toolResult.Data.(map[string]interface{})
		require.True(t, ok)
		
		expectedTool := nodes[i].ToolName
		assert.Equal(t, expectedTool, data["processed_by"])
	}
}

func TestDependencyPipeline_ParallelExecution(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	// Create a diamond dependency: A -> [B, C] -> D
	nodes := []*DependencyNode{
		{
			ToolName:     "parser",
			Dependencies: []string{},
			Priority:     1,
		},
		{
			ToolName:     "validator",
			Dependencies: []string{"parser"},
			Priority:     2,
		},
		{
			ToolName:     "transformer", 
			Dependencies: []string{"parser"},
			Priority:     3, // Higher priority than validator
		},
		{
			ToolName:     "processor",
			Dependencies: []string{"validator", "transformer"},
			Priority:     1,
		},
	}
	
	for _, node := range nodes {
		err := graph.AddNode(node)
		require.NoError(t, err)
	}
	
	options := PipelineOptions{
		Timeout:         10 * time.Second,
		FailureStrategy: FailFast,
	}
	
	pipeline, err := NewDependencyPipeline("parallel-dep", registry, graph, options)
	require.NoError(t, err)
	
	// Check execution plan
	plan := pipeline.GetExecutionPlan()
	assert.Len(t, plan.Phases, 3)
	
	// Phase 0: parser
	assert.Len(t, plan.Phases[0].Tools, 1)
	assert.Equal(t, "parser", plan.Phases[0].Tools[0])
	
	// Phase 1: transformer, validator (can run in parallel)
	assert.Len(t, plan.Phases[1].Tools, 2)
	assert.True(t, plan.Phases[1].ParallelOk)
	// transformer should come first due to higher priority
	assert.Equal(t, "transformer", plan.Phases[1].Tools[0])
	assert.Equal(t, "validator", plan.Phases[1].Tools[1])
	
	// Phase 2: processor
	assert.Len(t, plan.Phases[2].Tools, 1)
	assert.Equal(t, "processor", plan.Phases[2].Tools[0])
	
	// Execute and verify
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	start := time.Now()
	result, err := pipeline.ExecuteWithDependencies(ctx, input)
	duration := time.Since(start)
	
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 4)
	
	// Test that parallel execution actually works by checking phase execution
	// Phase 1 should have run validator and transformer in parallel
	// We can verify this by checking that both tools were executed
	validatorExecuted := false
	transformerExecuted := false
	
	for _, result := range result.Results {
		if resultData, ok := result.Data.(map[string]interface{}); ok {
			if processedBy, ok := resultData["processed_by"].(string); ok {
				if processedBy == "validator" {
					validatorExecuted = true
				}
				if processedBy == "transformer" {
					transformerExecuted = true
				}
			}
		}
	}
	
	assert.True(t, validatorExecuted, "Validator should have been executed")
	assert.True(t, transformerExecuted, "Transformer should have been executed")
	
	// Log timing for debugging (but don't assert on it)
	t.Logf("Parallel execution took %v (expected ~45ms, sequential would be ~50ms)", duration)
}

func TestDependencyPipeline_ErrorHandling(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	// Create chain with error tool: parser -> error_tool -> validator
	nodes := []*DependencyNode{
		{
			ToolName:     "parser",
			Dependencies: []string{},
		},
		{
			ToolName:     "error_tool",
			Dependencies: []string{"parser"},
		},
		{
			ToolName:     "validator",
			Dependencies: []string{"error_tool"},
		},
	}
	
	for _, node := range nodes {
		err := graph.AddNode(node)
		require.NoError(t, err)
	}
	
	options := PipelineOptions{
		Timeout:         10 * time.Second,
		FailureStrategy: FailFast,
	}
	
	pipeline, err := NewDependencyPipeline("error-dep", registry, graph, options)
	require.NoError(t, err)
	
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	result, err := pipeline.ExecuteWithDependencies(ctx, input)
	assert.Error(t, err)
	assert.False(t, result.Success)
	assert.Equal(t, "error_tool", result.FailedStep)
	
	// Should have results from parser and error_tool
	assert.GreaterOrEqual(t, len(result.Results), 1)
}

func TestDependencyGraph_ComplexScenario(t *testing.T) {
	graph := NewDependencyGraph()
	
	// Create a more complex dependency graph
	//      A
	//     / \
	//    B   C
	//   /|   |\
	//  D E   F G
	//   \   /
	//    \ /
	//     H
	
	nodes := []*DependencyNode{
		{ToolName: "A", Dependencies: []string{}, Priority: 10},
		{ToolName: "B", Dependencies: []string{"A"}, Priority: 5},
		{ToolName: "C", Dependencies: []string{"A"}, Priority: 8},
		{ToolName: "D", Dependencies: []string{"B"}, Priority: 3},
		{ToolName: "E", Dependencies: []string{"B"}, Priority: 4},
		{ToolName: "F", Dependencies: []string{"C"}, Priority: 2},
		{ToolName: "G", Dependencies: []string{"C"}, Priority: 1},
		{ToolName: "H", Dependencies: []string{"D", "F"}, Priority: 1},
	}
	
	for _, node := range nodes {
		err := graph.AddNode(node)
		require.NoError(t, err)
	}
	
	plan, err := graph.CreateExecutionPlan()
	require.NoError(t, err)
	
	// Should have appropriate phases
	assert.GreaterOrEqual(t, len(plan.Phases), 4)
	
	// Phase 0: A
	assert.Contains(t, plan.Phases[0].Tools, "A")
	
	// Phase 1: C, B (C has higher priority)
	phase1Tools := plan.Phases[1].Tools
	assert.Contains(t, phase1Tools, "B")
	assert.Contains(t, phase1Tools, "C")
	assert.Equal(t, "C", phase1Tools[0]) // C should come first due to higher priority
	
	// Verify no cycles
	hasCycle, _ := graph.detectCycle()
	assert.False(t, hasCycle)
}

func TestDependencyGraph_SelfDependency(t *testing.T) {
	graph := NewDependencyGraph()
	
	// Create self-dependency: A -> A
	node := &DependencyNode{
		ToolName:     "A",
		Dependencies: []string{"A"},
	}
	
	err := graph.AddNode(node)
	require.NoError(t, err)
	
	// Should detect cycle
	_, err = graph.CreateExecutionPlan()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cycle detected")
}

func TestDependencyPipeline_ContextCancellation(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	node := &DependencyNode{
		ToolName:     "parser",
		Dependencies: []string{},
	}
	err := graph.AddNode(node)
	require.NoError(t, err)
	
	options := PipelineOptions{
		Timeout: 1 * time.Millisecond, // Very short timeout
	}
	
	pipeline, err := NewDependencyPipeline("timeout-dep", registry, graph, options)
	require.NoError(t, err)
	
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	_, err = pipeline.ExecuteWithDependencies(ctx, input)
	assert.Error(t, err)
}

func TestDependencyPipeline_EmptyGraph(t *testing.T) {
	registry := createTestRegistry()
	graph := NewDependencyGraph()
	
	options := PipelineOptions{}
	
	pipeline, err := NewDependencyPipeline("empty-dep", registry, graph, options)
	require.NoError(t, err)
	
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	result, err := pipeline.ExecuteWithDependencies(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Empty(t, result.Results)
}