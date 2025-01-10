package agents

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

// TestXMLTaskParser covers the XML parsing functionality.
func TestXMLTaskParser(t *testing.T) {
	// Initialize test cases to cover different aspects of XML parsing
	tests := []struct {
		name           string
		analyzerOutput map[string]interface{}
		expected       []Task
		expectError    bool
		errorMessage   string
	}{
		{
			name: "Valid XML with single task",
			analyzerOutput: map[string]interface{}{
				"tasks": `<tasks>
                    <task id="task1" type="test" processor="proc1" priority="1">
                        <description>Test task</description>
                        <dependencies>
                            <dep>dep1</dep>
                        </dependencies>
                        <metadata>
                            <item key="key1">value1</item>
                        </metadata>
                    </task>
                </tasks>`,
			},
			expected: []Task{
				{
					ID:            "task1",
					Type:          "test",
					ProcessorType: "proc1",
					Priority:      1,
					Dependencies:  []string{"dep1"},
					Metadata:      map[string]interface{}{"key1": "value1"},
				},
			},
			expectError: false,
		},
		{
			name: "Invalid tasks format",
			analyzerOutput: map[string]interface{}{
				"tasks": 123, // Wrong type
			},
			expectError:  true,
			errorMessage: "invalid tasks format in analyzer output",
		},
		{
			name: "Invalid XML format",
			analyzerOutput: map[string]interface{}{
				"tasks": "not xml",
			},
			expectError:  true,
			errorMessage: "no valid XML tasks found in output",
		},
		{
			name: "Missing required fields",
			analyzerOutput: map[string]interface{}{
				"tasks": `<tasks><task></task></tasks>`,
			},
			expectError:  true,
			errorMessage: "missing task ID",
		},
	}

	parser := &XMLTaskParser{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tasks, err := parser.Parse(tt.analyzerOutput)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMessage)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.expected, tasks)
		})
	}
}

// TestDependencyPlanCreator covers the plan creation functionality.
func TestDependencyPlanCreator(t *testing.T) {
	tests := []struct {
		name          string
		tasks         []Task
		maxTasksPhase int
		expected      [][]Task
		expectError   bool
		errorMessage  string
	}{
		{
			name: "Simple linear dependency",
			tasks: []Task{
				{ID: "task1", Dependencies: []string{}},
				{ID: "task2", Dependencies: []string{"task1"}},
			},
			maxTasksPhase: 2,
			expected: [][]Task{
				{{ID: "task1", Dependencies: []string{}}},
				{{ID: "task2", Dependencies: []string{"task1"}}},
			},
		},
		{
			name: "Parallel tasks",
			tasks: []Task{
				{ID: "task1", Dependencies: []string{}},
				{ID: "task2", Dependencies: []string{}},
			},
			maxTasksPhase: 2,
			expected: [][]Task{
				{
					{ID: "task1", Dependencies: []string{}},
					{ID: "task2", Dependencies: []string{}},
				},
			},
		},
		{
			name: "Cyclic dependency",
			tasks: []Task{
				{ID: "task1", Dependencies: []string{"task2"}},
				{ID: "task2", Dependencies: []string{"task1"}},
			},
			expectError:  true,
			errorMessage: "cycle detected",
		},
		{
			name: "Respect max tasks per phase",
			tasks: []Task{
				{ID: "task1", Dependencies: []string{}, Priority: 1},
				{ID: "task2", Dependencies: []string{}, Priority: 1},
				{ID: "task3", Dependencies: []string{}, Priority: 1},
			},
			maxTasksPhase: 2,
			expected: [][]Task{
				{
					{ID: "task1", Dependencies: []string{}, Priority: 1},
					{ID: "task2", Dependencies: []string{}, Priority: 1},
				},
				{
					{ID: "task3", Dependencies: []string{}, Priority: 1},
				},
			},
		},
		{
			name: "Priority-based ordering",
			tasks: []Task{
				{ID: "task1", Dependencies: []string{}, Priority: 2},
				{ID: "task2", Dependencies: []string{}, Priority: 1},
				{ID: "task3", Dependencies: []string{}, Priority: 3},
			},
			maxTasksPhase: 3,
			expected: [][]Task{
				{
					{ID: "task2", Dependencies: []string{}, Priority: 1},
					{ID: "task1", Dependencies: []string{}, Priority: 2},
					{ID: "task3", Dependencies: []string{}, Priority: 3},
				},
			},
		},
		{
			name:          "Empty task list",
			tasks:         []Task{},
			maxTasksPhase: 2,
			expected:      [][]Task{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			planner := NewDependencyPlanCreator(tt.maxTasksPhase)
			plan, err := planner.CreatePlan(tt.tasks)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMessage)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, len(tt.expected), len(plan))
			for i, expectedPhase := range tt.expected {
				actualPhase := plan[i]
				assert.Equal(t, len(expectedPhase), len(actualPhase),
					"Phase %d should have same number of tasks", i)

				// Create maps of task IDs for comparison
				expectedIDs := make(map[string]bool)
				actualIDs := make(map[string]bool)

				for _, task := range expectedPhase {
					expectedIDs[task.ID] = true
				}
				for _, task := range actualPhase {
					actualIDs[task.ID] = true
				}

				// Compare the sets of task IDs
				assert.Equal(t, expectedIDs, actualIDs,
					"Phase %d should contain the same set of tasks", i)

				// Verify each task's properties
				for _, actualTask := range actualPhase {
					// Find corresponding expected task
					var expectedTask Task
					for _, et := range expectedPhase {
						if et.ID == actualTask.ID {
							expectedTask = et
							break
						}
					}

					assert.Equal(t, expectedTask.Dependencies, actualTask.Dependencies,
						"Task %s should have correct dependencies", actualTask.ID)
					assert.Equal(t, expectedTask.Priority, actualTask.Priority,
						"Task %s should have correct priority", actualTask.ID)
				}
			}
		})
	}
}

// TestHelperFunctions covers the utility functions.
func TestHelperFunctions(t *testing.T) {
	t.Run("buildDependencyGraph", func(t *testing.T) {
		tasks := []Task{
			{ID: "task1", Dependencies: []string{"task2"}},
			{ID: "task2", Dependencies: []string{"task3"}},
			{ID: "task3", Dependencies: []string{}},
		}

		graph := buildDependencyGraph(tasks)

		assert.Equal(t, []string{"task2"}, graph["task1"])
		assert.Equal(t, []string{"task3"}, graph["task2"])
		assert.Equal(t, []string{}, graph["task3"])
	})

	t.Run("detectCycles", func(t *testing.T) {
		// Test case with no cycles
		noCycles := map[string][]string{
			"task1": {"task2"},
			"task2": {"task3"},
			"task3": {},
		}
		assert.NoError(t, detectCycles(noCycles))

		// Test case with cycles
		withCycles := map[string][]string{
			"task1": {"task2"},
			"task2": {"task1"},
		}
		err := detectCycles(withCycles)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cycle detected")
	})

	t.Run("canExecute", func(t *testing.T) {
		completed := map[string]bool{
			"task1": true,
			"task2": true,
		}

		// Test tasks with different dependency states
		tests := []struct {
			task     Task
			expected bool
		}{
			{
				Task{ID: "task3", Dependencies: []string{}},
				true,
			},
			{
				Task{ID: "task4", Dependencies: []string{"task1", "task2"}},
				true,
			},
			{
				Task{ID: "task5", Dependencies: []string{"task1", "missing"}},
				false,
			},
		}

		for _, tt := range tests {
			assert.Equal(t, tt.expected, canExecute(tt.task, completed))
		}
	})
}

// TestNewDependencyPlanCreator tests the constructor.
func TestNewDependencyPlanCreator(t *testing.T) {
	t.Run("Default max tasks", func(t *testing.T) {
		planner := NewDependencyPlanCreator(0)
		assert.Equal(t, 10, planner.MaxTasksPerPhase)
	})

	t.Run("Custom max tasks", func(t *testing.T) {
		planner := NewDependencyPlanCreator(5)
		assert.Equal(t, 5, planner.MaxTasksPerPhase)
	})
}
