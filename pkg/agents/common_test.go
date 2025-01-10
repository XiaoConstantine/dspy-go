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

func TestDependencyPlanCreator(t *testing.T) {
	tests := []struct {
		name           string
		tasks          []Task
		maxTasksPhase  int
		validatePhases func(t *testing.T, phases [][]Task)
		expectError    bool
		errorMessage   string
	}{
		{
			// Basic test case: Simple linear dependency
			name: "Linear dependency chain",
			tasks: []Task{
				{
					ID:            "task1",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task2",
					Dependencies:  []string{"task1"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
			},
			maxTasksPhase: 2,
			validatePhases: func(t *testing.T, phases [][]Task) {
				require.Equal(t, 2, len(phases), "Should have two phases")

				// First phase should have task1
				require.Equal(t, 1, len(phases[0]), "First phase should have one task")
				require.Equal(t, "task1", phases[0][0].ID, "First task should be task1")

				// Second phase should have task2
				require.Equal(t, 1, len(phases[1]), "Second phase should have one task")
				require.Equal(t, "task2", phases[1][0].ID, "Second task should be task2")
			},
		},
		{
			// Test case: Multiple parallel tasks with max phase limit
			name: "Respect max tasks per phase",
			tasks: []Task{
				{
					ID:            "task1",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task2",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task3",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
			},
			maxTasksPhase: 2,
			validatePhases: func(t *testing.T, phases [][]Task) {
				// Verify phase structure
				require.Equal(t, 2, len(phases), "Should have exactly 2 phases")
				require.Equal(t, 2, len(phases[0]), "First phase should have exactly 2 tasks")
				require.Equal(t, 1, len(phases[1]), "Second phase should have exactly 1 task")

				// Track all tasks to ensure completeness
				allTasks := make(map[string]bool)
				for _, phase := range phases {
					for _, task := range phase {
						// Verify no duplicates
						require.False(t, allTasks[task.ID],
							"Task %s appears more than once", task.ID)
						allTasks[task.ID] = true

						// Verify task properties
						require.Equal(t, []string{}, task.Dependencies,
							"Task %s should have empty dependencies", task.ID)
						require.Equal(t, 1, task.Priority,
							"Task %s should have priority 1", task.ID)
					}
				}

				// Verify all tasks present
				require.Equal(t, map[string]bool{
					"task1": true,
					"task2": true,
					"task3": true,
				}, allTasks, "All tasks should be present")
			},
		},
		{
			// Test case: Priority-based ordering
			name: "Priority-based ordering",
			tasks: []Task{
				{
					ID:            "task1",
					Dependencies:  []string{},
					Priority:      3,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task2",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task3",
					Dependencies:  []string{},
					Priority:      2,
					Type:          "test",
					ProcessorType: "test",
				},
			},
			maxTasksPhase: 3,
			validatePhases: func(t *testing.T, phases [][]Task) {
				require.Equal(t, 1, len(phases), "Should have one phase")
				phase := phases[0]
				require.Equal(t, 3, len(phase), "Phase should have all tasks")

				// Verify priority ordering
				require.Equal(t, "task2", phase[0].ID, "First task should be lowest priority")
				require.Equal(t, "task3", phase[1].ID, "Second task should be medium priority")
				require.Equal(t, "task1", phase[2].ID, "Third task should be highest priority")
			},
		},
		{
			// Test case: Cyclic dependency detection
			name: "Cyclic dependency",
			tasks: []Task{
				{
					ID:            "task1",
					Dependencies:  []string{"task2"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task2",
					Dependencies:  []string{"task1"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
			},
			maxTasksPhase: 2,
			expectError:   true,
			errorMessage:  "cycle detected",
		},
		{
			// Test case: Empty task list
			name:          "Empty task list",
			tasks:         []Task{},
			maxTasksPhase: 2,
			validatePhases: func(t *testing.T, phases [][]Task) {
				require.Equal(t, 0, len(phases), "Should have no phases for empty task list")
			},
		},
		{
			// Test case: Complex dependency chain
			name: "Complex dependency chain",
			tasks: []Task{
				{
					ID:            "task1",
					Dependencies:  []string{},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task2",
					Dependencies:  []string{"task1"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task3",
					Dependencies:  []string{"task2"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
				{
					ID:            "task4",
					Dependencies:  []string{"task1"},
					Priority:      1,
					Type:          "test",
					ProcessorType: "test",
				},
			},
			maxTasksPhase: 2,
			validatePhases: func(t *testing.T, phases [][]Task) {
				require.Equal(t, 3, len(phases), "Should have three phases")

				// Validate phase structure
				require.Equal(t, 1, len(phases[0]), "First phase should have one task")
				require.Equal(t, "task1", phases[0][0].ID, "First phase should have task1")

				// Second phase can have task2 and task4 in any order
				require.Equal(t, 2, len(phases[1]), "Second phase should have two tasks")
				secondPhaseIDs := map[string]bool{
					phases[1][0].ID: true,
					phases[1][1].ID: true,
				}
				require.True(t, secondPhaseIDs["task2"], "Second phase should contain task2")
				require.True(t, secondPhaseIDs["task4"], "Second phase should contain task4")

				// Third phase should have task3
				require.Equal(t, 1, len(phases[2]), "Third phase should have one task")
				require.Equal(t, "task3", phases[2][0].ID, "Third phase should have task3")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			planner := NewDependencyPlanCreator(tt.maxTasksPhase)
			plan, err := planner.CreatePlan(tt.tasks)

			if tt.expectError {
				require.Error(t, err, "Expected an error but got none")
				require.Contains(t, err.Error(), tt.errorMessage,
					"Error message should contain expected text")
				return
			}

			require.NoError(t, err, "Should not return an error")
			if tt.validatePhases != nil {
				tt.validatePhases(t, plan)
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
