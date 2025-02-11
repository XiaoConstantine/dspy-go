package workflows

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockMemory implements agents.Memory interface for testing.
type MockMemory struct {
	mock.Mock
}

func (m *MockMemory) Store(key string, value interface{}) error {
	args := m.Called(key, value)
	return args.Error(0)
}

func (m *MockMemory) Retrieve(key string) (interface{}, error) {
	args := m.Called(key)
	return args.Get(0), args.Error(1)
}

func (m *MockMemory) List() ([]string, error) {
	args := m.Called()
	return args.Get(0).([]string), args.Error(1)
}

func (m *MockMemory) Clear() error {
	args := m.Called()
	return args.Error(0)
}

// MockModule implements core.Module interface for testing.
type MockModule struct {
	mock.Mock
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	args := m.Called(ctx, inputs)
	// Handle nil return case properly
	if args.Get(0) == nil {
		return make(map[string]any), args.Error(1)
	}
	return args.Get(0).(map[string]any), args.Error(1)
}

func (m *MockModule) GetSignature() core.Signature {
	args := m.Called()
	return args.Get(0).(core.Signature)
}

func (m *MockModule) SetLLM(llm core.LLM) {
	m.Called(llm)
}

func (m *MockModule) Clone() core.Module {
	args := m.Called()
	return args.Get(0).(core.Module)
}

func TestBaseWorkflow(t *testing.T) {
	t.Run("AddStep success", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewBaseWorkflow(memory)

		step := &Step{
			ID:     "test_step",
			Module: new(MockModule),
		}

		err := workflow.AddStep(step)
		assert.NoError(t, err)
		assert.Len(t, workflow.GetSteps(), 1)
		assert.Equal(t, step, workflow.stepIndex["test_step"])
	})

	t.Run("AddStep duplicate ID", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewBaseWorkflow(memory)

		step := &Step{
			ID:     "test_step",
			Module: new(MockModule),
		}

		_ = workflow.AddStep(step)
		err := workflow.AddStep(step)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already exists")
	})

	t.Run("ValidateWorkflow success", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewBaseWorkflow(memory)

		step1 := &Step{
			ID:        "step1",
			Module:    new(MockModule),
			NextSteps: []string{"step2"},
		}

		step2 := &Step{
			ID:     "step2",
			Module: new(MockModule),
		}

		_ = workflow.AddStep(step1)
		_ = workflow.AddStep(step2)

		err := workflow.ValidateWorkflow()
		assert.NoError(t, err)
	})

	t.Run("ValidateWorkflow cyclic dependency", func(t *testing.T) {
		memory := new(MockMemory)
		workflow := NewBaseWorkflow(memory)

		step1 := &Step{
			ID:        "step1",
			Module:    new(MockModule),
			NextSteps: []string{"step2"},
		}

		step2 := &Step{
			ID:        "step2",
			Module:    new(MockModule),
			NextSteps: []string{"step1"},
		}

		_ = workflow.AddStep(step1)
		_ = workflow.AddStep(step2)

		err := workflow.ValidateWorkflow()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cycle detected")
	})
}
