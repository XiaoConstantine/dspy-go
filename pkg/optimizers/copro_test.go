package optimizers

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockModule is a mock implementation of core.Module.
type MockModule struct {
	mock.Mock
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	args := m.Called(ctx, inputs)
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

// MockOptimizer is a mock implementation of core.Optimizer.
type MockOptimizer struct {
	mock.Mock
}

func (m *MockOptimizer) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	args := m.Called(ctx, program, dataset, metric)
	return args.Get(0).(core.Program), args.Error(1)
}

// MockDataset is a mock implementation of core.Dataset.
type MockDataset struct {
	mock.Mock
}

func (m *MockDataset) Next() (core.Example, bool) {
	args := m.Called()
	return args.Get(0).(core.Example), args.Bool(1)
}

func (m *MockDataset) Reset() {
	m.Called()
}

func TestCoproCompile(t *testing.T) {
	// Create mock objects
	mockModule := new(MockModule)
	mockSubOptimizer := new(MockOptimizer)
	mockDataset := new(MockDataset)

	// Create a test program
	testProgram := core.Program{
		Modules: map[string]core.Module{
			"test": mockModule,
		},
	}

	// Create a Copro instance
	copro := NewCopro(
		func(example, prediction map[string]interface{}, trace *core.Trace) bool { return true },
		5,
		mockSubOptimizer,
	)

	// Set up expectations
	mockModule.On("Clone").Return(mockModule)
	// We no longer expect Compile to be called on non-Predict modules
	// mockSubOptimizer.On("Compile", mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(testProgram, nil)

	// Create a context with trace manager
	ctx := core.WithTraceManager(context.Background())

	// Call Compile
	compiledProgram, err := copro.Compile(ctx, testProgram, mockDataset, nil)

	// Assert expectations
	assert.NoError(t, err)
	assert.NotNil(t, compiledProgram)
	assert.Equal(t, 1, len(compiledProgram.Modules))
	assert.Contains(t, compiledProgram.Modules, "test")
	assert.Equal(t, mockModule, compiledProgram.Modules["test"]) // The module should be unchanged

	mockModule.AssertExpectations(t)
	mockSubOptimizer.AssertExpectations(t)
}

func TestCoproCompileWithPredict(t *testing.T) {
	// Create mock objects
	mockPredict := modules.NewPredict(core.Signature{})
	mockSubOptimizer := new(MockOptimizer)
	mockDataset := new(MockDataset)

	// Create a test program
	testProgram := core.Program{
		Modules: map[string]core.Module{
			"predict": mockPredict,
		},
	}

	// Create a Copro instance
	copro := NewCopro(
		func(example, prediction map[string]interface{}, trace *core.Trace) bool { return true },
		5,
		mockSubOptimizer,
	)

	// Set up expectations
	mockSubOptimizer.On("Compile", mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(testProgram, nil)

	// Create a context with trace manager
	ctx := core.WithTraceManager(context.Background())

	// Call Compile
	compiledProgram, err := copro.Compile(ctx, testProgram, mockDataset, nil)

	// Assert expectations
	assert.NoError(t, err)
	assert.NotNil(t, compiledProgram)
	assert.Equal(t, 1, len(compiledProgram.Modules))
	assert.Contains(t, compiledProgram.Modules, "predict")

	mockSubOptimizer.AssertExpectations(t)
}

func TestCoproCompileError(t *testing.T) {
	// Create mock objects
	mockSubOptimizer := new(MockOptimizer)
	mockDataset := new(MockDataset)

	// Create a test program
	testProgram := core.Program{
		Modules: map[string]core.Module{
			"test": &modules.Predict{}, // Use a real Predict module instead of mockModule
		},
	}

	// Create a Copro instance
	copro := NewCopro(
		func(example, prediction map[string]interface{}, trace *core.Trace) bool { return true },
		5,
		mockSubOptimizer,
	)

	// Set up expectations
	mockSubOptimizer.On("Compile", mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(core.Program{}, assert.AnError)

	// Create a context with trace manager
	ctx := core.WithTraceManager(context.Background())

	// Call Compile
	_, err := copro.Compile(ctx, testProgram, mockDataset, nil)

	// Assert expectations
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error compiling module test")

	mockSubOptimizer.AssertExpectations(t)
}
