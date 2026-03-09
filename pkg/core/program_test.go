package core

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockModule is a mock implementation of Module interface for testing.
type MockModule struct {
	mock.Mock
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	args := m.Called(ctx, inputs)
	return args.Get(0).(map[string]any), args.Error(1)
}

func (m *MockModule) GetSignature() Signature {
	args := m.Called()
	return args.Get(0).(Signature)
}

func (m *MockModule) SetLLM(llm LLM) {
	m.Called(llm)
}

func (m *MockModule) Clone() Module {
	args := m.Called()
	return args.Get(0).(Module)
}

func (m *MockModule) SetSignature(signature Signature) {
	m.Called(signature)
}

func (m *MockModule) GetDisplayName() string {
	args := m.Called()
	if len(args) > 0 {
		return args.String(0)
	}
	return "MockModule"
}

func (m *MockModule) GetModuleType() string {
	args := m.Called()
	if len(args) > 0 {
		return args.String(0)
	}
	return "mock"
}

func TestProgram(t *testing.T) {
	t.Run("NewProgram", func(t *testing.T) {
		mockModule := new(MockModule)
		modules := map[string]Module{"test": mockModule}
		forward := func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return nil, nil
		}

		program := NewProgram(modules, forward)
		assert.Equal(t, modules, program.Modules)
		assert.NotNil(t, program.Forward)
	})

	t.Run("Execute with valid inputs", func(t *testing.T) {
		mockModule := new(MockModule)
		expectedOutputs := map[string]interface{}{"result": "success"}
		forward := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return expectedOutputs, nil
		}

		program := NewProgram(map[string]Module{"test": mockModule}, forward)
		ctx := WithExecutionState(context.Background())

		outputs, err := program.Execute(ctx, map[string]interface{}{"input": "test"})
		assert.NoError(t, err)
		assert.Equal(t, expectedOutputs, outputs)

		// Verify spans were created
		spans := CollectSpans(ctx)
		require.NotEmpty(t, spans)
		assert.Equal(t, "Program", spans[0].Operation)
	})

	t.Run("Execute with nil forward function", func(t *testing.T) {
		program := NewProgram(nil, nil)
		ctx := WithExecutionState(context.Background())

		_, err := program.Execute(ctx, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "forward function is not defined")
	})

	t.Run("Execute with forward error", func(t *testing.T) {
		expectedErr := errors.New("forward error")
		forward := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return nil, expectedErr
		}

		program := NewProgram(nil, forward)
		ctx := WithExecutionState(context.Background())

		_, err := program.Execute(ctx, nil)
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)

		// Verify error was recorded in span
		spans := CollectSpans(ctx)
		require.NotEmpty(t, spans)
		assert.Equal(t, expectedErr, spans[0].Error)
	})

	t.Run("GetSignature", func(t *testing.T) {
		mockModule := new(MockModule)
		expectedSig := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)
		mockModule.On("GetSignature").Return(expectedSig)

		program := NewProgram(map[string]Module{"test": mockModule}, nil)
		sig := program.GetSignature()
		assert.Equal(t, expectedSig, sig)
		mockModule.AssertExpectations(t)
	})

	t.Run("Clone", func(t *testing.T) {
		expectedSig := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)

		// Create and configure the original mock module
		mockOriginal := new(MockModule)
		// GetSignature gets called once directly on the original program.
		mockOriginal.On("GetSignature").Return(expectedSig).Once()

		// Create and configure the cloned mock module
		mockCloned := new(MockModule)
		// GetSignature gets called on clone after it's created
		mockCloned.On("GetSignature").Return(expectedSig).Times(1)

		// Set up the Clone expectation
		mockOriginal.On("Clone").Return(mockCloned)

		mockOriginal.On("Process", mock.Anything, mock.Anything).Return(map[string]interface{}{"test": "original"}, nil).Once()
		mockCloned.On("Process", mock.Anything, mock.Anything).Return(map[string]interface{}{"test": "clone"}, nil).Once()

		// Create the original program using a clone-safe forward factory.
		original := NewProgramWithForwardFactory(map[string]Module{"test": mockOriginal}, func(modules map[string]Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return modules["test"].Process(ctx, inputs)
			}
		})

		// Test GetSignature (first call)
		sig := original.GetSignature()
		assert.Equal(t, expectedSig, sig)

		// Clone the program
		clone := original.Clone()

		// Get signature from clone (triggers call on cloned module)
		cloneSig := clone.GetSignature()
		assert.Equal(t, expectedSig, cloneSig)

		// Verify the clone's structure
		assert.Len(t, clone.Modules, 1)
		assert.Contains(t, clone.Modules, "test")
		assert.NotSame(t, original.Modules["test"], clone.Modules["test"])
		assert.Same(t, mockCloned, clone.Modules["test"])

		// Test the forward function behavior
		ctx := context.Background()
		originalResult, originalErr := original.Forward(ctx, map[string]interface{}{"input": "test"})
		cloneResult, cloneErr := clone.Forward(ctx, map[string]interface{}{"input": "test"})

		// Verify the clone's forward function rebinds to the cloned module.
		assert.Equal(t, map[string]interface{}{"test": "original"}, originalResult)
		assert.Equal(t, map[string]interface{}{"test": "clone"}, cloneResult)
		assert.Equal(t, originalErr, cloneErr)

		// Verify all mock expectations were met
		mockOriginal.AssertExpectations(t)
		mockCloned.AssertExpectations(t)
	})

	t.Run("RebindModules preserves plain forward when no factory exists", func(t *testing.T) {
		mockModule := new(MockModule)
		forward := func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		}

		program := NewProgram(map[string]Module{"test": mockModule}, forward)
		rebound := program.RebindModules(map[string]Module{"test": mockModule})

		assert.NotNil(t, rebound.Forward)
		assert.Equal(t, reflect.ValueOf(program.Forward).Pointer(), reflect.ValueOf(rebound.Forward).Pointer())
	})

	t.Run("Equal with identical programs", func(t *testing.T) {
		mockModule1 := new(MockModule)
		mockModule2 := new(MockModule)
		sig := NewSignature(nil, nil)

		mockModule1.On("GetSignature").Return(sig)
		mockModule2.On("GetSignature").Return(sig)

		prog1 := NewProgram(map[string]Module{"test": mockModule1}, nil)
		prog2 := NewProgram(map[string]Module{"test": mockModule2}, nil)

		assert.True(t, prog1.Equal(prog2))
		mockModule1.AssertExpectations(t)
		mockModule2.AssertExpectations(t)
	})

	t.Run("Equal with different programs", func(t *testing.T) {
		mockModule1 := new(MockModule)
		mockModule2 := new(MockModule)
		sig1 := NewSignature([]InputField{{Field: Field{Name: "input1"}}}, nil)
		sig2 := NewSignature([]InputField{{Field: Field{Name: "input2"}}}, nil)

		mockModule1.On("GetSignature").Return(sig1)
		mockModule2.On("GetSignature").Return(sig2)

		prog1 := NewProgram(map[string]Module{"test": mockModule1}, nil)
		prog2 := NewProgram(map[string]Module{"test": mockModule2}, nil)

		assert.False(t, prog1.Equal(prog2))
		mockModule1.AssertExpectations(t)
		mockModule2.AssertExpectations(t)
	})

	t.Run("GetModules returns sorted modules", func(t *testing.T) {
		mockModule1 := new(MockModule)
		mockModule2 := new(MockModule)
		mockModule3 := new(MockModule)

		program := NewProgram(map[string]Module{
			"c": mockModule3,
			"a": mockModule1,
			"b": mockModule2,
		}, nil)

		modules := program.GetModules()
		assert.Len(t, modules, 3)
		// Verify modules are returned in alphabetical order by key
		assert.Same(t, mockModule1, modules[0])
		assert.Same(t, mockModule2, modules[1])
		assert.Same(t, mockModule3, modules[2])
	})

	t.Run("AddModule and SetForward", func(t *testing.T) {
		program := NewProgram(make(map[string]Module), nil)
		mockModule := new(MockModule)

		program.AddModule("test", mockModule)
		assert.Contains(t, program.Modules, "test")
		assert.Same(t, mockModule, program.Modules["test"])

		forward := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return nil, nil
		}
		program.SetForward(forward)
		assert.NotNil(t, program.Forward)
	})

	t.Run("SetForwardFactory", func(t *testing.T) {
		mockModule := new(MockModule)
		mockModule.On("Process", mock.Anything, mock.Anything).Return(map[string]interface{}{"result": "ok"}, nil).Once()

		program := NewProgram(map[string]Module{"test": mockModule}, nil)
		program.SetForwardFactory(func(modules map[string]Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return modules["test"].Process(ctx, inputs)
			}
		})

		output, err := program.Execute(context.Background(), map[string]interface{}{"input": "x"})
		require.NoError(t, err)
		assert.Equal(t, map[string]interface{}{"result": "ok"}, output)
		mockModule.AssertExpectations(t)
	})
}
