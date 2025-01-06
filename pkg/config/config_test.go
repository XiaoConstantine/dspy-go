package config

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockLLM is a mock implementation of core.LLM for testing
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

// TestConfigureDefaultLLM tests the configuration of the default LLM
func TestConfigureDefaultLLM(t *testing.T) {
	// Reset GlobalConfig before each test
	GlobalConfig = &Config{ConcurrencyLevel: 1}

	t.Run("Successful configuration", func(t *testing.T) {
		// Test with Anthropic model
		err := ConfigureDefaultLLM("test-api-key", core.ModelAnthropicSonnet)
		assert.NoError(t, err)
		assert.NotNil(t, GlobalConfig.DefaultLLM)
	})

	t.Run("Configuration with invalid model", func(t *testing.T) {
		err := ConfigureDefaultLLM("test-api-key", "invalid-model")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported model ID")
	})
}

// TestConfigureTeacherLLM tests the configuration of the teacher LLM
func TestConfigureTeacherLLM(t *testing.T) {
	// Reset GlobalConfig before each test
	GlobalConfig = &Config{ConcurrencyLevel: 1}

	t.Run("Successful configuration", func(t *testing.T) {
		err := ConfigureTeacherLLM("test-api-key", core.ModelAnthropicSonnet)
		assert.NoError(t, err)
		assert.NotNil(t, GlobalConfig.TeacherLLM)
	})

	t.Run("Configuration with invalid model", func(t *testing.T) {
		err := ConfigureTeacherLLM("test-api-key", "invalid-model")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported model ID")
	})
}

// TestGetDefaultLLM tests retrieving the default LLM
func TestGetDefaultLLM(t *testing.T) {
	// Reset GlobalConfig before each test
	GlobalConfig = &Config{ConcurrencyLevel: 1}

	t.Run("Get configured LLM", func(t *testing.T) {
		mockLLM := new(MockLLM)
		GlobalConfig.DefaultLLM = mockLLM

		llm := GetDefaultLLM()
		assert.Equal(t, mockLLM, llm)
	})

	t.Run("Get nil when not configured", func(t *testing.T) {
		GlobalConfig.DefaultLLM = nil

		llm := GetDefaultLLM()
		assert.Nil(t, llm)
	})
}

// TestGetTeacherLLM tests retrieving the teacher LLM
func TestGetTeacherLLM(t *testing.T) {
	// Reset GlobalConfig before each test
	GlobalConfig = &Config{ConcurrencyLevel: 1}

	t.Run("Get configured teacher LLM", func(t *testing.T) {
		mockLLM := new(MockLLM)
		GlobalConfig.TeacherLLM = mockLLM

		llm := GetTeacherLLM()
		assert.Equal(t, mockLLM, llm)
	})

	t.Run("Get nil when not configured", func(t *testing.T) {
		GlobalConfig.TeacherLLM = nil

		llm := GetTeacherLLM()
		assert.Nil(t, llm)
	})
}

// TestSetConcurrencyOptions tests setting concurrency options
func TestSetConcurrencyOptions(t *testing.T) {
	// Reset GlobalConfig before each test
	GlobalConfig = &Config{ConcurrencyLevel: 1}

	tests := []struct {
		name          string
		level         int
		expectedLevel int
	}{
		{
			name:          "Set positive concurrency level",
			level:         5,
			expectedLevel: 5,
		},
		{
			name:          "Set zero concurrency level",
			level:         0,
			expectedLevel: 1, // Should maintain previous level
		},
		{
			name:          "Set negative concurrency level",
			level:         -1,
			expectedLevel: 1, // Should maintain previous level
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetConcurrencyOptions(tt.level)
			assert.Equal(t, tt.expectedLevel, GlobalConfig.ConcurrencyLevel)
		})
	}
}
