package optimizers

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/mock"
)

// setupTestMockLLM configures a mock LLM for testing with proper cleanup.
// This should be called at the beginning of each test that needs a mock LLM.
func setupTestMockLLM(t *testing.T) *testutil.MockLLM {
	t.Helper()

	// Save original values
	origDefaultLLM := core.GlobalConfig.DefaultLLM
	origTeacherLLM := core.GlobalConfig.TeacherLLM
	origConcurrency := core.GlobalConfig.ConcurrencyLevel

	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: `answer:
	Paris`}, nil).Maybe()
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(map[string]interface{}{"answer": "Paris"}, nil).Maybe()

	core.GlobalConfig.DefaultLLM = mockLLM
	core.GlobalConfig.TeacherLLM = mockLLM
	core.GlobalConfig.ConcurrencyLevel = 1

	// Restore original values when test completes
	t.Cleanup(func() {
		core.GlobalConfig.DefaultLLM = origDefaultLLM
		core.GlobalConfig.TeacherLLM = origTeacherLLM
		core.GlobalConfig.ConcurrencyLevel = origConcurrency
	})

	return mockLLM
}
