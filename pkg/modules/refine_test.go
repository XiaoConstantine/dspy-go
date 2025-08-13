package modules

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestNewRefine(t *testing.T) {
	// Create a mock module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	mockModule := NewPredict(signature)

	// Simple reward function
	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		answer, ok := outputs["answer"].(string)
		if !ok || answer == "" {
			return 0.0
		}
		return 1.0 // Simple binary reward
	}

	config := RefineConfig{
		N:         3,
		RewardFn:  rewardFn,
		Threshold: 0.8,
	}

	refine := NewRefine(mockModule, config)

	assert.NotNil(t, refine)
	assert.Equal(t, 3, refine.config.N)
	assert.Equal(t, 0.8, refine.config.Threshold)
	assert.NotNil(t, refine.config.RewardFn)
	assert.Equal(t, signature, refine.GetSignature())
}

func TestNewRefineDefaults(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)
	mockModule := NewPredict(signature)

	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		return 0.5
	}

	// Test with zero values to trigger defaults
	config := RefineConfig{
		N:        0, // Should default to 3
		RewardFn: rewardFn,
		// Threshold: 0, // Should default to 0.7
	}

	refine := NewRefine(mockModule, config)

	assert.Equal(t, 3, refine.config.N)
	assert.Equal(t, 0.7, refine.config.Threshold)
}

func TestRefineProcess(t *testing.T) {
	ctx := context.Background()

	// Create signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create mock LLM that returns predictable responses
	mockLLM := &testutil.MockLLM{}
	// Return creative answer which should get highest reward
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		&core.LLMResponse{
			Content: "answer: creative answer",
		}, nil,
	)

	// Create predict module with mock LLM
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Reward function that prefers "creative" answers
	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		answer, ok := outputs["answer"].(string)
		if !ok {
			return 0.0
		}

		if strings.Contains(answer, "creative") {
			return 1.0 // Perfect score for creative answers
		} else if strings.Contains(answer, "moderate") {
			return 0.7 // Good score for moderate answers
		} else {
			return 0.3 // Low score for conservative answers
		}
	}

	config := RefineConfig{
		N:         3,
		RewardFn:  rewardFn,
		Threshold: 0.9, // High threshold to force multiple attempts
	}

	refine := NewRefine(predict, config)

	// Test inputs
	inputs := map[string]interface{}{
		"question": "What is the best approach?",
	}

	// Execute refinement
	outputs, err := refine.Process(ctx, inputs)

	require.NoError(t, err)
	require.NotNil(t, outputs)

	// Should get the creative answer due to higher reward
	answer, ok := outputs["answer"].(string)
	require.True(t, ok)
	assert.Contains(t, answer, "creative", "Should select the creative answer with highest reward")
}

func TestRefineEarlyTermination(t *testing.T) {
	ctx := context.Background()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)

	attemptCount := 0
	mockLLM := &testutil.MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		&core.LLMResponse{
			Content: "output: excellent result",
		}, nil,
	).Run(func(args mock.Arguments) {
		attemptCount++
	})

	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Reward function that always returns high score
	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		return 1.0 // Always perfect score
	}

	config := RefineConfig{
		N:         5, // Allow up to 5 attempts
		RewardFn:  rewardFn,
		Threshold: 0.9, // Threshold that will be met on first attempt
	}

	refine := NewRefine(predict, config)

	inputs := map[string]interface{}{
		"input": "test",
	}

	outputs, err := refine.Process(ctx, inputs)

	require.NoError(t, err)
	require.NotNil(t, outputs)

	// Should terminate early after first attempt
	assert.Equal(t, 1, attemptCount, "Should terminate early when threshold is met")
}

func TestRefineAllAttemptsFail(t *testing.T) {
	ctx := context.Background()

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)

	mockLLM := &testutil.MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		(*core.LLMResponse)(nil), fmt.Errorf("mock LLM error"),
	)

	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		return 0.5
	}

	config := RefineConfig{
		N:         3,
		RewardFn:  rewardFn,
		Threshold: 0.8,
	}

	refine := NewRefine(predict, config)

	inputs := map[string]interface{}{
		"input": "test",
	}

	outputs, err := refine.Process(ctx, inputs)

	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "all refinement attempts failed")
}

func TestGenerateTemperatureSequence(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)
	mockModule := NewPredict(signature)

	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		return 0.5
	}

	tests := []struct {
		name     string
		n        int
		expected int
	}{
		{"Single attempt", 1, 1},
		{"Two attempts", 2, 2},
		{"Three attempts", 3, 3},
		{"Five attempts", 5, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := RefineConfig{
				N:         tt.n,
				RewardFn:  rewardFn,
				Threshold: 0.7,
			}

			refine := NewRefine(mockModule, config)
			temps := refine.generateTemperatureSequence()

			assert.Equal(t, tt.expected, len(temps))

			// All temperatures should be in valid range
			for i, temp := range temps {
				assert.True(t, temp >= 0.1 && temp <= 1.0,
					"Temperature %d (%.2f) should be between 0.1 and 1.0", i, temp)
			}

			// First temperature should be conservative for multi-attempt scenarios
			if len(temps) > 1 {
				assert.Equal(t, 0.3, temps[0], "First temperature should be conservative")
			}
		})
	}
}

func TestRefineClone(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)
	mockModule := NewPredict(signature)

	rewardFn := func(inputs, outputs map[string]interface{}) float64 {
		return 0.5
	}

	config := RefineConfig{
		N:         3,
		RewardFn:  rewardFn,
		Threshold: 0.8,
	}

	original := NewRefine(mockModule, config)
	clone := original.Clone().(*Refine)

	// Should be different instances
	assert.NotSame(t, original, clone)
	assert.NotSame(t, original.module, clone.module)

	// But should have same configuration
	assert.Equal(t, original.config.N, clone.config.N)
	assert.Equal(t, original.config.Threshold, clone.config.Threshold)
	assert.Equal(t, original.GetSignature(), clone.GetSignature())
}

func TestRefineUpdateConfig(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)
	mockModule := NewPredict(signature)

	rewardFn1 := func(inputs, outputs map[string]interface{}) float64 {
		return 0.5
	}

	rewardFn2 := func(inputs, outputs map[string]interface{}) float64 {
		return 0.8
	}

	config1 := RefineConfig{
		N:         3,
		RewardFn:  rewardFn1,
		Threshold: 0.7,
	}

	refine := NewRefine(mockModule, config1)

	// Update configuration
	config2 := RefineConfig{
		N:         5,
		RewardFn:  rewardFn2,
		Threshold: 0.9,
	}

	refine.UpdateConfig(config2)

	updatedConfig := refine.GetConfig()
	assert.Equal(t, 5, updatedConfig.N)
	assert.Equal(t, 0.9, updatedConfig.Threshold)
}

func TestNewOfferFeedback(t *testing.T) {
	feedback := NewOfferFeedback()

	assert.NotNil(t, feedback)
	assert.NotNil(t, feedback.predict)

	sig := feedback.GetSignature()
	assert.Len(t, sig.Inputs, 4)
	assert.Len(t, sig.Outputs, 2)

	// Check input fields
	inputNames := make([]string, len(sig.Inputs))
	for i, field := range sig.Inputs {
		inputNames[i] = field.Name
	}
	assert.Contains(t, inputNames, "program_inputs")
	assert.Contains(t, inputNames, "program_outputs")
	assert.Contains(t, inputNames, "reward_value")
	assert.Contains(t, inputNames, "target_threshold")

	// Check output fields
	outputNames := make([]string, len(sig.Outputs))
	for i, field := range sig.Outputs {
		outputNames[i] = field.Name
	}
	assert.Contains(t, outputNames, "discussion")
	assert.Contains(t, outputNames, "advice")
}

func TestOfferFeedbackProcess(t *testing.T) {
	ctx := context.Background()

	mockLLM := &testutil.MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		&core.LLMResponse{
			Content: "<response><discussion>The performance needs improvement</discussion><advice>Try increasing temperature</advice></response>",
		}, nil,
	)

	feedback := NewOfferFeedback()
	feedback.SetLLM(mockLLM)

	inputs := map[string]interface{}{
		"program_inputs":   "test input",
		"program_outputs":  "weak output",
		"reward_value":     "0.3",
		"target_threshold": "0.8",
	}

	outputs, err := feedback.Process(ctx, inputs)

	require.NoError(t, err)
	require.NotNil(t, outputs)

	discussion, ok := outputs["discussion"].(string)
	assert.True(t, ok)
	assert.Contains(t, discussion, "performance needs improvement")

	advice, ok := outputs["advice"].(string)
	assert.True(t, ok)
	assert.Contains(t, advice, "increasing temperature")
}

func TestOfferFeedbackClone(t *testing.T) {
	original := NewOfferFeedback()
	clone := original.Clone().(*OfferFeedback)

	// Should be different instances
	assert.NotSame(t, original, clone)
	assert.NotSame(t, original.predict, clone.predict)

	// But should have same signature
	assert.Equal(t, original.GetSignature(), clone.GetSignature())
}

// Example test showing backwards compatibility with Python DSPy pattern.
func TestRefineBackwardsCompatibility(t *testing.T) {
	ctx := context.Background()

	// Python DSPy style usage:
	// refine = dspy.Refine(module=qa, N=3, reward_fn=check_answer, threshold=0.8)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	mockLLM := &testutil.MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		&core.LLMResponse{
			Content: "answer: The capital of France is Paris",
		}, nil,
	)

	// Create base module (equivalent to Python's 'module=qa')
	qa := NewPredict(signature)
	qa.SetLLM(mockLLM)

	// Create reward function (equivalent to Python's 'reward_fn=check_answer')
	checkAnswer := func(inputs, outputs map[string]interface{}) float64 {
		answer, ok := outputs["answer"].(string)
		if !ok {
			return 0.0
		}

		question, ok := inputs["question"].(string)
		if !ok {
			return 0.0
		}

		// Simple reward logic
		if strings.Contains(strings.ToLower(question), "capital") &&
			strings.Contains(strings.ToLower(answer), "paris") {
			return 1.0
		}
		return 0.5
	}

	// Create refine module with Python-compatible API
	config := RefineConfig{
		N:         3,           // N=3
		RewardFn:  checkAnswer, // reward_fn=check_answer
		Threshold: 0.8,         // threshold=0.8
	}
	refine := NewRefine(qa, config) // module=qa

	// Execute (equivalent to Python's refine(question="..."))
	inputs := map[string]interface{}{
		"question": "What is the capital of France?",
	}

	outputs, err := refine.Process(ctx, inputs)

	require.NoError(t, err)
	require.NotNil(t, outputs)

	answer, ok := outputs["answer"].(string)
	require.True(t, ok)
	assert.Contains(t, strings.ToLower(answer), "paris")
}
