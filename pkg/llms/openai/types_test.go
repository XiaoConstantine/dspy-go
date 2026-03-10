package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionRequestApplyOptions(t *testing.T) {
	t.Run("nil options", func(t *testing.T) {
		req := &ChatCompletionRequest{Model: "gpt-4o"}
		req.ApplyOptions(nil)
		assert.Nil(t, req.MaxTokens)
		assert.Nil(t, req.MaxCompletionTokens)
		assert.Nil(t, req.Temperature)
	})

	t.Run("gpt5 uses max completion tokens and omits temperature", func(t *testing.T) {
		req := &ChatCompletionRequest{Model: " GPT-5-mini "}
		req.ApplyOptions(&GenerateOptions{
			MaxTokens:        256,
			Temperature:      0,
			TopP:             0.9,
			PresencePenalty:  0.4,
			FrequencyPenalty: -0.1,
			Stop:             []string{"END"},
		})

		require.NotNil(t, req.MaxCompletionTokens)
		assert.Equal(t, 256, *req.MaxCompletionTokens)
		assert.Nil(t, req.MaxTokens)
		assert.Nil(t, req.Temperature)
		require.NotNil(t, req.TopP)
		assert.Equal(t, 0.9, *req.TopP)
		require.NotNil(t, req.PresencePenalty)
		assert.Equal(t, 0.4, *req.PresencePenalty)
		require.NotNil(t, req.FrequencyPenalty)
		assert.Equal(t, -0.1, *req.FrequencyPenalty)
		assert.Equal(t, []string{"END"}, req.Stop)
	})

	t.Run("non gpt5 uses max tokens and skips invalid values", func(t *testing.T) {
		req := &ChatCompletionRequest{Model: "gpt-4o"}
		req.ApplyOptions(&GenerateOptions{
			MaxTokens:        128,
			Temperature:      -1,
			TopP:             0,
			PresencePenalty:  0,
			FrequencyPenalty: 0,
		})

		require.NotNil(t, req.MaxTokens)
		assert.Equal(t, 128, *req.MaxTokens)
		assert.Nil(t, req.MaxCompletionTokens)
		assert.Nil(t, req.Temperature)
		assert.Nil(t, req.TopP)
		assert.Nil(t, req.PresencePenalty)
		assert.Nil(t, req.FrequencyPenalty)
	})
}

func TestUsesMaxCompletionTokens(t *testing.T) {
	assert.True(t, usesMaxCompletionTokens("gpt-5"))
	assert.True(t, usesMaxCompletionTokens(" GPT-5-nano "))
	assert.False(t, usesMaxCompletionTokens("gpt-4o"))
	assert.False(t, usesMaxCompletionTokens(""))
}

func TestSupportsCustomTemperature(t *testing.T) {
	assert.False(t, supportsCustomTemperature("gpt-5-mini"))
	assert.False(t, supportsCustomTemperature(" GPT-5.2-codex "))
	assert.True(t, supportsCustomTemperature("gpt-4o"))
	assert.True(t, supportsCustomTemperature(""))
}

func TestNewGenerateOptions(t *testing.T) {
	opts := NewGenerateOptions(512, 0.2, 0.95, 0.1, -0.2, []string{"STOP"})
	require.NotNil(t, opts)
	assert.Equal(t, 512, opts.MaxTokens)
	assert.Equal(t, 0.2, opts.Temperature)
	assert.Equal(t, 0.95, opts.TopP)
	assert.Equal(t, 0.1, opts.PresencePenalty)
	assert.Equal(t, -0.2, opts.FrequencyPenalty)
	assert.Equal(t, []string{"STOP"}, opts.Stop)
}
