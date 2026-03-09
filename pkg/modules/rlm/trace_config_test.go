package rlm

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRLMCompleteWithTrace_RecordsStructuredSteps(t *testing.T) {
	llm := &mockLLM{
		responses: []string{
			"Reasoning:\nI'll provide the answer directly.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ntrace answer",
		},
	}

	module := NewFromLLM(llm, WithMaxIterations(1))

	result, trace, err := module.CompleteWithTrace(context.Background(), "ctx", "what is the answer?")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, trace)

	assert.Equal(t, "trace answer", result.Response)
	assert.Equal(t, "final_answer", trace.TerminationCause)
	assert.Equal(t, 1, trace.Iterations)
	assert.Equal(t, "trace answer", trace.Output["answer"])
	require.Len(t, trace.Steps, 1)
	assert.Equal(t, "final", trace.Steps[0].Action)
	assert.Equal(t, "I'll provide the answer directly.", trace.Steps[0].Thought)
}

func TestRLMSetConfig_RebuildsPromptInstructions(t *testing.T) {
	llm := &mockLLM{
		responses: []string{
			"Reasoning:\nDone.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\ncomplete",
		},
	}

	module := NewFromLLM(llm)
	cfg := module.Config()
	cfg.OuterInstruction = "Custom outer instruction."
	cfg.IterationInstruction = "Custom iteration instruction."
	cfg.UseIterationDemos = true
	module.SetConfig(cfg)

	assert.Equal(t, "Custom outer instruction.", module.GetSignature().Instruction)
	require.NotNil(t, module.iterationModule)
	assert.Equal(t, "Custom iteration instruction.", module.iterationModule.GetSignature().Instruction)
	assert.True(t, module.Config().UseIterationDemos)
}
