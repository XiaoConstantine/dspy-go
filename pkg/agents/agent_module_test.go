package agents

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHarnessPromptFieldsUseSignatureInputs(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context")},
			{Field: core.NewField("query")},
		},
		[]core.OutputField{{Field: core.NewField("answer")}},
	).WithInstruction("Answer from context.")
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil, WithInstructions(signature), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)

	_, err = harness.Prompt(context.Background(), Prompt{Fields: map[string]any{
		"context": "facts", "query": "what matters?",
	}})
	require.NoError(t, err)
	require.Len(t, model.requests, 1)
	assert.Equal(t, "context: facts\nquery: what matters?", model.requests[0].Messages[1].TextContent())

	other, err := NewHarness(&loopTestModel{}, nil, WithInstructions(signature), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	_, err = other.Prompt(context.Background(), Prompt{Fields: map[string]any{"query": "missing context"}})
	require.ErrorContains(t, err, `missing signature input "context"`)
	_, err = other.Prompt(context.Background(), Prompt{Fields: map[string]any{"context": "facts", "query": "q", "question": "hidden"}})
	require.ErrorContains(t, err, `field "question" is not a signature input`)
}

func TestHarnessPromptFieldRenderingAllowsReentrantReads(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil, WithInstructions(signature), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() {
		_, promptErr := harness.Prompt(context.Background(), Prompt{Fields: map[string]any{
			"task": reentrantPromptValue{harness: harness},
		}})
		done <- promptErr
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("prompt field rendering deadlocked on reentrant Messages call")
	}
}

func TestHarnessPromptMetadataCloningAllowsReentrantCancel(t *testing.T) {
	harness, err := NewHarness(statelessAgentModuleModel{}, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	done := make(chan error, 1)
	go func() {
		_, promptErr := harness.Prompt(context.Background(), Prompt{
			Text: "work", Metadata: map[string]any{"hook": reentrantCancelValue{harness: harness}},
		})
		done <- promptErr
	}()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(2 * time.Second):
		t.Fatal("prompt metadata cloning deadlocked on reentrant Cancel call")
	}
}

func TestAgentModuleMapsNamedInputsAndOutputs(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{
			{Field: core.NewField("answer")},
			{Field: core.NewField("completed")},
			{Field: core.NewField("stop_reason")},
		},
	).WithInstruction("Complete the task.")
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "module answer")}}}
	harness, err := NewHarness(model, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	module, err := NewAgentModule(harness, signature)
	require.NoError(t, err)

	output, err := module.Process(context.Background(), map[string]any{"task": "do work"})
	require.NoError(t, err)
	assert.Equal(t, "module answer", output["answer"])
	assert.Equal(t, true, output["completed"])
	assert.Equal(t, "text", output["stop_reason"])
	require.Len(t, model.requests, 1)
	assert.Equal(t, "task: do work", model.requests[0].Messages[1].TextContent())
	assert.Equal(t, "agent_harness", module.GetModuleType())
}

func TestAgentModuleProcessIsolatesIndependentEvaluations(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	model := &loopTestModel{responses: []ModelResponse{
		{Message: NewTextMessage(RoleAssistant, "first answer")},
		{Message: NewTextMessage(RoleAssistant, "second answer")},
	}}
	harness, err := NewHarness(model, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	module, err := NewAgentModule(harness, signature)
	require.NoError(t, err)

	first, err := module.Process(context.Background(), map[string]any{"task": "first"})
	require.NoError(t, err)
	second, err := module.Process(context.Background(), map[string]any{"task": "second"})
	require.NoError(t, err)
	assert.Equal(t, "first answer", first["answer"])
	assert.Equal(t, "second answer", second["answer"])
	require.Len(t, model.requests, 2)
	assert.Len(t, model.requests[0].Messages, 2)
	assert.Len(t, model.requests[1].Messages, 2)
	assert.Equal(t, "task: second", model.requests[1].Messages[1].TextContent())

	cloned := module.Clone().(*AgentModule)
	updated := core.NewSignature(
		[]core.InputField{{Field: core.NewField("query")}},
		[]core.OutputField{{Field: core.NewField("result")}},
	)
	cloned.SetSignature(updated)
	assert.Equal(t, "query", cloned.GetSignature().Inputs[0].Name)
}

func TestAgentModuleConcurrentProcessAndSignatureUpdate(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	harness, err := NewHarness(statelessAgentModuleModel{}, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	module, err := NewAgentModule(harness, signature)
	require.NoError(t, err)

	var wait sync.WaitGroup
	errs := make(chan error, 50)
	for index := 0; index < 25; index++ {
		wait.Add(2)
		go func() {
			defer wait.Done()
			_, processErr := module.Process(context.Background(), map[string]any{"task": "work"})
			if processErr != nil {
				errs <- processErr
			}
		}()
		go func() {
			defer wait.Done()
			module.SetSignature(signature.WithInstruction("policy"))
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
}

func TestAgentModuleSetterErrorsDoNotMaskEachOther(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	newModule := func(t *testing.T) *AgentModule {
		harness, err := NewHarness(statelessAgentModuleModel{}, nil, WithHarnessCompletion(TextCompletion()))
		require.NoError(t, err)
		module, err := NewAgentModule(harness, signature)
		require.NoError(t, err)
		return module
	}

	module := newModule(t)
	module.SetSignature(core.Signature{})
	module.SetLLM(&adapterTestLLM{textResponse: &core.LLMResponse{Content: "done"}})
	_, err := module.Process(context.Background(), map[string]any{"task": "work"})
	require.ErrorContains(t, err, "signature requires at least one input")

	module = newModule(t)
	module.SetLLM(nil)
	module.SetSignature(signature.WithInstruction("valid"))
	_, err = module.Process(context.Background(), map[string]any{"task": "work"})
	require.ErrorContains(t, err, "llm is required")
}

func TestAgentModuleSetSignatureUpdatesPreRunPromptContract(t *testing.T) {
	initial := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	updated := core.NewSignature(
		[]core.InputField{{Field: core.NewField("query")}},
		[]core.OutputField{{Field: core.NewField("result")}},
	).WithInstruction("Use the optimized instruction.")
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "optimized")}}}
	harness, err := NewHarness(model, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	module, err := NewAgentModule(harness, initial)
	require.NoError(t, err)
	module.SetSignature(updated)

	output, err := module.Process(context.Background(), map[string]any{"query": "new input"})
	require.NoError(t, err)
	assert.Equal(t, "optimized", output["result"])
	assert.Contains(t, model.requests[0].Messages[0].TextContent(), "Use the optimized instruction.")
	assert.Equal(t, "query: new input", model.requests[0].Messages[1].TextContent())
}

type reentrantPromptValue struct {
	harness *Harness
}

func (v reentrantPromptValue) CloneMessageValue() any {
	_ = v.harness.Messages()
	return v
}

func (v reentrantPromptValue) String() string {
	_ = v.harness.Messages()
	return "work"
}

type reentrantCancelValue struct {
	harness *Harness
}

func (v reentrantCancelValue) CloneMessageValue() any {
	_ = v.harness.Cancel()
	return v
}

type statelessAgentModuleModel struct{}

func (statelessAgentModuleModel) Complete(context.Context, ModelRequest) (ModelResponse, error) {
	return ModelResponse{Message: NewTextMessage(RoleAssistant, "done")}, nil
}
func (statelessAgentModuleModel) ModelID() string      { return "stateless" }
func (statelessAgentModuleModel) ProviderName() string { return "test" }
