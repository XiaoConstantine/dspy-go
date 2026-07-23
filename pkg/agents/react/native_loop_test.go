package react

import (
	"context"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	contextmgmt "github.com/XiaoConstantine/dspy-go/pkg/agents/context"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReActAgentNativeFunctionCallingUsesSharedLoop(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{
		{"function_call": map[string]any{"id": "call-1", "name": "lookup", "arguments": map[string]any{"query": "facts"}}},
		{"function_call": map[string]any{"id": "finish-1", "name": "Finish", "arguments": map[string]any{"answer": "done"}}},
	}}
	agent := NewReActAgent("native-react", "Native ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{
		IncludeFinishTool: true,
	}))
	require.NoError(t, agent.RegisterTool(&mockTool{name: "lookup", result: core.ToolResult{Data: "found facts"}}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	).WithInstruction("Use tools to answer.")
	require.NoError(t, agent.Initialize(llm, signature))

	output, err := agent.Execute(context.Background(), map[string]any{"task": "find facts"})
	require.NoError(t, err)
	assert.Equal(t, "done", output["answer"])
	assert.Equal(t, true, output["completed"])
	assert.Equal(t, "finish", output["stop_reason"])
	assert.Equal(t, 2, llm.calls())

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, "finish", trace.TerminationCause)
	require.Len(t, trace.Steps, 2)
	assert.Equal(t, "lookup", trace.Steps[0].Tool)
	assert.Equal(t, "found facts", trace.Steps[0].Observation)
	assert.Equal(t, "Finish", trace.Steps[1].Tool)
}

func TestReActAgentNativeFunctionCallingMakesUnknownToolVisible(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{
		{"function_call": map[string]any{"id": "missing-1", "name": "missing", "arguments": map[string]any{}}},
		{"function_call": map[string]any{"id": "finish-1", "name": "Finish", "arguments": map[string]any{"answer": "recovered"}}},
	}}
	agent := NewReActAgent("native-react", "Native ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))

	output, err := agent.Execute(context.Background(), map[string]any{"task": "recover"})
	require.NoError(t, err)
	assert.Equal(t, "recovered", output["answer"])
	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.False(t, trace.Steps[0].Success)
	assert.Contains(t, trace.Steps[0].Error, "unknown tool")
	assert.True(t, trace.Steps[0].Synthetic)
	assert.Equal(t, agents.TraceStatusSuccess, trace.Status)
	assert.NotEmpty(t, trace.Steps[0].ActionRaw)
	assert.NotContains(t, trace.ToolUsageCount, "Finish")
}

func TestReActAgentNativeFunctionCallingUsesArtifactsAndNonStrictTextCompletion(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{{
		"content": "free-form answer",
		"_usage":  &core.TokenInfo{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}}}
	agent := NewReActAgent("native-react", "Native ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{
		IncludeFinishTool: true,
		StrictMode:        false,
	}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))
	require.NoError(t, agent.SetArtifacts(optimize.AgentArtifacts{Text: map[optimize.ArtifactKey]string{
		optimize.ArtifactSkillPack: "Use repository-specific evidence.",
	}}))

	output, err := agent.Execute(context.Background(), map[string]any{"task": "answer directly"})
	require.NoError(t, err)
	assert.Equal(t, "free-form answer", output["answer"])
	assert.Equal(t, true, output["completed"])
	assert.Contains(t, llm.lastPrompt(), "Use repository-specific evidence.")
	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, agents.TraceStatusSuccess, trace.Status)
	assert.Equal(t, int64(6), trace.TokenUsage["total_tokens"])
}

func TestReActAgentNativeFunctionCallingStoppedResultAgreesWithTrace(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{{"function_call": map[string]any{
		"id": "call-1", "name": "lookup", "arguments": map[string]any{},
	}}}}
	agent := NewReActAgent("native-react", "Native ReAct",
		WithMaxIterations(1),
		WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true, StrictMode: true}),
	)
	require.NoError(t, agent.RegisterTool(&mockTool{name: "lookup", result: core.ToolResult{Data: "not finished"}}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))

	output, err := agent.Execute(context.Background(), map[string]any{"task": "keep working"})
	require.NoError(t, err)
	assert.Equal(t, false, output["completed"])
	assert.Equal(t, "max_turns", output["stop_reason"])
	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, agents.TraceStatusPartial, trace.Status)
	assert.Equal(t, "max_turns", trace.TerminationCause)
}

func TestReActAgentNativeFunctionCallingFallsBackToTextWithoutCapability(t *testing.T) {
	llm := &mockLLM{response: &core.LLMResponse{Content: `thought: complete
action: <action><tool_name>Finish</tool_name></action>
answer: text fallback`}}
	agent := NewReActAgent("fallback-react", "Fallback ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))
	output, err := agent.Execute(context.Background(), map[string]any{"task": "finish"})
	require.NoError(t, err)
	assert.Equal(t, "text fallback", output["answer"])
}

func TestReActAgentNativeFunctionCallingRejectsUnrepresentableOutputs(t *testing.T) {
	llm := &nativeReActLLM{}
	agent := NewReActAgent("native-react", "Native ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}, {Field: core.NewField("reasoning")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))
	_, err := agent.Execute(context.Background(), map[string]any{"task": "work"})
	require.ErrorContains(t, err, "exactly one answer output")
	assert.Equal(t, 0, llm.calls())

	imageSignature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewImageField("image")}},
	)
	agent = NewReActAgent("native-react-image", "Native ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	require.NoError(t, agent.Initialize(llm, imageSignature))
	_, err = agent.Execute(context.Background(), map[string]any{"task": "work"})
	require.ErrorContains(t, err, `answer output "image" must be text`)
}

func TestReActAgentNativeCapabilityResolutionHandlesWrappersAndCycles(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	base := &nativeReActLLM{results: []map[string]any{{"content": "wrapped answer"}}}
	wrapped := core.NewModelContextDecorator(base)
	agent := NewReActAgent("wrapped-react", "Wrapped ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	require.NoError(t, agent.Initialize(wrapped, signature))
	output, err := agent.Execute(context.Background(), map[string]any{"task": "work"})
	require.NoError(t, err)
	assert.Equal(t, "wrapped answer", output["answer"])

	cycle := &core.BaseDecorator{}
	cycle.LLM = cycle
	agent = NewReActAgent("cycle-react", "Cycle ReAct", WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true}))
	require.NoError(t, agent.Initialize(cycle, signature))
	_, err = agent.Execute(context.Background(), map[string]any{"task": "work"})
	require.ErrorContains(t, err, "llm unwrap cycle detected")
}

func TestReActAgentNativeStoppedRunDoesNotPersistContextErrorOrSuccess(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{{"function_call": map[string]any{
		"id": "call-1", "name": "lookup", "arguments": map[string]any{},
	}}}}
	agent := NewReActAgent("context-stopped", "Context Stopped",
		WithMaxIterations(1),
		WithContextEngineering(t.TempDir(), contextmgmt.DefaultConfig()),
		WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: true, StrictMode: true}),
	)
	require.NoError(t, agent.RegisterTool(&mockTool{name: "lookup", result: core.ToolResult{Data: "partial"}}))
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))
	output, err := agent.Execute(context.Background(), map[string]any{"task": "stop after one turn"})
	require.NoError(t, err)
	assert.Equal(t, false, output["completed"])
	metrics := agent.contextManager.GetPerformanceMetrics()
	errorMetrics := metrics["errors"].(map[string]any)
	assert.Equal(t, int64(0), errorMetrics["total_errors"])
	assert.Equal(t, int64(0), errorMetrics["total_successes"])
}

func TestReActAgentNativeTraceRetainsContextMetadata(t *testing.T) {
	llm := &nativeReActLLM{results: []map[string]any{{"content": "context answer"}}}
	agent := NewReActAgent("context-native", "Context Native",
		WithContextEngineering(t.TempDir(), contextmgmt.DefaultConfig()),
		WithNativeFunctionCalling(interceptors.FunctionCallingConfig{IncludeFinishTool: false}),
	)
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	require.NoError(t, agent.Initialize(llm, signature))
	_, err := agent.Execute(context.Background(), map[string]any{"task": "use context"})
	require.NoError(t, err)
	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Contains(t, trace.ContextMetadata, "compression_ratio")
	assert.Contains(t, trace.ContextMetadata, "cost_savings")
}

func TestEnableNativeFunctionCallingCopiesAndResetsConfig(t *testing.T) {
	agent := NewReActAgent("native-react", "Native ReAct")
	require.NoError(t, agent.Initialize(&nativeReActLLM{}, core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)))
	config := interceptors.FunctionCallingConfig{IncludeFinishTool: false, StrictMode: true}
	require.NoError(t, agent.EnableNativeFunctionCalling(&config))
	config.StrictMode = false
	assert.True(t, agent.config.FunctionCallingConfig.StrictMode)

	require.NoError(t, agent.EnableNativeFunctionCalling(nil))
	assert.True(t, agent.config.FunctionCallingConfig.IncludeFinishTool)
	assert.False(t, agent.config.FunctionCallingConfig.StrictMode)
}

type nativeReActLLM struct {
	mockLLM
	mu      sync.Mutex
	results []map[string]any
	count   int
	prompts []string
}

func (m *nativeReActLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling}
}

func (m *nativeReActLLM) GenerateWithFunctions(_ context.Context, prompt string, _ []map[string]any, _ ...core.GenerateOption) (map[string]any, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.prompts = append(m.prompts, prompt)
	if m.count >= len(m.results) {
		return nil, assert.AnError
	}
	result := m.results[m.count]
	m.count++
	return result, nil
}

func (m *nativeReActLLM) lastPrompt() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.prompts) == 0 {
		return ""
	}
	return m.prompts[len(m.prompts)-1]
}

func (m *nativeReActLLM) calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.count
}
