package tblite

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewToolCallingAgent_RequiresToolCalling(t *testing.T) {
	_, err := NewToolCallingAgent(&toolCallingStubLLM{}, ToolCallingAgentConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support tool calling")
}

func TestToolCallingAgent_RunTask_WritesTraceAndCompletes(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name": "write_file",
					"arguments": map[string]any{
						"path":    "answer.txt",
						"content": "done",
					},
				},
				"_usage": &core.TokenInfo{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			},
			{
				"function_call": map[string]any{
					"name": "Finish",
					"arguments": map[string]any{
						"answer": "updated answer.txt",
					},
				},
				"_usage": &core.TokenInfo{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
			},
		},
	}

	agent, err := NewToolCallingAgent(llm, ToolCallingAgentConfig{MaxTurns: 4})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	testsDir := filepath.Join(taskDir, "tests")
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-1",
		Instruction:      "Write answer.txt with the word done.",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         testsDir,
		TestScriptPath:   filepath.Join(taskDir, "test.sh"),
		MaxTurns:         4,
	})
	require.NoError(t, err)
	require.True(t, result.Completed)
	require.NotNil(t, llm.lastOptions)
	assert.Equal(t, "updated answer.txt", result.FinalAnswer)
	assert.Equal(t, 1, result.ToolCalls)
	assert.Equal(t, int64(9), result.TokenUsage.PromptTokens)
	assert.Equal(t, int64(5), result.TokenUsage.CompletionTokens)
	assert.Equal(t, int64(14), result.TokenUsage.TotalTokens)
	assert.Equal(t, 0.0, llm.lastOptions.Temperature)
	assert.NotEmpty(t, result.TracePath)

	content, err := os.ReadFile(filepath.Join(envDir, "answer.txt"))
	require.NoError(t, err)
	assert.Equal(t, "done", string(content))

	traceBytes, err := os.ReadFile(result.TracePath)
	require.NoError(t, err)
	assert.Contains(t, string(traceBytes), "\"tool_name\": \"write_file\"")
	assert.Contains(t, string(traceBytes), "\"final_answer\": \"updated answer.txt\"")
}

func TestToolCallingAgent_RunTask_FailsAfterMaxTurns(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{"content": "I think the answer is probably in the files."},
		},
	}

	agent, err := NewToolCallingAgent(llm, ToolCallingAgentConfig{MaxTurns: 1})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-2",
		Instruction:      "Use a tool before finishing.",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         filepath.Join(taskDir, "tests"),
		MaxTurns:         1,
	})
	require.NoError(t, err)
	require.False(t, result.Completed)
	assert.Contains(t, result.Error, "max turns reached")
}

type toolCallingStubLLM struct {
	results      []map[string]any
	index        int
	capabilities []core.Capability
	prompts      []string
	lastOptions  *core.GenerateOptions
}

func (m *toolCallingStubLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected Generate call")
}

func (m *toolCallingStubLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("unexpected GenerateWithJSON call")
}

func (m *toolCallingStubLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.prompts = append(m.prompts, prompt)
	opts := core.NewGenerateOptions()
	for _, option := range options {
		option(opts)
	}
	m.lastOptions = opts
	if m.index >= len(m.results) {
		return nil, fmt.Errorf("no more stubbed results")
	}
	result := m.results[m.index]
	m.index++
	return result, nil
}

func (m *toolCallingStubLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbedding call")
}

func (m *toolCallingStubLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbeddings call")
}

func (m *toolCallingStubLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerate call")
}

func (m *toolCallingStubLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected GenerateWithContent call")
}

func (m *toolCallingStubLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerateWithContent call")
}

func (m *toolCallingStubLLM) ProviderName() string { return "stub" }
func (m *toolCallingStubLLM) ModelID() string      { return "stub-model" }
func (m *toolCallingStubLLM) Capabilities() []core.Capability {
	return m.capabilities
}
