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

func TestNewNativeAgent_RequiresToolCalling(t *testing.T) {
	_, err := NewNativeAgent(&toolCallingStubLLM{}, NativeAgentConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support tool calling")
}

func TestNativeAgent_RunTask_WritesTraceAndCompletes(t *testing.T) {
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

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 4})
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

func TestNativeAgent_RunTask_FailsAfterMaxTurns(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{"content": "I think the answer is probably in the files."},
		},
	}

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 1})
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

func TestNativeAgent_BuildTaskPrompt_IncludesFinishGuidance(t *testing.T) {
	agent, err := NewNativeAgent(&toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}, NativeAgentConfig{MaxTurns: 20})
	require.NoError(t, err)

	prompt := agent.buildTaskPrompt(TerminalTaskRequest{
		TaskID:           "task-3",
		Instruction:      "Build the API and verify it.",
		TaskDir:          "/tmp/task",
		WorkingDirectory: "/tmp/task/environment",
		EnvironmentDir:   "/tmp/task/environment",
		TestsDir:         "/tmp/task/tests",
		TestScriptPath:   "/tmp/task/test.sh",
		MaxTurns:         20,
	})

	assert.Contains(t, prompt, "call Finish when done")
	assert.Contains(t, prompt, "health check, evaluation script, or verifier-style command succeeds")
}

func TestNativeAgent_RunTask_UsesNativeToolCallingWhenAvailable(t *testing.T) {
	llm := &nativeToolCallingStubLLM{
		toolCallingStubLLM: toolCallingStubLLM{
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
		},
	}

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 4})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	testsDir := filepath.Join(taskDir, "tests")
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-native",
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
	assert.Len(t, llm.messages, 2)
	assert.Empty(t, llm.prompts)
}

func TestNativeAgent_RunTask_FailsFastAfterRepeatedNoCallResponses(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"content": "No content or function call received from model",
				"provider_diagnostic": map[string]any{
					"provider":      "google",
					"provider_mode": "functions",
					"reason":        "empty_content_and_function_call",
					"finish_reason": "STOP",
				},
			},
			{
				"content": "No content or function call received from model",
				"provider_diagnostic": map[string]any{
					"provider":      "google",
					"provider_mode": "functions",
					"reason":        "empty_content_and_function_call",
					"finish_reason": "STOP",
				},
			},
			{
				"content": "No content or function call received from model",
				"provider_diagnostic": map[string]any{
					"provider":      "google",
					"provider_mode": "functions",
					"reason":        "empty_content_and_function_call",
					"finish_reason": "STOP",
				},
			},
		},
	}

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 20})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-no-call",
		Instruction:      "Use a tool before finishing.",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         filepath.Join(taskDir, "tests"),
		MaxTurns:         20,
	})
	require.NoError(t, err)
	require.False(t, result.Completed)
	assert.Contains(t, result.Error, "repeated model responses without tool calls")
	assert.NotEmpty(t, result.TracePath)

	traceBytes, err := os.ReadFile(result.TracePath)
	require.NoError(t, err)
	assert.Contains(t, string(traceBytes), "\"reason\": \"empty_content_and_function_call\"")
	assert.Contains(t, string(traceBytes), "\"finish_reason\": \"STOP\"")
}

func TestNativeAgent_RunTask_WritesTraceAndDebugSnapshotOnEarlyAgentError(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		generateErr:  fmt.Errorf("synthetic provider failure"),
	}

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 3})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	testsDir := filepath.Join(taskDir, "tests")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-error",
		Instruction:      "Inspect and fix the workspace.",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         testsDir,
		TestScriptPath:   filepath.Join(taskDir, "test.sh"),
		MaxTurns:         3,
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.Completed)
	assert.Contains(t, result.Error, "synthetic provider failure")

	traceBytes, readErr := os.ReadFile(filepath.Join(taskDir, traceFileName))
	require.NoError(t, readErr)
	assert.Contains(t, string(traceBytes), "\"error\": \"synthetic provider failure\"")

	debugBytes, readErr := os.ReadFile(filepath.Join(taskDir, debugFileName))
	require.NoError(t, readErr)
	assert.Contains(t, string(debugBytes), "\"error\": \"synthetic provider failure\"")
	assert.Contains(t, string(debugBytes), "\"tool_policy\"")
	assert.Contains(t, string(debugBytes), "\"task_prompt\"")
}

func TestNativeAgent_RunTask_PromptUsesContainerAliases(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name": "Finish",
					"arguments": map[string]any{
						"answer": "done",
					},
				},
			},
		},
	}

	agent, err := NewNativeAgent(llm, NativeAgentConfig{MaxTurns: 1})
	require.NoError(t, err)

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	testsDir := filepath.Join(taskDir, "tests")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "task-prompt",
		Instruction:      "Inspect the workspace.",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         testsDir,
		TestScriptPath:   filepath.Join(taskDir, "test.sh"),
		ContainerEnv: []string{
			EnvTaskRoot + "=" + containerTaskRoot,
			EnvTaskEnvDir + "=/app",
			EnvTaskTestsDir + "=" + containerTestsDir,
		},
		MaxTurns: 1,
	})
	require.NoError(t, err)
	require.True(t, result.Completed)
	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "- working directory: /app")
	assert.Contains(t, llm.prompts[0], "- tests directory: /task/tests")
	assert.Contains(t, llm.prompts[0], "- test script: /task/test.sh")
	assert.NotContains(t, llm.prompts[0], taskDir)
	assert.NotContains(t, llm.prompts[0], envDir)
	assert.NotContains(t, llm.prompts[0], testsDir)
}

type toolCallingStubLLM struct {
	results      []map[string]any
	index        int
	capabilities []core.Capability
	prompts      []string
	lastOptions  *core.GenerateOptions
	generateErr  error
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
	if m.generateErr != nil {
		return nil, m.generateErr
	}
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

type nativeToolCallingStubLLM struct {
	toolCallingStubLLM
	messages [][]core.ChatMessage
}

func (m *nativeToolCallingStubLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	messagesCopy := append([]core.ChatMessage(nil), messages...)
	m.messages = append(m.messages, messagesCopy)
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
