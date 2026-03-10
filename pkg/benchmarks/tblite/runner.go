package tblite

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

const (
	EnvTaskRoot     = "DSPY_TBLITE_TASK_ROOT"
	EnvTaskEnvDir   = "DSPY_TBLITE_ENV_DIR"
	EnvTaskTestsDir = "DSPY_TBLITE_TESTS_DIR"
)

// Agent executes a single materialized terminal benchmark task.
// Implementations should return task-level failures in TerminalTaskResult.Error.
// The returned error is reserved for runner/transport failures.
type Agent interface {
	RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error)
}

// TerminalTaskRequest is the benchmark execution contract that a dspy-go
// terminal-capable agent must expose for TBLite-style tasks.
type TerminalTaskRequest struct {
	TaskID           string
	Instruction      string
	TaskDir          string
	WorkingDirectory string
	EnvironmentDir   string
	TestsDir         string
	TestScriptPath   string
	DockerImage      string
	MaxTurns         int
	AgentTimeout     time.Duration
}

// TokenUsage reports model usage for a benchmark task run.
type TokenUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// TerminalTaskResult captures the agent side of a benchmark run.
type TerminalTaskResult struct {
	Completed  bool              `json:"completed"`
	FinalAnswer string           `json:"final_answer,omitempty"`
	Error      string            `json:"error,omitempty"`
	Duration   time.Duration     `json:"duration"`
	ToolCalls  int               `json:"tool_calls,omitempty"`
	TokenUsage TokenUsage        `json:"token_usage,omitempty"`
	TracePath  string            `json:"trace_path,omitempty"`
	Metadata   map[string]any    `json:"metadata,omitempty"`
}

// TestResult captures benchmark verification output from test.sh.
type TestResult struct {
	Passed   bool          `json:"passed"`
	ExitCode int           `json:"exit_code"`
	Stdout   string        `json:"stdout,omitempty"`
	Stderr   string        `json:"stderr,omitempty"`
	Duration time.Duration `json:"duration"`
}

// EvaluationResult combines the agent run and benchmark verification outcome.
type EvaluationResult struct {
	Task             datasets.TBLiteTask `json:"task"`
	MaterializedTask *MaterializedTask   `json:"-"`
	AgentResult      *TerminalTaskResult `json:"agent_result,omitempty"`
	TestResult       *TestResult         `json:"test_result,omitempty"`
}

// RunnerConfig configures TBLite benchmark execution.
type RunnerConfig struct {
	MaxTurns int
	ShellPath string
	ExtraEnv []string
}

// Runner materializes and evaluates TBLite tasks using a benchmark agent.
type Runner struct {
	agent  Agent
	config RunnerConfig
}

// NewRunner creates a new TBLite benchmark runner.
func NewRunner(agent Agent, cfg RunnerConfig) *Runner {
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 60
	}
	if cfg.ShellPath == "" {
		cfg.ShellPath = "/bin/sh"
	}
	return &Runner{
		agent:  agent,
		config: cfg,
	}
}

// EvaluateTask runs a materialized TBLite task and executes its verifier.
func (r *Runner) EvaluateTask(ctx context.Context, task datasets.TBLiteTask, rootDir string) (*EvaluationResult, error) {
	if r.agent == nil {
		return nil, fmt.Errorf("tblite runner agent is required")
	}

	materialized, err := MaterializeTask(task, rootDir)
	if err != nil {
		return nil, err
	}

	req := TerminalTaskRequest{
		TaskID:           task.TaskName,
		Instruction:      task.Instruction,
		TaskDir:          materialized.RootDir,
		WorkingDirectory: materialized.EnvironmentDir,
		EnvironmentDir:   materialized.EnvironmentDir,
		TestsDir:         materialized.TestsDir,
		TestScriptPath:   materialized.TestScriptPath,
		DockerImage:      task.DockerImage,
		MaxTurns:         r.config.MaxTurns,
		AgentTimeout:     time.Duration(task.AgentTimeoutSec) * time.Second,
	}

	agentResult, err := r.agent.RunTask(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("run task agent: %w", err)
	}
	if agentResult == nil {
		agentResult = &TerminalTaskResult{
			Completed: false,
			Error:     "agent returned nil result",
		}
	}

	testResult, err := r.executeVerifier(ctx, materialized)
	if err != nil {
		return nil, err
	}

	return &EvaluationResult{
		Task:             task.Normalize(),
		MaterializedTask: materialized,
		AgentResult:      agentResult,
		TestResult:       testResult,
	}, nil
}

func (r *Runner) executeVerifier(ctx context.Context, task *MaterializedTask) (*TestResult, error) {
	startedAt := time.Now()

	cmd := exec.CommandContext(ctx, r.config.ShellPath, task.TestScriptPath)
	cmd.Dir = task.EnvironmentDir
	cmd.Env = append(os.Environ(), r.config.ExtraEnv...)
	cmd.Env = append(cmd.Env,
		EnvTaskRoot+"="+task.RootDir,
		EnvTaskEnvDir+"="+task.EnvironmentDir,
		EnvTaskTestsDir+"="+task.TestsDir,
	)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	duration := time.Since(startedAt)
	result := &TestResult{
		Passed:   runErr == nil,
		ExitCode: 0,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: duration,
	}

	if runErr == nil {
		return result, nil
	}

	var exitErr *exec.ExitError
	if errors.As(runErr, &exitErr) {
		result.ExitCode = exitErr.ExitCode()
		return result, nil
	}

	return nil, fmt.Errorf("run verifier: %w", runErr)
}
