package tblite

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
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
	TaskID           string        `json:"task_id"`
	Instruction      string        `json:"instruction"`
	TaskDir          string        `json:"task_dir"`
	WorkingDirectory string        `json:"working_directory"`
	EnvironmentDir   string        `json:"environment_dir"`
	TestsDir         string        `json:"tests_dir"`
	TestScriptPath   string        `json:"test_script_path"`
	DockerImage      string        `json:"docker_image,omitempty"`
	ContainerID      string        `json:"container_id,omitempty"`
	ContainerEnv     []string      `json:"container_env,omitempty"`
	MaxTurns         int           `json:"max_turns,omitempty"`
	AgentTimeout     time.Duration `json:"agent_timeout,omitempty"`
}

// TokenUsage reports model usage for a benchmark task run.
type TokenUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// TerminalTaskResult captures the agent side of a benchmark run.
type TerminalTaskResult struct {
	Completed   bool           `json:"completed"`
	FinalAnswer string         `json:"final_answer,omitempty"`
	Error       string         `json:"error,omitempty"`
	Duration    time.Duration  `json:"duration"`
	ToolCalls   int            `json:"tool_calls,omitempty"`
	TokenUsage  TokenUsage     `json:"token_usage,omitempty"`
	TracePath   string         `json:"trace_path,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
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
	MaxTurns             int
	ShellPath            string
	ExtraEnv             []string
	UseTaskContainers    bool
	RespectAgentMaxTurns bool
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
		cfg.ShellPath = defaultShellPath
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
		AgentTimeout:     time.Duration(task.AgentTimeoutSec) * time.Second,
	}
	if !r.config.RespectAgentMaxTurns {
		req.MaxTurns = r.config.MaxTurns
	}

	var runtime *dockerTaskRuntime
	if r.config.UseTaskContainers && strings.TrimSpace(task.DockerImage) != "" {
		runtime, err = startDockerTaskRuntime(ctx, materialized, task.DockerImage)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = runtime.Close(context.Background())
		}()
		req.ContainerID = runtime.containerID
		req.ContainerEnv = []string{
			EnvTaskRoot + "=" + containerTaskRoot,
			EnvTaskEnvDir + "=" + runtime.containerEnvironmentRoot(),
			EnvTaskTestsDir + "=" + containerTestsDir,
		}
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

	testResult, err := r.executeVerifier(ctx, materialized, runtime)
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

func (r *Runner) executeVerifier(ctx context.Context, task *MaterializedTask, runtime *dockerTaskRuntime) (*TestResult, error) {
	startedAt := time.Now()

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	var runErr error
	if runtime != nil {
		var output []byte
		verifierEnv := append([]string{}, r.config.ExtraEnv...)
		verifierEnv = append(verifierEnv,
			EnvTaskRoot+"="+containerTaskRoot,
			EnvTaskEnvDir+"="+runtime.containerEnvironmentRoot(),
			EnvTaskTestsDir+"="+containerTestsDir,
		)
		output, runErr = runtime.Exec(ctx, task.EnvironmentDir, r.config.ShellPath, fmt.Sprintf("%s %s", r.config.ShellPath, containerTaskRoot+"/test.sh"), verifierEnv)
		if runErr != nil {
			stderr.Write(output)
		} else {
			stdout.Write(output)
		}
	} else {
		cmd := exec.CommandContext(ctx, r.config.ShellPath, task.TestScriptPath)
		cmd.Dir = task.EnvironmentDir
		cmd.Env = append(os.Environ(), r.config.ExtraEnv...)
		cmd.Env = append(cmd.Env,
			EnvTaskRoot+"="+task.RootDir,
			EnvTaskEnvDir+"="+task.EnvironmentDir,
			EnvTaskTestsDir+"="+task.TestsDir,
		)
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		runErr = cmd.Run()
	}
	duration := time.Since(startedAt)
	result := &TestResult{
		Passed:   runErr == nil,
		ExitCode: 0,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: duration,
	}

	if runtime != nil {
		reward, err := runtime.ReadFile(ctx, "/logs/verifier/reward.txt")
		if err != nil {
			result.Passed = false
			if result.ExitCode == 0 {
				result.ExitCode = 1
			}
			if result.Stderr != "" {
				result.Stderr += "\n"
			}
			result.Stderr += fmt.Sprintf("missing verifier reward: %v", err)
		} else {
			result.Passed = verifierRewardPassed(reward)
			if !result.Passed && result.ExitCode == 0 {
				result.ExitCode = 1
			}
		}
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

func verifierRewardPassed(reward string) bool {
	reward = strings.TrimSpace(reward)
	if reward == "" {
		return false
	}
	if reward == "1" {
		return true
	}
	value, err := strconv.ParseFloat(reward, 64)
	if err != nil {
		return false
	}
	return value >= 1.0
}
