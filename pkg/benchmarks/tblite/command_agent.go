package tblite

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

// CommandAgentConfig describes an external benchmark agent command.
// The command receives TerminalTaskRequest JSON on stdin and must emit
// TerminalTaskResult JSON on stdout.
type CommandAgentConfig struct {
	Command string
	Args    []string
	Env     []string
}

// CommandAgent adapts an external CLI process to the TBLite Agent interface.
type CommandAgent struct {
	config CommandAgentConfig
}

// NewCommandAgent creates a benchmark agent backed by an external command.
func NewCommandAgent(cfg CommandAgentConfig) (*CommandAgent, error) {
	if cfg.Command == "" {
		return nil, fmt.Errorf("command is required")
	}
	return &CommandAgent{config: cfg}, nil
}

// RunTask sends the task request to the configured command over stdin.
func (a *CommandAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	requestBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal task request: %w", err)
	}

	cmd := exec.CommandContext(ctx, a.config.Command, a.config.Args...)
	cmd.Dir = req.TaskDir
	cmd.Env = append(os.Environ(), a.config.Env...)
	cmd.Stdin = bytes.NewReader(requestBytes)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("run command agent: %w: %s", err, stderr.String())
	}

	if stdout.Len() == 0 {
		return nil, fmt.Errorf("command agent returned empty stdout")
	}

	var result TerminalTaskResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("parse command agent stdout: %w", err)
	}
	return &result, nil
}
