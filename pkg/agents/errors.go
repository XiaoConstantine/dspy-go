package agents

import "errors"

var (
	// ErrAgentFinished indicates the agent has completed execution.
	ErrAgentFinished = errors.New("agent has finished execution")

	// ErrMaxIterations indicates the agent hit the iteration limit.
	ErrMaxIterations = errors.New("max iterations reached")

	// ErrContextOverflow indicates the context window was exceeded.
	ErrContextOverflow = errors.New("context window overflow")

	// ErrAborted indicates the agent execution was aborted.
	ErrAborted = errors.New("agent execution aborted")
)

// IsContextOverflow checks if an error is or wraps ErrContextOverflow.
func IsContextOverflow(err error) bool { return errors.Is(err, ErrContextOverflow) }
