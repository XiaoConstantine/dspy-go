package subagent

import (
	"context"
	"maps"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// SessionPolicy controls whether and how a child agent run should persist
// session state. The default is ephemeral so subagents stay isolated unless
// the caller opts into persistence.
type SessionPolicy int

const (
	SessionPolicyEphemeral SessionPolicy = iota
	SessionPolicyDerived
	SessionPolicyExplicit
)

func (p SessionPolicy) String() string {
	switch p {
	case SessionPolicyDerived:
		return "derived"
	case SessionPolicyExplicit:
		return "explicit"
	default:
		return "ephemeral"
	}
}

// ParentContext is best-effort metadata about the delegating parent run.
// Fields may be zero-valued when the parent runtime does not supply them.
type ParentContext struct {
	TaskID          string
	ParentAgentID   string
	ParentAgentType string
	SessionID       string
	BranchID        string
	Input           map[string]any
}

// ResultContext captures the child agent outcome surfaced to BuildResult.
type ResultContext struct {
	Output    map[string]any
	Duration  time.Duration
	Completed bool
	Trace     *agents.ExecutionTrace
}

// SubagentTraceRef is a concise trace summary that can be preserved in tool
// result details without forcing the full child transcript into the model text.
type SubagentTraceRef struct {
	Name      string                 `json:"name,omitempty"`
	Completed bool                   `json:"completed"`
	Duration  time.Duration          `json:"duration,omitempty"`
	Trace     *agents.ExecutionTrace `json:"trace,omitempty"`
}

type parentContextKey struct{}

// WithParentContext attaches best-effort parent metadata to a context so a
// subagent tool can project it into child input later.
func WithParentContext(ctx context.Context, parent ParentContext) context.Context {
	return context.WithValue(ctx, parentContextKey{}, cloneParentContext(parent))
}

// ParentContextFromContext returns any parent metadata attached to ctx.
func ParentContextFromContext(ctx context.Context) ParentContext {
	if ctx == nil {
		return ParentContext{}
	}
	parent, _ := ctx.Value(parentContextKey{}).(ParentContext)
	return cloneParentContext(parent)
}

func cloneParentContext(parent ParentContext) ParentContext {
	parent.Input = maps.Clone(parent.Input)
	return parent
}

func mergeParentContext(base, overlay ParentContext) ParentContext {
	merged := cloneParentContext(base)
	if overlay.TaskID != "" {
		merged.TaskID = overlay.TaskID
	}
	if overlay.ParentAgentID != "" {
		merged.ParentAgentID = overlay.ParentAgentID
	}
	if overlay.ParentAgentType != "" {
		merged.ParentAgentType = overlay.ParentAgentType
	}
	if overlay.SessionID != "" {
		merged.SessionID = overlay.SessionID
	}
	if overlay.BranchID != "" {
		merged.BranchID = overlay.BranchID
	}
	if overlay.Input != nil {
		merged.Input = maps.Clone(overlay.Input)
	}
	return merged
}
