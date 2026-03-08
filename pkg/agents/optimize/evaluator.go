package optimize

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// SideInfo carries diagnostic information beyond the scalar evaluation score.
type SideInfo struct {
	Trace       *agents.ExecutionTrace
	Diagnostics map[string]interface{}
	Scores      map[string]float64
	Cost        float64
	LatencyMS   float64
	Tokens      map[string]int64
	PassedTests []string
	FailedTests []string
}

// EvalResult is the output of an AgentEvaluator.
type EvalResult struct {
	Score    float64
	SideInfo *SideInfo
}

// AgentExample describes a single evaluation task for an optimizable agent.
type AgentExample struct {
	ID       string
	Inputs   map[string]interface{}
	Outputs  map[string]interface{}
	Metadata map[string]interface{}
}

// AgentEvaluator evaluates an agent on a concrete task and returns score plus ASI.
type AgentEvaluator interface {
	Evaluate(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error)
}
