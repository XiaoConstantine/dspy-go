package agents

import (
	"context"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// LoopStrategy defines how to interact with the LLM within the loop.
// Strategy owns the LLM call (via CallLLM), not RunLoop.
type LoopStrategy interface {
	PrepareRequest(ctx context.Context, messages []Message, tools []core.Tool) (*LLMRequest, error)
	CallLLM(ctx context.Context, env *LoopEnv, req *LLMRequest) (*LLMResult, error)
	ParseResponse(ctx context.Context, response *LLMResult) (*LoopAction, error)
	ShouldContinue(ctx context.Context, action *LoopAction, iteration int) bool
	ToolUsePolicy() string
}

// SignatureStrategy is an optional interface for strategies that carry
// a signature. Optimizers use this to access and modify the signature
// without knowing the concrete strategy type.
type SignatureStrategy interface {
	LoopStrategy
	Signature() core.Signature
	SetSignature(sig core.Signature)
	Demos() []core.Example
	SetDemos(demos []core.Example)
}

// NeedsLLM is an optional interface for strategies that need LLM propagation.
// RunLoop calls SetLLM before the first iteration if the strategy implements this.
type NeedsLLM interface {
	SetLLM(llm core.LLM)
}

// LoopEnv holds the environment available to strategies during LLM calls.
type LoopEnv struct {
	LLM          core.LLM
	Tools        core.ToolRegistry
	GenerateOpts []core.GenerateOption
	ModuleOpts   []core.Option
}

// LoopConfig holds all configuration for a RunLoop execution.
type LoopConfig struct {
	LLM                core.LLM
	Tools              core.ToolRegistry
	Strategy           LoopStrategy
	MaxIterations      int
	TransformContext   func(ctx context.Context, messages []Message) ([]Message, error)
	OnEvent            func(AgentEvent)
	GenerateOpts       []core.GenerateOption
	ModuleOpts         []core.Option
	ParallelTools      bool
	MaxToolConcurrency int
	ToolTimeout        time.Duration
}

const (
	StopReasonFinish           = "finish"
	StopReasonMaxIterations    = "max_iterations"
	StopReasonError            = "error"
	StopReasonAborted          = "aborted"
	StopReasonBudgetExhausted  = "budget_exhausted"
	StopReasonFinishOverBudget = "finish_over_budget"
)

// LoopResult holds the result of a RunLoop execution.
type LoopResult struct {
	Messages   []Message
	Output     map[string]any
	// StopReason is one of the StopReason* constants.
	StopReason string
	Iterations int
	Usage      *core.TokenInfo
}

// LoopAction represents parsed LLM output — what the loop should do next.
type LoopAction struct {
	ToolCalls  []core.ToolCall
	Thought    string
	Result     map[string]any
	StopReason string
}

// LLMRequest is built by PrepareRequest for the strategy's CallLLM.
type LLMRequest struct {
	Prompt       string
	Content      []core.ContentBlock
	Functions    []map[string]interface{}
	ChatMessages []core.ChatMessage
	Options      []core.GenerateOption
}

// LLMResult is returned by CallLLM and consumed by ParseResponse.
type LLMResult struct {
	TextResponse     *core.LLMResponse
	FunctionResponse map[string]interface{}
	ModuleResponse   map[string]any
}
