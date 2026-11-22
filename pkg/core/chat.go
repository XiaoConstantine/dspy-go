package core

import (
	"context"
	"encoding/json"
	"slices"
)

// FinishReason describes why a chat generation stopped.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCall      FinishReason = "tool_call"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonRefusal       FinishReason = "refusal"
)

// ResponseFormatMode selects the wire response format requested from a ChatLLM.
type ResponseFormatMode uint8

const (
	ResponseFormatText ResponseFormatMode = iota
	ResponseFormatJSONObject
	ResponseFormatJSONSchema
)

// ToolStrictness controls provider-side schema enforcement policy.
type ToolStrictness string

const (
	ToolStrictDefault ToolStrictness = ""
	ToolStrictPrefer  ToolStrictness = "prefer"
	ToolStrictRequire ToolStrictness = "require"
)

// ResponseSchema describes an ordered JSON Schema for ResponseFormatJSONSchema.
type ResponseSchema struct {
	Name       string          `json:"name,omitempty"`
	Schema     json.RawMessage `json:"schema"`
	Strictness ToolStrictness  `json:"strictness,omitempty"`
}

// ToolChoiceMode controls whether and which tools the model may invoke.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = ""
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

// ToolChoice is a provider-neutral tool selection directive.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode,omitempty"`
	Name string         `json:"name,omitempty"`
}

// ReasoningEffort controls native model reasoning effort.
type ReasoningEffort string

const (
	ReasoningEffortDefault ReasoningEffort = ""
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
)

// ChatTool describes one function tool with an ordered raw JSON Schema.
type ChatTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      bool            `json:"strict,omitempty"`
	Strictness  ToolStrictness  `json:"strictness,omitempty"`
}

// ChatRequest is the unified provider-neutral chat generation request used by
// adapters, Predict, ChainOfThought, ReAct, and pkg/agents.
type ChatRequest struct {
	Messages              []ChatMessage
	Tools                 []ChatTool
	ToolChoice            ToolChoice
	ParallelToolCalls     *bool
	ResponseFormat        ResponseFormatMode
	ResponseSchema        *ResponseSchema
	ReasoningEffort       ReasoningEffort
	ReasoningBudgetTokens int
	Options               []GenerateOption
}

// ChatResponse is the unified provider-neutral chat generation result.
type ChatResponse struct {
	ID               string
	Model            string
	Content          string
	ContentBlocks    []ContentBlock
	ToolCalls        []ToolCall
	FinishReason     FinishReason
	Refusal          *string
	ReasoningSummary string
	ProviderData     json.RawMessage
	Usage            *TokenInfo
	Metadata         map[string]any
}

// ChatLLM is the unified typed chat transport implemented by GeneratorLLM and
// forwarded by decorators and cache wrappers.
type ChatLLM interface {
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	StreamChat(ctx context.Context, req ChatRequest) (*StreamResponse, error)
}

// History carries multi-turn conversation history as structured input/output
// turn maps so adapters render them as real user/assistant turns.
type History struct {
	Messages []map[string]any `json:"messages"`
}

type replyCacheBypassKey struct{}

// WithoutReplyCache marks ctx so that a wrapper that caches raw model replies,
// such as cache.CachedLLM, calls its model directly. Structured-output adapters
// use it: they cache only replies that passed validation, and a raw-reply cache
// would replay a reply that failed to parse.
func WithoutReplyCache(ctx context.Context) context.Context {
	return context.WithValue(ctx, replyCacheBypassKey{}, true)
}

// ReplyCacheBypassed reports whether ctx was marked by WithoutReplyCache.
func ReplyCacheBypassed(ctx context.Context) bool {
	bypassed, _ := ctx.Value(replyCacheBypassKey{}).(bool)
	return bypassed
}

// HasCapability reports whether llm advertises capability.
func HasCapability(llm CapabilityProvider, capability Capability) bool {
	if llm == nil {
		return false
	}
	return slices.Contains(llm.Capabilities(), capability)
}

// UnwrapLLM unwraps decorators/wrappers implementing Unwrap() LLM with a cycle guard.
func UnwrapLLM(model LLM) LLM {
	type unwrapper interface {
		Unwrap() LLM
	}
	current := model
	for range 64 {
		if current == nil {
			return nil
		}
		wrapped, ok := current.(unwrapper)
		if !ok {
			return current
		}
		next := wrapped.Unwrap()
		if next == nil || next == current {
			return current
		}
		current = next
	}
	return current
}

// ResolveChatLLM returns the outermost ChatLLM view of model when supported,
// checking wrapped models via Unwrap() while preserving outer decorators that
// also implement ChatLLM.
func ResolveChatLLM(model LLM) (ChatLLM, bool) {
	if model == nil {
		return nil, false
	}
	if base := UnwrapLLM(model); base != nil {
		if _, baseOK := base.(ChatLLM); !baseOK {
			return nil, false
		}
	}
	if chat, ok := model.(ChatLLM); ok {
		return chat, true
	}
	if base := UnwrapLLM(model); base != nil {
		if chat, ok := base.(ChatLLM); ok {
			return chat, true
		}
	}
	return nil, false
}
