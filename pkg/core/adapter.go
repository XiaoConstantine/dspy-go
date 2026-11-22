package core

import "context"

// AdapterCallOptions configures one adapter invocation.
type AdapterCallOptions struct {
	GenerateOptions       []GenerateOption
	StreamHandler         StreamHandler
	Tools                 []ChatTool
	ToolChoice            ToolChoice
	ParallelToolCalls     *bool
	ReasoningEffort       ReasoningEffort
	ReasoningBudgetTokens int
	DisableCache          bool
}

// AdapterResult carries the validated outputs and telemetry metadata from one
// adapter invocation.
type AdapterResult struct {
	Outputs           map[string]any
	ToolCalls         []ToolCall
	ReasoningSummary  string
	FinishReason      FinishReason
	Usage             *TokenInfo
	AdapterName       string
	AdapterVersion    string
	FallbackUsed      bool
	SchemaEnforcement string // "native", "json_object", "prompt_only", or ""
	CacheHit          bool
	RawContent        string
}

// Adapter translates a Signature, demos, and inputs into a model request and
// parses and validates the model response into typed output fields.
type Adapter interface {
	// Name returns a stable short identifier for this adapter (e.g. "xml", "json").
	Name() string
	// Version returns a stable format version string recorded in saved programs.
	Version() string
	// Format renders the signature, demos, and inputs into canonical chat messages.
	Format(signature Signature, demos []Example, inputs map[string]any) ([]ChatMessage, error)
	// Parse extracts and validates output fields from a raw completion string.
	Parse(signature Signature, content string) (map[string]any, error)
	// Call executes the full format -> model call -> parse + validate pipeline
	// (including streaming, native reasoning resolution, caching, and any
	// adapter-owned fallback).
	Call(ctx context.Context, model LLM, signature Signature, demos []Example, inputs map[string]any, opts AdapterCallOptions) (*AdapterResult, error)
}

// AdapterProvider is implemented by modules whose adapter identity should be
// persisted by SaveProgram.
type AdapterProvider interface {
	GetAdapterInfo() (name string, version string)
}

// AdapterConsumer is implemented by modules that can restore or inspect a
// saved adapter identity during LoadProgram.
type AdapterConsumer interface {
	SetAdapterByName(name string, version string) error
}
