package adapters

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/schema"
)

// AdapterCache caches validated adapter results keyed on the canonical rendered
// chat request, model identity, adapter identity, and generation options.
// Entries are written only after parsing and validation succeed.
type AdapterCache struct {
	store cache.Cache
	ttl   time.Duration
	// namespace prefixes every key. It is the routing identity of the model a
	// cache.CachedLLM wraps, for a cache taken from one.
	namespace string
}

// modelCache returns the validated-output cache for a model that is wrapped in
// an enabled cache.CachedLLM, as llms.NewLLM returns it: the wrapper's store,
// entry lifetime, and key namespace. It returns nil for any other model. An
// adapter with no cache of its own uses it, so a caller who had caching keeps
// it, for validated outputs only.
func modelCache(model core.LLM) *AdapterCache {
	for range 64 {
		if cached, ok := model.(*cache.CachedLLM); ok {
			store, namespace, ttl, ok := cached.ValidatedOutputCache()
			if !ok {
				return nil
			}
			return &AdapterCache{store: store, ttl: ttl, namespace: namespace}
		}
		wrapper, ok := model.(interface{ Unwrap() core.LLM })
		if !ok {
			return nil
		}
		if model = wrapper.Unwrap(); model == nil {
			return nil
		}
	}
	return nil
}

// NewAdapterCache wraps a cache.Cache store for validated adapter outputs.
func NewAdapterCache(store cache.Cache, ttl time.Duration) *AdapterCache {
	if store == nil {
		return nil
	}
	return &AdapterCache{store: store, ttl: ttl}
}

type cachedAdapterEntry struct {
	Outputs           map[string]any    `json:"outputs"`
	ToolCalls         []core.ToolCall   `json:"tool_calls,omitempty"`
	ReasoningSummary  string            `json:"reasoning_summary,omitempty"`
	FinishReason      core.FinishReason `json:"finish_reason,omitempty"`
	SchemaEnforcement string            `json:"schema_enforcement,omitempty"`
	FallbackUsed      bool              `json:"fallback_used,omitempty"`
}

type canonicalCacheMessage struct {
	Role            string                `json:"role"`
	Content         []canonicalCacheBlock `json:"content,omitempty"`
	ToolCalls       []core.ToolCall       `json:"tool_calls,omitempty"`
	ToolResults     []canonicalToolResult `json:"tool_results,omitempty"`
	RawProviderData json.RawMessage       `json:"raw_provider_data,omitempty"`
}

type canonicalCacheBlock struct {
	Type     core.FieldType `json:"type"`
	Text     string         `json:"text,omitempty"`
	DataB64  string         `json:"data_b64,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
}

type canonicalToolResult struct {
	ToolCallID string                `json:"tool_call_id"`
	Name       string                `json:"name"`
	Content    []canonicalCacheBlock `json:"content,omitempty"`
	IsError    bool                  `json:"is_error,omitempty"`
}

func canonicalCacheBlocks(blocks []core.ContentBlock) []canonicalCacheBlock {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]canonicalCacheBlock, len(blocks))
	for i, b := range blocks {
		cb := canonicalCacheBlock{
			Type:     b.Type,
			Text:     b.Text,
			MimeType: b.MimeType,
		}
		if len(b.Data) > 0 {
			cb.DataB64 = base64.StdEncoding.EncodeToString(b.Data)
		}
		out[i] = cb
	}
	return out
}

func buildAdapterCacheKey(
	adapterName, adapterVersion string,
	model core.LLM,
	req core.ChatRequest,
) (string, error) {
	genOpts := core.NewGenerateOptions()
	for _, opt := range req.Options {
		if opt != nil {
			opt(genOpts)
		}
	}

	msgs := make([]canonicalCacheMessage, len(req.Messages))
	for i, m := range req.Messages {
		cm := canonicalCacheMessage{
			Role:            m.Role,
			Content:         canonicalCacheBlocks(m.Content),
			ToolCalls:       m.ToolCalls,
			RawProviderData: m.RawProviderData,
		}
		if len(m.ToolResults) > 0 {
			cm.ToolResults = make([]canonicalToolResult, len(m.ToolResults))
			for j, tr := range m.ToolResults {
				cm.ToolResults[j] = canonicalToolResult{
					ToolCallID: tr.ToolCallID,
					Name:       tr.Name,
					Content:    canonicalCacheBlocks(tr.Content),
					IsError:    tr.IsError,
				}
			}
		} else if m.ToolResult != nil {
			cm.ToolResults = []canonicalToolResult{{
				ToolCallID: m.ToolResult.ToolCallID,
				Name:       m.ToolResult.Name,
				Content:    canonicalCacheBlocks(m.ToolResult.Content),
				IsError:    m.ToolResult.IsError,
			}}
		}
		msgs[i] = cm
	}

	payload := struct {
		AdapterName    string                  `json:"adapter_name"`
		AdapterVersion string                  `json:"adapter_version"`
		Provider       string                  `json:"provider"`
		Model          string                  `json:"model"`
		Messages       []canonicalCacheMessage `json:"messages"`
		Tools          []core.ChatTool         `json:"tools,omitempty"`
		ToolChoice     core.ToolChoice         `json:"tool_choice,omitempty"`
		ParallelTools  *bool                   `json:"parallel_tools,omitempty"`
		ResponseFormat core.ResponseFormatMode `json:"response_format"`
		ResponseSchema *core.ResponseSchema    `json:"response_schema,omitempty"`
		Reasoning      core.ReasoningEffort    `json:"reasoning_effort,omitempty"`
		Budget         int                     `json:"reasoning_budget,omitempty"`
		Options        *core.GenerateOptions   `json:"options"`
	}{
		AdapterName:    adapterName,
		AdapterVersion: adapterVersion,
		Provider:       model.ProviderName(),
		Model:          model.ModelID(),
		Messages:       msgs,
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		ParallelTools:  req.ParallelToolCalls,
		ResponseFormat: req.ResponseFormat,
		ResponseSchema: req.ResponseSchema,
		Reasoning:      req.ReasoningEffort,
		Budget:         req.ReasoningBudgetTokens,
		Options:        genOpts,
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("adapter:%s:%s:%s", adapterName, model.ModelID(), hex.EncodeToString(sum[:])), nil
}

// get returns the entry stored under key with its outputs in the Go types the
// adapter returned when it stored them. JSON does not keep those types, so
// every declared field is coerced again; an entry that no longer fits the
// signature is a miss.
func (c *AdapterCache) get(ctx context.Context, key string, signature core.Signature) (*cachedAdapterEntry, bool) {
	if c == nil || c.store == nil || key == "" {
		return nil, false
	}
	raw, found, err := c.store.Get(ctx, c.namespace+key)
	if err != nil || !found {
		return nil, false
	}
	var entry cachedAdapterEntry
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&entry); err != nil {
		return nil, false
	}
	for _, field := range signature.Outputs {
		value, ok := entry.Outputs[field.Name]
		if !ok {
			continue
		}
		typed, err := schema.CoerceFieldValue(field, value)
		if err != nil {
			return nil, false
		}
		entry.Outputs[field.Name] = typed
	}
	return &entry, true
}

// result returns the adapter result for a cache hit. It reports no usage,
// because a hit spends no tokens.
func (e *cachedAdapterEntry) result(adapter core.Adapter) *core.AdapterResult {
	return &core.AdapterResult{
		Outputs:           cloneOutputMap(e.Outputs),
		ToolCalls:         e.ToolCalls,
		ReasoningSummary:  e.ReasoningSummary,
		FinishReason:      e.FinishReason,
		AdapterName:       adapter.Name(),
		AdapterVersion:    adapter.Version(),
		FallbackUsed:      e.FallbackUsed,
		SchemaEnforcement: e.SchemaEnforcement,
		CacheHit:          true,
	}
}

func (c *AdapterCache) set(ctx context.Context, key string, entry cachedAdapterEntry) {
	if c == nil || c.store == nil || key == "" {
		return
	}
	raw, err := json.Marshal(entry)
	if err != nil {
		return
	}
	_ = c.store.Set(ctx, c.namespace+key, raw, c.ttl)
}
