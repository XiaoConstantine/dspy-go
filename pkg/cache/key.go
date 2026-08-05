package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// KeyGenerator generates cache keys for LLM requests.
type KeyGenerator struct {
	// Prefix for all cache keys (e.g., "dspy_")
	prefix string
	// Include model version in key
	includeModelVersion bool
}

// NewKeyGenerator creates a new cache key generator.
func NewKeyGenerator(prefix string) *KeyGenerator {
	if prefix == "" {
		prefix = "dspy_"
	}
	return &KeyGenerator{
		prefix:              prefix,
		includeModelVersion: true,
	}
}

// GenerateKey creates a deterministic cache key from LLM request parameters.
// It returns an empty key when the request cannot be serialized safely.
func (g *KeyGenerator) GenerateKey(modelID string, prompt string, options []core.GenerateOption) string {
	// Merge options to get final parameters
	opts := g.mergeOptions(options)

	// Create an exact representation of all parameters.
	keyData, err := g.marshalKeyData(modelID, prompt, opts)
	if err != nil {
		return ""
	}

	// Generate SHA256 hash
	h := sha256.New()
	h.Write(keyData)
	hash := hex.EncodeToString(h.Sum(nil))

	return fmt.Sprintf("%s%s_%s", g.prefix, modelID, hash)
}

// GenerateJSONKey creates a cache key for JSON-structured requests.
// It returns an empty key when the request cannot be serialized safely.
func (g *KeyGenerator) GenerateJSONKey(modelID string, prompt string, schema any, options []core.GenerateOption) string {
	// Merge options
	opts := g.mergeOptions(options)

	// Serialize schema
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return ""
	}
	request, err := g.marshalKeyData(modelID, prompt, opts)
	if err != nil {
		return ""
	}

	// Create key data including schema
	keyData, err := json.Marshal(struct {
		Request string          `json:"request"`
		Schema  json.RawMessage `json:"schema"`
	}{string(request), schemaJSON})
	if err != nil {
		return ""
	}

	// Generate hash
	h := sha256.New()
	h.Write([]byte(keyData))
	hash := hex.EncodeToString(h.Sum(nil))

	return fmt.Sprintf("%sjson_%s_%s", g.prefix, modelID, hash)
}

// Content represents content for cache key generation.
type Content struct {
	Type     string
	Text     string
	Data     string
	MimeType string
	Metadata map[string]any
}

// GenerateContentKey creates a cache key for multimodal content requests.
// It returns an empty key when the request cannot be serialized safely.
func (g *KeyGenerator) GenerateContentKey(modelID string, contents []Content, options []core.GenerateOption) string {
	// Merge options
	opts := g.mergeOptions(options)

	// Content.Data remains a string for API compatibility, but it can contain
	// arbitrary image or audio bytes. Marshal a byte view so encoding/json uses
	// base64 instead of replacing invalid UTF-8 sequences with U+FFFD.
	type exactContent struct {
		Type     string         `json:"type"`
		Text     string         `json:"text"`
		Data     []byte         `json:"data"`
		MimeType string         `json:"mime_type"`
		Metadata map[string]any `json:"metadata"`
	}
	exactContents := make([]exactContent, len(contents))
	for i, content := range contents {
		exactContents[i] = exactContent{
			Type:     content.Type,
			Text:     content.Text,
			Data:     []byte(content.Data),
			MimeType: content.MimeType,
			Metadata: content.Metadata,
		}
	}

	// JSON preserves slice order and encoding/json deterministically orders map
	// keys. Explicit fields retain the complete content block identity.
	keyData, err := json.Marshal(struct {
		Model    string                `json:"model"`
		Contents []exactContent        `json:"contents"`
		Options  *core.GenerateOptions `json:"options"`
	}{modelID, exactContents, opts})
	if err != nil {
		return ""
	}

	// Generate hash
	h := sha256.New()
	h.Write([]byte(keyData))
	hash := hex.EncodeToString(h.Sum(nil))

	return fmt.Sprintf("%scontent_%s_%s", g.prefix, modelID, hash)
}

// mergeOptions combines multiple generate options into a single config.
func (g *KeyGenerator) mergeOptions(options []core.GenerateOption) *core.GenerateOptions {
	config := core.NewGenerateOptions()

	for _, opt := range options {
		opt(config)
	}

	return config
}

// marshalKeyData creates an exact serialized representation of request parameters.
func (g *KeyGenerator) marshalKeyData(modelID string, prompt string, config *core.GenerateOptions) ([]byte, error) {
	return json.Marshal(struct {
		Model   string                `json:"model"`
		Prompt  string                `json:"prompt"`
		Options *core.GenerateOptions `json:"options"`
	}{modelID, prompt, config})
}

// createKeyData retains the legacy string helper used by callers and tests.
// An empty result means the request cannot be safely cached.
func (g *KeyGenerator) createKeyData(modelID string, prompt string, config *core.GenerateOptions) string {
	data, err := g.marshalKeyData(modelID, prompt, config)
	if err != nil {
		return ""
	}
	return string(data)
}

// optionsToString converts generate config to a deterministic string.
func (g *KeyGenerator) optionsToString(config *core.GenerateOptions) string {
	data, err := json.Marshal(config)
	if err != nil {
		return ""
	}
	return string(data)
}

// InvalidatePattern generates a pattern for invalidating cache entries.
// This can be used to clear cache entries matching certain criteria.
func (g *KeyGenerator) InvalidatePattern(modelID string) string {
	if modelID == "" {
		return g.prefix + "*"
	}
	return fmt.Sprintf("%s%s_*", g.prefix, modelID)
}
