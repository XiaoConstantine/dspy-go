package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

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
func (g *KeyGenerator) GenerateKey(modelID string, prompt string, options []core.GenerateOption) string {
	// Merge options to get final parameters
	opts := g.mergeOptions(options)

	// Create a normalized representation of all parameters
	keyData := g.createKeyData(modelID, prompt, opts)

	// Generate SHA256 hash
	h := sha256.New()
	h.Write([]byte(keyData))
	hash := hex.EncodeToString(h.Sum(nil))

	// Return prefixed key (truncate hash for readability)
	return fmt.Sprintf("%s%s_%s", g.prefix, modelID, hash[:16])
}

// GenerateJSONKey creates a cache key for JSON-structured requests.
func (g *KeyGenerator) GenerateJSONKey(modelID string, prompt string, schema interface{}, options []core.GenerateOption) string {
	// Merge options
	opts := g.mergeOptions(options)

	// Serialize schema
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		// If schema serialization fails, use empty schema
		schemaJSON = []byte("{}")
	}

	// Create key data including schema
	keyData := g.createKeyData(modelID, prompt, opts) + string(schemaJSON)

	// Generate hash
	h := sha256.New()
	h.Write([]byte(keyData))
	hash := hex.EncodeToString(h.Sum(nil))

	return fmt.Sprintf("%sjson_%s_%s", g.prefix, modelID, hash[:16])
}

// Content represents content for cache key generation.
type Content struct {
	Type string
	Data string
}

// GenerateContentKey creates a cache key for multimodal content requests.
func (g *KeyGenerator) GenerateContentKey(modelID string, contents []Content, options []core.GenerateOption) string {
	// Merge options
	opts := g.mergeOptions(options)

	// Create normalized content representation
	var contentParts []string
	for _, content := range contents {
		part := fmt.Sprintf("%s:%s", content.Type, content.Data)
		contentParts = append(contentParts, part)
	}
	sort.Strings(contentParts)

	// Create key data
	keyData := fmt.Sprintf("%s|%s|%s", modelID, strings.Join(contentParts, "|"), g.optionsToString(opts))

	// Generate hash
	h := sha256.New()
	h.Write([]byte(keyData))
	hash := hex.EncodeToString(h.Sum(nil))

	return fmt.Sprintf("%scontent_%s_%s", g.prefix, modelID, hash[:16])
}

// mergeOptions combines multiple generate options into a single config.
func (g *KeyGenerator) mergeOptions(options []core.GenerateOption) *core.GenerateOptions {
	config := core.NewGenerateOptions()

	for _, opt := range options {
		opt(config)
	}

	return config
}

// createKeyData creates a normalized string representation of request parameters.
func (g *KeyGenerator) createKeyData(modelID string, prompt string, config *core.GenerateOptions) string {
	// Normalize prompt (trim whitespace, lowercase for consistency)
	normalizedPrompt := strings.ToLower(strings.TrimSpace(prompt))

	// Create parameter string
	params := g.optionsToString(config)

	// Combine all parts
	return fmt.Sprintf("%s|%s|%s", modelID, normalizedPrompt, params)
}

// optionsToString converts generate config to a deterministic string.
func (g *KeyGenerator) optionsToString(config *core.GenerateOptions) string {
	// Create sorted parameter list for deterministic ordering
	var params []string

	// Add temperature with fixed precision
	params = append(params, fmt.Sprintf("temp:%.2f", config.Temperature))

	// Add max tokens
	params = append(params, fmt.Sprintf("max:%d", config.MaxTokens))

	// Add top-p if set
	if config.TopP > 0 {
		params = append(params, fmt.Sprintf("topp:%.2f", config.TopP))
	}

	// Add presence penalty if set
	if config.PresencePenalty != 0 {
		params = append(params, fmt.Sprintf("presence:%.2f", config.PresencePenalty))
	}

	// Add frequency penalty if set
	if config.FrequencyPenalty != 0 {
		params = append(params, fmt.Sprintf("frequency:%.2f", config.FrequencyPenalty))
	}

	// Add stop sequences if present
	if len(config.Stop) > 0 {
		stops := make([]string, len(config.Stop))
		copy(stops, config.Stop)
		sort.Strings(stops)
		params = append(params, fmt.Sprintf("stop:%s", strings.Join(stops, ",")))
	}

	// Sort parameters for consistency
	sort.Strings(params)

	return strings.Join(params, "|")
}

// InvalidatePattern generates a pattern for invalidating cache entries.
// This can be used to clear cache entries matching certain criteria.
func (g *KeyGenerator) InvalidatePattern(modelID string) string {
	if modelID == "" {
		return g.prefix + "*"
	}
	return fmt.Sprintf("%s%s_*", g.prefix, modelID)
}
