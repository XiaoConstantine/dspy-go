package cache

import (
	"math"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestNewKeyGenerator(t *testing.T) {
	t.Run("With prefix", func(t *testing.T) {
		generator := NewKeyGenerator("test_")
		assert.Equal(t, "test_", generator.prefix)
		assert.True(t, generator.includeModelVersion)
	})

	t.Run("Empty prefix gets default", func(t *testing.T) {
		generator := NewKeyGenerator("")
		assert.Equal(t, "dspy_", generator.prefix)
		assert.True(t, generator.includeModelVersion)
	})
}

func TestGenerateKey(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Basic key generation", func(t *testing.T) {
		key := generator.GenerateKey("gpt-4", "Hello world", nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_gpt-4_")
		assert.Equal(t, 75, len(key)) // test_gpt-4_ (11) + SHA-256 (64)
	})

	t.Run("Same inputs produce same key", func(t *testing.T) {
		key1 := generator.GenerateKey("gpt-4", "Hello world", nil)
		key2 := generator.GenerateKey("gpt-4", "Hello world", nil)
		assert.Equal(t, key1, key2)
	})

	t.Run("Different inputs produce different keys", func(t *testing.T) {
		key1 := generator.GenerateKey("gpt-4", "Hello world", nil)
		key2 := generator.GenerateKey("gpt-4", "Hello universe", nil)
		assert.NotEqual(t, key1, key2)
	})

	t.Run("Different models produce different keys", func(t *testing.T) {
		key1 := generator.GenerateKey("gpt-4", "Hello world", nil)
		key2 := generator.GenerateKey("gpt-3.5", "Hello world", nil)
		assert.NotEqual(t, key1, key2)
	})

	t.Run("With options", func(t *testing.T) {
		options := []core.GenerateOption{
			core.WithTemperature(0.7),
			core.WithMaxTokens(100),
		}
		key1 := generator.GenerateKey("gpt-4", "Hello world", options)
		key2 := generator.GenerateKey("gpt-4", "Hello world", nil)
		assert.NotEqual(t, key1, key2)
	})
}

func TestGenerateJSONKey(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Basic JSON key generation", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		key := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_json_gpt-4_")
	})

	t.Run("Same inputs produce same JSON key", func(t *testing.T) {
		schema := map[string]any{"type": "object"}
		key1 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		key2 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		assert.Equal(t, key1, key2)
	})

	t.Run("Different schemas produce different keys", func(t *testing.T) {
		schema1 := map[string]any{"type": "object"}
		schema2 := map[string]any{"type": "array"}
		key1 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema1, nil)
		key2 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema2, nil)
		assert.NotEqual(t, key1, key2)
	})

	t.Run("With nil schema", func(t *testing.T) {
		key := generator.GenerateJSONKey("gpt-4", "Generate JSON", nil, nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_json_gpt-4_")
	})
}

func TestGenerateContentKey(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Basic content key generation", func(t *testing.T) {
		contents := []Content{
			{Type: "text", Data: "Hello world"},
			{Type: "image", Data: "base64data"},
		}
		key := generator.GenerateContentKey("gpt-4", contents, nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_content_gpt-4_")
	})

	t.Run("Same inputs produce same content key", func(t *testing.T) {
		contents := []Content{
			{Type: "text", Data: "Hello world"},
		}
		key1 := generator.GenerateContentKey("gpt-4", contents, nil)
		key2 := generator.GenerateContentKey("gpt-4", contents, nil)
		assert.Equal(t, key1, key2)
	})

	t.Run("Different content produces different keys", func(t *testing.T) {
		contents1 := []Content{
			{Type: "text", Data: "Hello world"},
		}
		contents2 := []Content{
			{Type: "text", Data: "Hello universe"},
		}
		key1 := generator.GenerateContentKey("gpt-4", contents1, nil)
		key2 := generator.GenerateContentKey("gpt-4", contents2, nil)
		assert.NotEqual(t, key1, key2)
	})

	t.Run("Order is preserved", func(t *testing.T) {
		contents1 := []Content{
			{Type: "text", Data: "Hello"},
			{Type: "image", Data: "data"},
		}
		contents2 := []Content{
			{Type: "image", Data: "data"},
			{Type: "text", Data: "Hello"},
		}
		key1 := generator.GenerateContentKey("gpt-4", contents1, nil)
		key2 := generator.GenerateContentKey("gpt-4", contents2, nil)
		assert.NotEqual(t, key1, key2)
	})

	t.Run("Empty contents", func(t *testing.T) {
		key := generator.GenerateContentKey("gpt-4", []Content{}, nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_content_gpt-4_")
	})

	t.Run("With options", func(t *testing.T) {
		contents := []Content{
			{Type: "text", Data: "Hello world"},
		}
		options := []core.GenerateOption{
			core.WithTemperature(0.7),
		}
		key1 := generator.GenerateContentKey("gpt-4", contents, options)
		key2 := generator.GenerateContentKey("gpt-4", contents, nil)
		assert.NotEqual(t, key1, key2)
	})
}

func TestMergeOptions(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Merge single option", func(t *testing.T) {
		options := []core.GenerateOption{
			core.WithTemperature(0.7),
		}
		config := generator.mergeOptions(options)
		assert.Equal(t, 0.7, config.Temperature)
	})

	t.Run("Merge multiple options", func(t *testing.T) {
		options := []core.GenerateOption{
			core.WithTemperature(0.7),
			core.WithMaxTokens(100),
			core.WithTopP(0.9),
		}
		config := generator.mergeOptions(options)
		assert.Equal(t, 0.7, config.Temperature)
		assert.Equal(t, 100, config.MaxTokens)
		assert.Equal(t, 0.9, config.TopP)
	})

	t.Run("Empty options", func(t *testing.T) {
		config := generator.mergeOptions(nil)
		assert.NotNil(t, config)
	})

	t.Run("Overlapping options", func(t *testing.T) {
		options := []core.GenerateOption{
			core.WithTemperature(0.5),
			core.WithTemperature(0.7), // Should override
		}
		config := generator.mergeOptions(options)
		assert.Equal(t, 0.7, config.Temperature)
	})
}

func TestCreateKeyData(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Basic key data creation", func(t *testing.T) {
		config := core.NewGenerateOptions()
		core.WithTemperature(0.7)(config)
		core.WithMaxTokens(100)(config)

		keyData := generator.createKeyData("gpt-4", "Hello world", config)
		assert.Contains(t, keyData, "gpt-4")
		assert.Contains(t, keyData, "Hello world")
		assert.Contains(t, keyData, `"Temperature":0.7`)
		assert.Contains(t, keyData, `"MaxTokens":100`)
	})

	t.Run("Prompt whitespace is exact", func(t *testing.T) {
		config := core.NewGenerateOptions()
		keyData1 := generator.createKeyData("gpt-4", "  Hello world  ", config)
		keyData2 := generator.createKeyData("gpt-4", "Hello world", config)
		assert.NotEqual(t, keyData1, keyData2)
	})

	t.Run("Prompt case is exact", func(t *testing.T) {
		config := core.NewGenerateOptions()
		keyData1 := generator.createKeyData("gpt-4", "Hello World", config)
		keyData2 := generator.createKeyData("gpt-4", "hello world", config)
		assert.NotEqual(t, keyData1, keyData2)
	})
}

func TestOptionsToString(t *testing.T) {
	generator := NewKeyGenerator("test_")
	config := core.NewGenerateOptions()
	core.WithTemperature(0.7001)(config)
	core.WithMaxTokens(100)(config)
	core.WithTopP(0.9)(config)
	core.WithPresencePenalty(0.5)(config)
	core.WithFrequencyPenalty(0.3)(config)
	core.WithStopSequences("zebra", "apple")(config)

	result := generator.optionsToString(config)
	assert.Contains(t, result, `"Temperature":0.7001`)
	assert.Contains(t, result, `"TopP":0.9`)
	assert.Contains(t, result, `"PresencePenalty":0.5`)
	assert.Contains(t, result, `"FrequencyPenalty":0.3`)
	assert.Contains(t, result, `"Stop":["zebra","apple"]`)
	assert.Equal(t, result, generator.optionsToString(config))
}

func TestGenerateKeyPreservesOptionPrecisionAndStopOrder(t *testing.T) {
	generator := NewKeyGenerator("test_")
	precise := generator.GenerateKey("model", "prompt", []core.GenerateOption{core.WithTemperature(0.701)})
	rounded := generator.GenerateKey("model", "prompt", []core.GenerateOption{core.WithTemperature(0.704)})
	assert.NotEqual(t, precise, rounded)

	first := generator.GenerateKey("model", "prompt", []core.GenerateOption{core.WithStopSequences("a", "b")})
	second := generator.GenerateKey("model", "prompt", []core.GenerateOption{core.WithStopSequences("b", "a")})
	assert.NotEqual(t, first, second)
}

func TestKeyGenerationRejectsUnserializableRequests(t *testing.T) {
	generator := NewKeyGenerator("test_")

	assert.Empty(t, generator.GenerateKey("model", "prompt", []core.GenerateOption{
		core.WithTemperature(math.NaN()),
	}))
	assert.Empty(t, generator.GenerateJSONKey("model", "prompt", map[string]any{
		"invalid": make(chan int),
	}, nil))
	assert.Empty(t, generator.GenerateContentKey("model", []Content{{
		Type:     "text",
		Text:     "prompt",
		Metadata: map[string]any{"invalid": func() {}},
	}}, nil))
}

func TestContentKeyIncludesCompleteIdentityAndDeterministicMetadata(t *testing.T) {
	generator := NewKeyGenerator("test_")
	base := Content{Type: "image", Text: "caption", Data: "bytes", MimeType: "image/png", Metadata: map[string]any{"a": 1, "b": 2}}
	same := Content{Type: "image", Text: "caption", Data: "bytes", MimeType: "image/png", Metadata: map[string]any{"b": 2, "a": 1}}
	assert.Equal(t, generator.GenerateContentKey("model", []Content{base}, nil), generator.GenerateContentKey("model", []Content{same}, nil))

	variants := []Content{
		{Type: "image", Text: "other", Data: "bytes", MimeType: "image/png", Metadata: base.Metadata},
		{Type: "image", Text: "caption", Data: "other", MimeType: "image/png", Metadata: base.Metadata},
		{Type: "image", Text: "caption", Data: "bytes", MimeType: "image/jpeg", Metadata: base.Metadata},
		{Type: "image", Text: "caption", Data: "bytes", MimeType: "image/png", Metadata: map[string]any{"a": 2, "b": 2}},
	}
	baseKey := generator.GenerateContentKey("model", []Content{base}, nil)
	for _, variant := range variants {
		assert.NotEqual(t, baseKey, generator.GenerateContentKey("model", []Content{variant}, nil))
	}
}

func TestContentKeyPreservesArbitraryBinaryData(t *testing.T) {
	generator := NewKeyGenerator("test_")
	first := generator.GenerateContentKey("model", []Content{{
		Type: "image",
		Data: string([]byte{0xff}),
	}}, nil)
	second := generator.GenerateContentKey("model", []Content{{
		Type: "image",
		Data: string([]byte{0xfe}),
	}}, nil)

	assert.NotEqual(t, first, second, "distinct binary payloads must not share a cache key")
}

func TestInvalidatePattern(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Model-specific pattern", func(t *testing.T) {
		pattern := generator.InvalidatePattern("gpt-4")
		assert.Equal(t, "test_gpt-4_*", pattern)
	})

	t.Run("All models pattern", func(t *testing.T) {
		pattern := generator.InvalidatePattern("")
		assert.Equal(t, "test_*", pattern)
	})
}

func TestContent(t *testing.T) {
	content := Content{
		Type: "text",
		Data: "Hello world",
	}

	assert.Equal(t, "text", content.Type)
	assert.Equal(t, "Hello world", content.Data)
}

func TestKeyGeneratorDeterministic(t *testing.T) {
	generator := NewKeyGenerator("test_")

	// Test that the same inputs always produce the same output
	modelID := "gpt-4"
	prompt := "Hello world"
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
		core.WithMaxTokens(100),
	}

	key1 := generator.GenerateKey(modelID, prompt, options)
	key2 := generator.GenerateKey(modelID, prompt, options)
	key3 := generator.GenerateKey(modelID, prompt, options)

	assert.Equal(t, key1, key2)
	assert.Equal(t, key2, key3)
}

func TestKeyGeneratorSensitivity(t *testing.T) {
	generator := NewKeyGenerator("test_")

	baseKey := generator.GenerateKey("gpt-4", "Hello world", []core.GenerateOption{
		core.WithTemperature(0.7),
	})

	// Small temperature change should produce different key
	diffKey := generator.GenerateKey("gpt-4", "Hello world", []core.GenerateOption{
		core.WithTemperature(0.71),
	})

	assert.NotEqual(t, baseKey, diffKey)
}
