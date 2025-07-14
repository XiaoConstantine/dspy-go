package cache

import (
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
		assert.Equal(t, 27, len(key)) // test_gpt-4_ (11) + hash (16)
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
		schema := map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{"type": "string"},
			},
		}
		key := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		assert.True(t, len(key) > 0)
		assert.Contains(t, key, "test_json_gpt-4_")
	})

	t.Run("Same inputs produce same JSON key", func(t *testing.T) {
		schema := map[string]interface{}{"type": "object"}
		key1 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		key2 := generator.GenerateJSONKey("gpt-4", "Generate JSON", schema, nil)
		assert.Equal(t, key1, key2)
	})

	t.Run("Different schemas produce different keys", func(t *testing.T) {
		schema1 := map[string]interface{}{"type": "object"}
		schema2 := map[string]interface{}{"type": "array"}
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

	t.Run("Order independence", func(t *testing.T) {
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
		assert.Equal(t, key1, key2)
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
		assert.Contains(t, keyData, "temp:0.70")
		assert.Contains(t, keyData, "max:100")
	})

	t.Run("Prompt normalization", func(t *testing.T) {
		config := core.NewGenerateOptions()
		keyData1 := generator.createKeyData("gpt-4", "  Hello world  ", config)
		keyData2 := generator.createKeyData("gpt-4", "Hello world", config)
		assert.Equal(t, keyData1, keyData2)
	})
}

func TestOptionsToString(t *testing.T) {
	generator := NewKeyGenerator("test_")

	t.Run("Basic options", func(t *testing.T) {
		config := core.NewGenerateOptions()
		core.WithTemperature(0.7)(config)
		core.WithMaxTokens(100)(config)

		result := generator.optionsToString(config)
		assert.Contains(t, result, "temp:0.70")
		assert.Contains(t, result, "max:100")
	})

	t.Run("All options", func(t *testing.T) {
		config := core.NewGenerateOptions()
		core.WithTemperature(0.7)(config)
		core.WithMaxTokens(100)(config)
		core.WithTopP(0.9)(config)
		core.WithPresencePenalty(0.5)(config)
		core.WithFrequencyPenalty(0.3)(config)
		core.WithStopSequences("stop1", "stop2")(config)

		result := generator.optionsToString(config)
		assert.Contains(t, result, "temp:0.70")
		assert.Contains(t, result, "max:100")
		assert.Contains(t, result, "topp:0.90")
		assert.Contains(t, result, "presence:0.50")
		assert.Contains(t, result, "frequency:0.30")
		assert.Contains(t, result, "stop:stop1,stop2")
	})

	t.Run("Default values are excluded", func(t *testing.T) {
		config := core.NewGenerateOptions()
		config.Temperature = 0.7
		config.MaxTokens = 100
		// TopP, PresencePenalty, FrequencyPenalty are 0, Stop is empty

		result := generator.optionsToString(config)
		assert.Contains(t, result, "temp:0.70")
		assert.Contains(t, result, "max:100")
		assert.NotContains(t, result, "topp:")
		assert.NotContains(t, result, "presence:")
		assert.NotContains(t, result, "frequency:")
		assert.NotContains(t, result, "stop:")
	})

	t.Run("Stop sequences are sorted", func(t *testing.T) {
		config := core.NewGenerateOptions()
		core.WithStopSequences("zebra", "apple", "banana")(config)

		result := generator.optionsToString(config)
		assert.Contains(t, result, "stop:apple,banana,zebra")
	})

	t.Run("Consistent ordering", func(t *testing.T) {
		config := core.NewGenerateOptions()
		core.WithTemperature(0.7)(config)
		core.WithMaxTokens(100)(config)
		core.WithTopP(0.9)(config)

		result1 := generator.optionsToString(config)
		result2 := generator.optionsToString(config)
		assert.Equal(t, result1, result2)
	})
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