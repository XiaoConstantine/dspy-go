package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageContent_MarshalTextOnly(t *testing.T) {
	raw, err := json.Marshal(TextContent("hello"))
	require.NoError(t, err)
	assert.Equal(t, `"hello"`, string(raw))
}

func TestMessageContent_MarshalParts(t *testing.T) {
	content := PartsContent(
		ChatCompletionContentPart{Type: "text", Text: "see"},
		ChatCompletionContentPart{Type: "image_url", ImageURL: &ChatCompletionImageURLPart{URL: "data:image/png;base64,abc"}},
	)
	raw, err := json.Marshal(content)
	require.NoError(t, err)

	var parts []map[string]any
	require.NoError(t, json.Unmarshal(raw, &parts))
	require.Len(t, parts, 2)
	assert.Equal(t, "text", parts[0]["type"])
	assert.Equal(t, "image_url", parts[1]["type"])
}

func TestMessageContent_UnmarshalStringAndParts(t *testing.T) {
	var asString MessageContent
	require.NoError(t, json.Unmarshal([]byte(`"plain"`), &asString))
	assert.Equal(t, "plain", asString.Text())
	assert.False(t, asString.IsMultimodal())

	var asParts MessageContent
	require.NoError(t, json.Unmarshal([]byte(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`), &asParts))
	assert.Equal(t, "a\nb", asParts.Text())
	assert.True(t, asParts.IsMultimodal())
}
