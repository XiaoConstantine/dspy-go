package schema

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompileSignatureSchemaPreservesFieldOrder(t *testing.T) {
	type Nested struct {
		Zeta  string `json:"zeta"`
		Alpha *int   `json:"alpha"`
		Flag  bool   `json:"flag"`
	}
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{
			{Field: core.NewField("reasoning", core.WithDescription("Step-by-step reasoning"))},
			{Field: core.NewField("answer", core.WithDescription("Final answer"))},
			{Field: core.NewField("score", core.WithGoType(reflect.TypeOf(0)))},
			{Field: core.NewField("nested", core.WithGoType(reflect.TypeOf(Nested{})))},
			{Field: core.NewField("note", core.WithOptional())},
		},
	)

	compiled, err := CompileSignatureSchema(sig)
	require.NoError(t, err)
	assert.True(t, compiled.StrictCompatible)
	assert.Equal(t, []string{"reasoning", "answer", "score", "nested", "note"}, compiled.PropertyOrder)

	raw := string(compiled.Raw)
	idxReasoning := strings.Index(raw, `"reasoning":`)
	idxAnswer := strings.Index(raw, `"answer":`)
	idxScore := strings.Index(raw, `"score":`)
	idxNested := strings.Index(raw, `"nested":`)
	idxNote := strings.Index(raw, `"note":`)
	require.Greater(t, idxReasoning, 0)
	assert.Less(t, idxReasoning, idxAnswer, "reasoning must precede answer in compiled schema")
	assert.Less(t, idxAnswer, idxScore)
	assert.Less(t, idxScore, idxNested)
	assert.Less(t, idxNested, idxNote)

	// Nested struct also preserves declaration order (zeta before alpha before flag).
	idxZeta := strings.Index(raw, `"zeta":`)
	idxAlpha := strings.Index(raw, `"alpha":`)
	idxFlag := strings.Index(raw, `"flag":`)
	assert.Less(t, idxZeta, idxAlpha)
	assert.Less(t, idxAlpha, idxFlag)

	// Optional fields are nullable and listed in required (strict-compatible form).
	assert.Contains(t, raw, `"note":{"type":["string","null"]}`)
	assert.Contains(t, raw, `"alpha":{"type":["integer","null"]}`)
	assert.Contains(t, raw, `"additionalProperties":false`)
}

func TestCompileSignatureSchemaOpenMapNotStrictCompatible(t *testing.T) {
	sig := core.NewSignature(
		nil,
		[]core.OutputField{
			{Field: core.NewField("metadata", core.WithGoType(reflect.TypeOf(map[string]any{})))},
		},
	)
	compiled, err := CompileSignatureSchema(sig)
	require.NoError(t, err)
	assert.False(t, compiled.StrictCompatible)
}

func TestValidateJSONStrictness(t *testing.T) {
	sig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("answer")},
		{Field: core.NewField("count", core.WithGoType(reflect.TypeOf(0)))},
		{Field: core.NewField("opt", core.WithOptional())},
	})
	compiled, err := CompileSignatureSchema(sig)
	require.NoError(t, err)

	// Valid including blank string, zero int, null optional.
	require.NoError(t, ValidateJSON(compiled.Raw, json.RawMessage(`{"answer":"","count":0,"opt":null}`)))

	// Rejects missing required field.
	require.Error(t, ValidateJSON(compiled.Raw, json.RawMessage(`{"answer":"ok","opt":null}`)))

	// Rejects wrong type.
	require.Error(t, ValidateJSON(compiled.Raw, json.RawMessage(`{"answer":"ok","count":"five","opt":null}`)))

	// Rejects additionalProperties.
	require.Error(t, ValidateJSON(compiled.Raw, json.RawMessage(`{"answer":"ok","count":1,"opt":null,"extra":true}`)))

	// Rejects duplicate object keys.
	require.Error(t, ValidateJSON(compiled.Raw, json.RawMessage(`{"answer":"a","answer":"b","count":1,"opt":null}`)))

	// Rejects invalid UTF-8.
	badUTF8 := append([]byte(`{"answer":"`), 0xff, '"', ',', '"', 'c', 'o', 'u', 'n', 't', '"', ':', '1', ',', '"', 'o', 'p', 't', '"', ':', 'n', 'u', 'l', 'l', '}')
	require.Error(t, ValidateJSON(compiled.Raw, badUTF8))
}
