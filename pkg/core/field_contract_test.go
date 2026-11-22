package core

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToLegacySignaturePreservesGoTypeAndRequired(t *testing.T) {
	type Out struct {
		Reasoning string  `dspy:"reasoning"`
		Answer    string  `dspy:"answer"`
		Tag       *string `dspy:"tag"`
		Count     int     `dspy:"count"`
		Label     string  `dspy:"label" enum:"yes,no"`
	}
	ts := NewTypedSignature[struct{ Question string }, Out]()
	sig := ts.ToLegacySignature()
	require.Len(t, sig.Outputs, 5)

	assert.Equal(t, "reasoning", sig.Outputs[0].Name)
	assert.True(t, sig.Outputs[0].IsRequired())
	assert.Equal(t, reflect.TypeOf(""), sig.Outputs[0].GoType)

	assert.Equal(t, "tag", sig.Outputs[2].Name)
	assert.False(t, sig.Outputs[2].IsRequired())
	assert.Equal(t, reflect.TypeOf((*string)(nil)), sig.Outputs[2].GoType)

	assert.Equal(t, "count", sig.Outputs[3].Name)
	assert.True(t, sig.Outputs[3].IsRequired())
	assert.Equal(t, reflect.TypeOf(0), sig.Outputs[3].GoType)

	assert.Equal(t, []string{"yes", "no"}, sig.Outputs[4].EnumValues())
}

func TestTypedRequiredTagAndAdapterOptionalAreIndependent(t *testing.T) {
	type Out struct {
		Plain    string  `dspy:"plain"`
		Pointer  *string `dspy:"pointer"`
		Tagged   string  `dspy:"tagged,optional"`
		Required string  `dspy:"required,required"`
	}
	ts := NewTypedSignature[struct{ Question string }, Out]()

	// Typed validation still rejects only a zero value in a field tagged required.
	require.NoError(t, ts.ValidateOutput(Out{Required: "set"}))
	require.ErrorContains(t, ts.ValidateOutput(Out{Plain: "set"}), "required output field 'required' cannot be empty")

	// Adapters read Optional: only pointer fields and fields tagged optional may be omitted.
	optional := map[string]bool{}
	for _, field := range ts.ToLegacySignature().Outputs {
		optional[field.Name] = field.Optional
	}
	assert.Equal(t, map[string]bool{"plain": false, "pointer": true, "tagged": true, "required": false}, optional)
}

func TestFieldsAndStreamChunksCompareByValue(t *testing.T) {
	newField := func(enum ...string) Field {
		return NewField("label", WithEnum(enum...), WithSchema(json.RawMessage(`{"type":"string"}`)))
	}
	assert.True(t, newField("yes", "no") == newField("yes", "no"))
	assert.False(t, newField("yes", "no") == newField("yes"))
	assert.True(t, InputField{Field: newField("yes")} == InputField{Field: newField("yes")})
	assert.True(t, OutputField{Field: newField("yes")} == OutputField{Field: newField("yes")})
	assert.True(t, FieldMetadata{Name: "a"} == FieldMetadata{Name: "a"})
	assert.True(t, StreamChunk{Content: "a", Field: "answer"} == StreamChunk{Content: "a", Field: "answer"})
}

type mockStateAdapterModule struct {
	BaseModule
	demos          []Example
	adapterName    string
	adapterVersion string
}

func (m *mockStateAdapterModule) GetDemos() []Example  { return m.demos }
func (m *mockStateAdapterModule) SetDemos(d []Example) { m.demos = d }
func (m *mockStateAdapterModule) GetAdapterInfo() (string, string) {
	return m.adapterName, m.adapterVersion
}
func (m *mockStateAdapterModule) SetAdapterByName(n, v string) error {
	m.adapterName, m.adapterVersion = n, v
	return nil
}

func TestSaveLoadProgramMediaEnvelopeAndAdapter(t *testing.T) {
	sig := NewSignature(
		[]InputField{{Field: NewImageField("img")}},
		[]OutputField{{Field: NewField("caption")}},
	).WithInstruction("Describe image.")

	imgBytes := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A}
	mod := &mockStateAdapterModule{
		BaseModule:     *NewModule(sig),
		adapterName:    "xml",
		adapterVersion: "v1",
		demos: []Example{{
			Inputs:  map[string]any{"img": NewImageBlock(imgBytes, "image/png")},
			Outputs: map[string]any{"caption": "a png"},
		}},
	}
	prog := NewProgram(map[string]Module{"cap": mod}, nil)
	path := filepath.Join(t.TempDir(), "state.json")
	require.NoError(t, SaveProgram(&prog, path))

	// ContentBlock.Data still keeps json:"-".
	rawCB, err := json.Marshal(NewImageBlock(imgBytes, "image/png"))
	require.NoError(t, err)
	assert.NotContains(t, string(rawCB), "iVBOR")

	// Loading restores media bytes and adapter identity without warnings.
	loadedMod := &mockStateAdapterModule{BaseModule: *NewModule(sig)}
	loadedProg := NewProgram(map[string]Module{"cap": loadedMod}, nil)
	var warnings []string
	require.NoError(t, LoadProgramWithOptions(&loadedProg, path, LoadProgramOptions{
		WarningHandler: func(msg string) { warnings = append(warnings, msg) },
	}))
	assert.Empty(t, warnings)
	assert.Equal(t, "xml", loadedMod.adapterName)
	assert.Equal(t, "v1", loadedMod.adapterVersion)
	require.Len(t, loadedMod.demos, 1)
	cb, ok := loadedMod.demos[0].Inputs["img"].(ContentBlock)
	require.True(t, ok)
	assert.Equal(t, imgBytes, cb.Data)
	assert.Equal(t, "image/png", cb.MimeType)

	// Media over the cap is saved as it always was: without its bytes.
	oversized := &mockStateAdapterModule{
		BaseModule: *NewModule(sig),
		demos: []Example{{
			Inputs:  map[string]any{"img": NewImageBlock(bytes.Repeat([]byte{1}, DefaultMaxSavedMediaBytes+1), "image/png")},
			Outputs: map[string]any{"caption": "huge"},
		}},
	}
	overProg := NewProgram(map[string]Module{"cap": oversized}, nil)
	overPath := filepath.Join(t.TempDir(), "over.json")
	require.NoError(t, SaveProgram(&overProg, overPath))
	saved, err := os.ReadFile(overPath)
	require.NoError(t, err)
	assert.NotContains(t, string(saved), savedMediaEnvelopeType)
	assert.Less(t, len(saved), 1024)
}
