package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestField(t *testing.T) {
	t.Run("NewField with defaults", func(t *testing.T) {
		field := NewField("test")
		assert.Equal(t, "test", field.Name)
		assert.Equal(t, "test:", field.Prefix)
		assert.Empty(t, field.Description)
	})

	t.Run("NewField with options", func(t *testing.T) {
		field := NewField("test",
			WithDescription("test description"),
			WithCustomPrefix("custom:"),
		)
		assert.Equal(t, "test", field.Name)
		assert.Equal(t, "custom:", field.Prefix)
		assert.Equal(t, "test description", field.Description)
	})

	t.Run("NewField with no prefix", func(t *testing.T) {
		field := NewField("test", WithNoPrefix())
		assert.Equal(t, "test", field.Name)
		assert.Empty(t, field.Prefix)
	})
}
func TestSignature(t *testing.T) {
	t.Run("NewSignature", func(t *testing.T) {
		inputs := []InputField{
			{Field: Field{Name: "input1"}},
			{Field: Field{Name: "input2"}},
		}
		outputs := []OutputField{
			{Field: Field{Name: "output1"}},
			{Field: Field{Name: "output2"}},
		}

		sig := NewSignature(inputs, outputs)
		assert.Equal(t, inputs, sig.Inputs)
		assert.Equal(t, outputs, sig.Outputs)
		assert.Empty(t, sig.Instruction)
	})

	t.Run("WithInstruction", func(t *testing.T) {
		sig := NewSignature(nil, nil)
		sigWithInst := sig.WithInstruction("test instruction")
		assert.Equal(t, "test instruction", sigWithInst.Instruction)
	})

	t.Run("String representation", func(t *testing.T) {
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "input", Description: "input desc"}}},
			[]OutputField{{Field: Field{Name: "output", Description: "output desc"}}},
		).WithInstruction("test instruction")

		str := sig.String()
		assert.Contains(t, str, "Inputs:")
		assert.Contains(t, str, "input (input desc)")
		assert.Contains(t, str, "Outputs:")
		assert.Contains(t, str, "output (output desc)")
		assert.Contains(t, str, "Instruction: test instruction")
	})
}

func TestSignatureParser(t *testing.T) {
	t.Run("ParseSignature valid", func(t *testing.T) {
		signatureStr := "input1, input2 -> output1, output2"
		sig, err := ParseSignature(signatureStr)
		assert.NoError(t, err)
		assert.Len(t, sig.Inputs, 2)
		assert.Len(t, sig.Outputs, 2)
		assert.Equal(t, "input1", sig.Inputs[0].Name)
		assert.Equal(t, "output1", sig.Outputs[0].Name)
	})

	t.Run("ParseSignature invalid", func(t *testing.T) {
		invalidStr := "invalid signature format"
		_, err := ParseSignature(invalidStr)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid signature format")
	})

	t.Run("ShorthandNotation", func(t *testing.T) {
		sig, err := ShorthandNotation("in1, in2 -> out1, out2")
		assert.NoError(t, err)
		assert.Len(t, sig.Inputs, 2)
		assert.Len(t, sig.Outputs, 2)
	})

	t.Run("ParseSignature with whitespace", func(t *testing.T) {
		signatureStr := "  input1 ,  input2   ->   output1 , output2  "
		sig, err := ParseSignature(signatureStr)
		assert.NoError(t, err)
		assert.Len(t, sig.Inputs, 2)
		assert.Len(t, sig.Outputs, 2)
		assert.Equal(t, "input1", sig.Inputs[0].Name)
		assert.Equal(t, "output1", sig.Outputs[0].Name)
	})
}

func TestHelperFunctions(t *testing.T) {
	t.Run("parseInputFields", func(t *testing.T) {
		fields := parseInputFields("field1, field2")
		assert.Len(t, fields, 2)
		assert.Equal(t, "field1", fields[0].Name)
		assert.Equal(t, "field2", fields[1].Name)
	})

	t.Run("parseOutputFields", func(t *testing.T) {
		fields := parseOutputFields("field1, field2")
		assert.Len(t, fields, 2)
		assert.Equal(t, "field1", fields[0].Name)
		assert.Equal(t, "field2", fields[1].Name)
	})
}
