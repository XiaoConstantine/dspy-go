package core

import (
	"fmt"
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
			[]InputField{{Field: NewField("input", WithDescription("input desc"))}},
			[]OutputField{{Field: NewField("output", WithDescription("output desc"))}},
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

func TestSignatureAppendInput(t *testing.T) {
	t.Run("AppendInput basic functionality", func(t *testing.T) {
		// Create initial signature
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "original_input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)

		// Append a new input field
		newSig := sig.AppendInput("new_input", "New Input:", "A new input field")

		// Verify the original signature is unchanged
		assert.Len(t, sig.Inputs, 1)
		assert.Equal(t, "original_input", sig.Inputs[0].Name)

		// Verify the new signature has the appended field
		assert.Len(t, newSig.Inputs, 2)
		assert.Equal(t, "original_input", newSig.Inputs[0].Name)
		assert.Equal(t, "new_input", newSig.Inputs[1].Name)
		assert.Equal(t, "New Input:", newSig.Inputs[1].Prefix)
		assert.Equal(t, "A new input field", newSig.Inputs[1].Description)

		// Verify outputs are unchanged
		assert.Equal(t, sig.Outputs, newSig.Outputs)
	})

	t.Run("AppendInput multiple times", func(t *testing.T) {
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "input1"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)

		// Chain multiple AppendInput calls
		newSig := sig.
			AppendInput("input2", "Input2:", "Second input").
			AppendInput("input3", "Input3:", "Third input")

		assert.Len(t, newSig.Inputs, 3)
		assert.Equal(t, "input1", newSig.Inputs[0].Name)
		assert.Equal(t, "input2", newSig.Inputs[1].Name)
		assert.Equal(t, "input3", newSig.Inputs[2].Name)
		assert.Equal(t, "Input2:", newSig.Inputs[1].Prefix)
		assert.Equal(t, "Input3:", newSig.Inputs[2].Prefix)
	})

	t.Run("AppendInput with empty fields", func(t *testing.T) {
		sig := NewSignature(nil, nil)
		newSig := sig.AppendInput("first_input", "", "")

		assert.Len(t, newSig.Inputs, 1)
		assert.Equal(t, "first_input", newSig.Inputs[0].Name)
		assert.Empty(t, newSig.Inputs[0].Prefix)
		assert.Empty(t, newSig.Inputs[0].Description)
	})

	t.Run("AppendInput preserves instruction", func(t *testing.T) {
		sig := NewSignature(nil, nil).WithInstruction("Original instruction")
		newSig := sig.AppendInput("new_input", "prefix", "desc")

		assert.Equal(t, "Original instruction", newSig.Instruction)
	})
}

func TestSignaturePrependOutput(t *testing.T) {
	t.Run("PrependOutput basic functionality", func(t *testing.T) {
		// Create initial signature
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "original_output"}}},
		)

		// Prepend a new output field
		newSig := sig.PrependOutput("new_output", "New Output:", "A new output field")

		// Verify the original signature is unchanged
		assert.Len(t, sig.Outputs, 1)
		assert.Equal(t, "original_output", sig.Outputs[0].Name)

		// Verify the new signature has the prepended field
		assert.Len(t, newSig.Outputs, 2)
		assert.Equal(t, "new_output", newSig.Outputs[0].Name)
		assert.Equal(t, "original_output", newSig.Outputs[1].Name)
		assert.Equal(t, "New Output:", newSig.Outputs[0].Prefix)
		assert.Equal(t, "A new output field", newSig.Outputs[0].Description)

		// Verify inputs are unchanged
		assert.Equal(t, sig.Inputs, newSig.Inputs)
	})

	t.Run("PrependOutput multiple times", func(t *testing.T) {
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output1"}}},
		)

		// Chain multiple PrependOutput calls
		newSig := sig.
			PrependOutput("output2", "Output2:", "Second output").
			PrependOutput("output3", "Output3:", "Third output")

		assert.Len(t, newSig.Outputs, 3)
		assert.Equal(t, "output3", newSig.Outputs[0].Name) // Most recently prepended
		assert.Equal(t, "output2", newSig.Outputs[1].Name)
		assert.Equal(t, "output1", newSig.Outputs[2].Name) // Original
		assert.Equal(t, "Output3:", newSig.Outputs[0].Prefix)
		assert.Equal(t, "Output2:", newSig.Outputs[1].Prefix)
	})

	t.Run("PrependOutput with empty fields", func(t *testing.T) {
		sig := NewSignature(nil, nil)
		newSig := sig.PrependOutput("first_output", "", "")

		assert.Len(t, newSig.Outputs, 1)
		assert.Equal(t, "first_output", newSig.Outputs[0].Name)
		assert.Empty(t, newSig.Outputs[0].Prefix)
		assert.Empty(t, newSig.Outputs[0].Description)
	})

	t.Run("PrependOutput preserves instruction", func(t *testing.T) {
		sig := NewSignature(nil, nil).WithInstruction("Original instruction")
		newSig := sig.PrependOutput("new_output", "prefix", "desc")

		assert.Equal(t, "Original instruction", newSig.Instruction)
	})
}

func TestSignatureChaining(t *testing.T) {
	t.Run("Chain AppendInput and PrependOutput", func(t *testing.T) {
		sig := NewSignature(
			[]InputField{{Field: Field{Name: "input1"}}},
			[]OutputField{{Field: Field{Name: "output1"}}},
		).WithInstruction("Original instruction")

		// Chain operations
		newSig := sig.
			AppendInput("input2", "Input2:", "Second input").
			AppendInput("input3", "Input3:", "Third input").
			PrependOutput("output0", "Output0:", "Prepended output").
			PrependOutput("rationale", "Rationale:", "Reasoning output")

		// Verify inputs
		assert.Len(t, newSig.Inputs, 3)
		assert.Equal(t, "input1", newSig.Inputs[0].Name)
		assert.Equal(t, "input2", newSig.Inputs[1].Name)
		assert.Equal(t, "input3", newSig.Inputs[2].Name)

		// Verify outputs (prepended in reverse order)
		assert.Len(t, newSig.Outputs, 3)
		assert.Equal(t, "rationale", newSig.Outputs[0].Name)   // Last prepended
		assert.Equal(t, "output0", newSig.Outputs[1].Name)    // First prepended
		assert.Equal(t, "output1", newSig.Outputs[2].Name)    // Original

		// Verify instruction is preserved
		assert.Equal(t, "Original instruction", newSig.Instruction)
	})

	t.Run("MultiChainComparison use case", func(t *testing.T) {
		// Simulate the MultiChainComparison signature modification
		originalSig := NewSignature(
			[]InputField{{Field: Field{Name: "question"}}},
			[]OutputField{{Field: Field{Name: "answer"}}},
		).WithInstruction("Answer the question")

		M := 3
		modifiedSig := originalSig

		// Add M reasoning attempt inputs
		for i := 0; i < M; i++ {
			modifiedSig = modifiedSig.AppendInput(
				fmt.Sprintf("reasoning_attempt_%d", i+1),
				fmt.Sprintf("Student Attempt #%d:", i+1),
				"${reasoning attempt}",
			)
		}

		// Prepend rationale output
		modifiedSig = modifiedSig.PrependOutput(
			"rationale",
			"Accurate Reasoning: Thank you everyone. Let's now holistically",
			"${corrected reasoning}",
		)

		// Verify the transformation
		assert.Len(t, modifiedSig.Inputs, 4)  // 1 original + 3 reasoning attempts
		assert.Len(t, modifiedSig.Outputs, 2) // 1 prepended + 1 original

		// Check input order
		assert.Equal(t, "question", modifiedSig.Inputs[0].Name)
		assert.Equal(t, "reasoning_attempt_1", modifiedSig.Inputs[1].Name)
		assert.Equal(t, "reasoning_attempt_2", modifiedSig.Inputs[2].Name)
		assert.Equal(t, "reasoning_attempt_3", modifiedSig.Inputs[3].Name)

		// Check output order (rationale first)
		assert.Equal(t, "rationale", modifiedSig.Outputs[0].Name)
		assert.Equal(t, "answer", modifiedSig.Outputs[1].Name)

		// Check prefixes
		assert.Equal(t, "Student Attempt #1:", modifiedSig.Inputs[1].Prefix)
		assert.Equal(t, "Accurate Reasoning: Thank you everyone. Let's now holistically", modifiedSig.Outputs[0].Prefix)

		// Verify instruction is preserved
		assert.Equal(t, "Answer the question", modifiedSig.Instruction)
	})
}

func TestSignatureImmutability(t *testing.T) {
	t.Run("Original signature unchanged after AppendInput", func(t *testing.T) {
		original := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)

		// Store original state
		originalInputLen := len(original.Inputs)
		originalInputName := original.Inputs[0].Name

		// Modify and verify original is unchanged
		_ = original.AppendInput("new_input", "prefix", "desc")

		assert.Len(t, original.Inputs, originalInputLen)
		assert.Equal(t, originalInputName, original.Inputs[0].Name)
	})

	t.Run("Original signature unchanged after PrependOutput", func(t *testing.T) {
		original := NewSignature(
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)

		// Store original state
		originalOutputLen := len(original.Outputs)
		originalOutputName := original.Outputs[0].Name

		// Modify and verify original is unchanged
		_ = original.PrependOutput("new_output", "prefix", "desc")

		assert.Len(t, original.Outputs, originalOutputLen)
		assert.Equal(t, originalOutputName, original.Outputs[0].Name)
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
