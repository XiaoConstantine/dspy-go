package core_test

import (
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type questionInput struct {
	Question string `dspy:"question,required" description:"The question to answer"`
}

type answerOutput struct {
	Answer string `dspy:"answer" description:"A concise answer" prefix:"Answer:"`
}

func ExampleNewTypedSignature() {
	typed := core.NewTypedSignature[questionInput, answerOutput]().
		WithInstruction("Answer briefly.")
	signature := typed.ToLegacySignature()

	fmt.Println(signature.Instruction)
	fmt.Printf("%s -> %s\n", signature.Inputs[0].Name, signature.Outputs[0].Name)
	fmt.Println(signature.Outputs[0].Prefix)
	if err := typed.ValidateInput(questionInput{Question: "What is DSPy?"}); err == nil {
		fmt.Println("valid input")
	}

	// Output:
	// Answer briefly.
	// question -> answer
	// Answer:
	// valid input
}
