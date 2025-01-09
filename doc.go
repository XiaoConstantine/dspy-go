// Package dspy is a Go implementation of the DSPy framework for using language models
// to solve complex tasks through composable steps and prompting techniques.
//
// DSPy-Go provides a collection of modules, optimizers, and tools for building
// reliable LLM-powered applications. It focuses on making it easy to:
//   - Break down complex tasks into modular steps
//   - Optimize prompts and chain-of-thought reasoning
//   - Handle common LLM interaction patterns
//   - Evaluate and improve system performance
//
// Key Components:
//   - Modules: Building blocks like Predict and ChainOfThought for composing LLM workflows
//   - Optimizers: Tools for improving prompt effectiveness like BootstrapFewShot
//   - Workflows: Pre-built patterns for common tasks like routing and parallel execution
//   - Metrics: Evaluation tools for measuring performance
//
// Simple Example:
//
//	signature := core.NewSignature(
//		[]core.InputField{{Field: core.Field{Name: "question"}}},
//		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
//	)
//
//	// Create a chain-of-thought module
//	cot := modules.NewChainOfThought(signature)
//
//	// Create a program using the module
//	program := core.NewProgram(
//		map[string]core.Module{"cot": cot},
//		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
//			return cot.Process(ctx, inputs)
//		},
//	)
//
// For more examples and detailed documentation, visit:
// https://github.com/XiaoConstantine/dspy-go
package dspy
