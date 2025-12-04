// Package core provides benchmarks for Go 1.25 experimental features.
// Run with: GOEXPERIMENT=jsonv2 go test -bench=. ./pkg/core/...
// Compare with: go test -bench=. ./pkg/core/...
package core

import (
	"encoding/json"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// BenchmarkJSONMarshal benchmarks JSON marshaling performance.
// Compare results with and without GOEXPERIMENT=jsonv2 to measure improvement.
//
// Run comparison:
//
//	go test -bench=BenchmarkJSONMarshal -benchmem ./pkg/core/...
//	GOEXPERIMENT=jsonv2 go test -bench=BenchmarkJSONMarshal -benchmem ./pkg/core/...
func BenchmarkJSONMarshal(b *testing.B) {
	b.Run("Signature", func(b *testing.B) {
		sig := Signature{
			Instruction: "A test signature for benchmarking JSON marshaling performance",
			Inputs: []InputField{
				{Field: Field{Name: "question", Description: "The question to answer"}},
				{Field: Field{Name: "context", Description: "Background context for the question"}},
			},
			Outputs: []OutputField{
				{Field: Field{Name: "answer", Description: "The generated answer"}},
				{Field: Field{Name: "confidence", Description: "Confidence score"}},
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := json.Marshal(sig)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Example", func(b *testing.B) {
		example := Example{
			Inputs: map[string]interface{}{
				"question": "What is the capital of France?",
				"context":  "France is a country in Western Europe. Its capital city is known for the Eiffel Tower.",
			},
			Outputs: map[string]interface{}{
				"answer":     "Paris",
				"confidence": 0.95,
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := json.Marshal(example)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("LargeDataset", func(b *testing.B) {
		// Simulate a larger dataset like optimizer training data
		examples := make([]Example, 100)
		for i := range examples {
			examples[i] = Example{
				Inputs: map[string]interface{}{
					"question": "What is the meaning of life, the universe, and everything?",
					"context":  "This is a complex philosophical question that has been pondered by humans for millennia.",
					"metadata": map[string]interface{}{
						"source": "benchmark",
						"index":  i,
					},
				},
				Outputs: map[string]interface{}{
					"answer":     "42",
					"confidence": 0.99,
					"reasoning":  "According to The Hitchhiker's Guide to the Galaxy by Douglas Adams.",
				},
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := json.Marshal(examples)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkJSONUnmarshal benchmarks JSON unmarshaling performance.
// Compare results with and without GOEXPERIMENT=jsonv2 to measure improvement.
func BenchmarkJSONUnmarshal(b *testing.B) {
	b.Run("Signature", func(b *testing.B) {
		data := []byte(`{
			"name": "TestSignature",
			"description": "A test signature for benchmarking",
			"inputs": [
				{"name": "question", "description": "The question"},
				{"name": "context", "description": "The context"}
			],
			"outputs": [
				{"name": "answer", "description": "The answer"},
				{"name": "confidence", "description": "Confidence score"}
			]
		}`)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var sig Signature
			if err := json.Unmarshal(data, &sig); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Example", func(b *testing.B) {
		data := []byte(`{
			"inputs": {
				"question": "What is the capital of France?",
				"context": "France is a country in Western Europe."
			},
			"outputs": {
				"answer": "Paris",
				"confidence": 0.95
			}
		}`)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var example Example
			if err := json.Unmarshal(data, &example); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("NestedStructure", func(b *testing.B) {
		// Simulate complex nested JSON like LLM responses
		data := []byte(`{
			"response": {
				"model": "gpt-4",
				"choices": [
					{
						"message": {
							"role": "assistant",
							"content": "The answer to your question is Paris."
						},
						"finish_reason": "stop",
						"index": 0
					}
				],
				"usage": {
					"prompt_tokens": 50,
					"completion_tokens": 20,
					"total_tokens": 70
				}
			},
			"metadata": {
				"latency_ms": 150,
				"cached": false
			}
		}`)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			if err := json.Unmarshal(data, &result); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkGCPressure benchmarks memory allocation patterns relevant to GreenteaGC.
// Run with GOEXPERIMENT=greenteagc to compare GC overhead.
//
// Run comparison:
//
//	go test -bench=BenchmarkGCPressure -benchmem ./pkg/core/...
//	GOEXPERIMENT=greenteagc go test -bench=BenchmarkGCPressure -benchmem ./pkg/core/...
func BenchmarkGCPressure(b *testing.B) {
	b.Run("ExampleAllocation", func(b *testing.B) {
		// Simulates optimizer training loop allocations
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			examples := make([]Example, 50)
			for j := range examples {
				examples[j] = Example{
					Inputs: map[string]interface{}{
						"input": "test input data for allocation benchmark",
					},
					Outputs: map[string]interface{}{
						"output": "test output data",
					},
				}
			}
			runtime.KeepAlive(examples)
		}
	})

	b.Run("MapAllocation", func(b *testing.B) {
		// Simulates frequent map allocations in module processing
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			results := make(map[string]interface{})
			for j := 0; j < 100; j++ {
				results["key"+strconv.Itoa(j)] = map[string]interface{}{
					"value":  j,
					"nested": []int{j, j + 1, j + 2},
				}
			}
			runtime.KeepAlive(results)
		}
	})

	b.Run("SliceGrowth", func(b *testing.B) {
		// Simulates slice growth patterns in batch processing
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var results []map[string]interface{}
			for j := 0; j < 100; j++ {
				results = append(results, map[string]interface{}{
					"index":  j,
					"status": "processed",
				})
			}
			runtime.KeepAlive(results)
		}
	})

	b.Run("StringConcatenation", func(b *testing.B) {
		// Simulates prompt building allocations using strings.Builder
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var sb strings.Builder
			for j := 0; j < 50; j++ {
				sb.WriteString("This is line ")
				sb.WriteRune(rune('0' + j%10))
				sb.WriteString(" of the prompt.\n")
			}
			prompt := sb.String()
			runtime.KeepAlive(prompt)
		}
	})
}

// BenchmarkMixedWorkload simulates realistic dspy-go workload patterns.
// This benchmark is useful for comparing overall performance with experimental features.
func BenchmarkMixedWorkload(b *testing.B) {
	b.Run("OptimizationIteration", func(b *testing.B) {
		// Simulates a single optimization iteration
		sig := Signature{
			Instruction: "Optimizer signature for benchmarking",
			Inputs: []InputField{
				{Field: Field{Name: "input"}},
			},
			Outputs: []OutputField{
				{Field: Field{Name: "output"}},
			},
		}

		examples := make([]Example, 20)
		for i := range examples {
			examples[i] = Example{
				Inputs:  map[string]interface{}{"input": "test"},
				Outputs: map[string]interface{}{"output": "result"},
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Marshal signature (config serialization)
			sigData, err := json.Marshal(sig)
			if err != nil {
				b.Fatal(err)
			}

			// Process examples
			results := make([]map[string]interface{}, len(examples))
			for j, ex := range examples {
				// Simulate processing
				exData, err := json.Marshal(ex)
				if err != nil {
					b.Fatal(err)
				}
				var processed Example
				if err := json.Unmarshal(exData, &processed); err != nil {
					b.Fatal(err)
				}
				results[j] = processed.Outputs
			}

			// Marshal results
			resultsData, err := json.Marshal(results)
			if err != nil {
				b.Fatal(err)
			}

			runtime.KeepAlive(sigData)
			runtime.KeepAlive(resultsData)
		}
	})
}
