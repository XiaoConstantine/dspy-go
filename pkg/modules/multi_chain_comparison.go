package modules

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// MultiChainComparison implements the multi-chain comparison module
// that compares multiple reasoning attempts and produces a holistic evaluation.
type MultiChainComparison struct {
	core.BaseModule
	M               int             // Number of attempts to compare
	predict         *Predict        // Internal predict module
	lastKey         string          // Last key in the original signature
	defaultOptions  *core.ModuleOptions
}

// Ensure MultiChainComparison implements core.Module.
var _ core.Module = (*MultiChainComparison)(nil)

// NewMultiChainComparison creates a new MultiChainComparison module.
func NewMultiChainComparison(signature core.Signature, M int, temperature float64, opts ...core.Option) *MultiChainComparison {
	// Store the last key from the original signature
	var lastKey string
	if len(signature.Outputs) > 0 {
		lastKey = signature.Outputs[len(signature.Outputs)-1].Name
	}

	// Dynamically modify signature to include multiple reasoning attempts
	modifiedSignature := signature
	for idx := 0; idx < M; idx++ {
		modifiedSignature = modifiedSignature.AppendInput(
			fmt.Sprintf("reasoning_attempt_%d", idx+1),
			fmt.Sprintf("Student Attempt #%d:", idx+1),
			"${reasoning attempt}",
		)
	}

	// Prepend the rationale output field
	modifiedSignature = modifiedSignature.PrependOutput(
		"rationale",
		"Accurate Reasoning: Thank you everyone. Let's now holistically",
		"${corrected reasoning}",
	)

	// Create the predict module with modified signature
	predict := NewPredict(modifiedSignature)
	
	// Apply temperature and other options to the predict module
	options := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(options)
	}
	
	// Set temperature if provided
	if temperature > 0 {
		generateOpts := []core.GenerateOption{
			core.WithTemperature(temperature),
		}
		options.GenerateOptions = generateOpts
	}

	predict = predict.WithDefaultOptions(opts...)

	return &MultiChainComparison{
		BaseModule:     *core.NewModule(modifiedSignature),
		M:              M,
		predict:        predict,
		lastKey:        lastKey,
		defaultOptions: options,
	}
}

// Process implements the core.Module interface.
// It takes completions and processes them into reasoning attempts for comparison.
func (m *MultiChainComparison) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	
	// Extract completions from inputs
	completionsRaw, ok := inputs["completions"]
	if !ok {
		return nil, errors.WithFields(
			errors.New(errors.ValidationFailed, "completions not found in inputs"),
			errors.Fields{
				"module": "MultiChainComparison",
				"inputs": inputs,
			})
	}

	// Convert completions to the expected format
	completions, ok := completionsRaw.([]map[string]interface{})
	if !ok {
		return nil, errors.WithFields(
			errors.New(errors.ValidationFailed, "completions must be a slice of maps"),
			errors.Fields{
				"module": "MultiChainComparison",
				"type":   fmt.Sprintf("%T", completionsRaw),
			})
	}

	// Process completions into attempts
	attempts, err := m.processCompletions(completions)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to process completions"),
			errors.Fields{
				"module":      "MultiChainComparison",
				"completions": len(completions),
			})
	}

	// Validate that we have the expected number of attempts
	if len(attempts) != m.M {
		return nil, errors.WithFields(
			errors.New(errors.ValidationFailed, "number of attempts doesn't match expected M"),
			errors.Fields{
				"module":         "MultiChainComparison",
				"expected":       m.M,
				"actual":         len(attempts),
			})
	}

	logger.Debug(ctx, "MultiChainComparison processed %d attempts for comparison", len(attempts))

	// Add attempts to inputs for the predict module
	processedInputs := make(map[string]interface{})
	for key, value := range inputs {
		if key != "completions" { // Skip the completions key
			processedInputs[key] = value
		}
	}

	// Add reasoning attempts
	for idx, attempt := range attempts {
		processedInputs[fmt.Sprintf("reasoning_attempt_%d", idx+1)] = attempt
	}

	// Call the internal predict module
	return m.predict.Process(ctx, processedInputs, opts...)
}

// processCompletions converts completions into formatted reasoning attempts.
func (m *MultiChainComparison) processCompletions(completions []map[string]interface{}) ([]string, error) {
	attempts := make([]string, 0, len(completions))

	for _, completion := range completions {
		// Extract rationale from completion
		var rationale string
		if r, ok := completion["rationale"].(string); ok && r != "" {
			rationale = strings.Split(strings.TrimSpace(r), "\n")[0]
			rationale = strings.TrimSpace(rationale)
		} else if r, ok := completion["reasoning"].(string); ok && r != "" {
			rationale = strings.Split(strings.TrimSpace(r), "\n")[0]
			rationale = strings.TrimSpace(rationale)
		}

		// Extract answer from completion using the last key
		var answer string
		if m.lastKey != "" {
			if a, ok := completion[m.lastKey]; ok {
				answer = strings.Split(strings.TrimSpace(fmt.Sprintf("%v", a)), "\n")[0]
				answer = strings.TrimSpace(answer)
			}
		}

		// Format the attempt
		attempt := fmt.Sprintf("«I'm trying to %s I'm not sure but my prediction is %s»", rationale, answer)
		attempts = append(attempts, attempt)
	}

	return attempts, nil
}

// Clone creates a deep copy of the MultiChainComparison module.
func (m *MultiChainComparison) Clone() core.Module {
	clonedPredict := m.predict.Clone().(*Predict)
	return &MultiChainComparison{
		BaseModule:     *m.BaseModule.Clone().(*core.BaseModule),
		M:              m.M,
		predict:        clonedPredict,
		lastKey:        m.lastKey,
		defaultOptions: m.defaultOptions,
	}
}

// SetLLM sets the LLM for the internal predict module.
func (m *MultiChainComparison) SetLLM(llm core.LLM) {
	m.predict.SetLLM(llm)
}

// GetSignature returns the signature of the module.
func (m *MultiChainComparison) GetSignature() core.Signature {
	return m.BaseModule.GetSignature()
}