package modules

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// RewardFunction represents a function that evaluates the quality of a prediction.
// It takes the inputs used and the outputs produced, and returns a reward score.
// Higher scores indicate better predictions.
type RewardFunction func(inputs map[string]interface{}, outputs map[string]interface{}) float64

// RefineConfig holds configuration options for the Refine module.
type RefineConfig struct {
	// Number of refinement attempts
	N int
	// Reward function to evaluate predictions
	RewardFn RewardFunction
	// Minimum threshold for acceptable predictions
	Threshold float64
	// Number of failed attempts before giving up (optional)
	FailCount *int
}

// Refine implements a refinement module that runs predictions multiple times
// with varying temperatures to improve quality based on a reward function.
type Refine struct {
	core.BaseModule
	module core.Module
	config RefineConfig
	rng    *rand.Rand
}

// Ensure Refine implements required interfaces.
var _ core.Module = (*Refine)(nil)

// NewRefine creates a new Refine module with the specified configuration.
func NewRefine(module core.Module, config RefineConfig) *Refine {
	if config.N <= 0 {
		config.N = 3 // Default to 3 attempts
	}
	if config.Threshold == 0 {
		config.Threshold = 0.7 // Default threshold
	}

	baseModule := core.NewModule(module.GetSignature())
	baseModule.ModuleType = "Refine"
	baseModule.DisplayName = "" // Will be set by user or derived from context

	return &Refine{
		BaseModule: *baseModule,
		module:     module,
		config:     config,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// WithName sets a semantic name for this Refine instance.
func (r *Refine) WithName(name string) *Refine {
	r.DisplayName = name
	return r
}

// Process executes the refinement logic by running the module multiple times
// with different temperatures and selecting the best result based on the reward function.
func (r *Refine) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	logger := logging.GetLogger()

	// Use semantic name if set, otherwise fall back to operation name
	displayName := r.GetDisplayName()
	if displayName == "" || displayName == "BaseModule" {
		displayName = "Refine"
	}

	metadata := map[string]interface{}{
		"module_type":   r.GetModuleType(),
		"module_config": r.GetSignature().String(),
	}
	ctx, span := core.StartSpanWithContext(ctx, "Refine", displayName, metadata)
	defer core.EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)
	span.WithAnnotation("refine_attempts", r.config.N)
	span.WithAnnotation("reward_threshold", r.config.Threshold)

	// Generate temperature sequence for refinement attempts
	temperatures := r.generateTemperatureSequence()

	var bestOutputs map[string]interface{}
	var bestReward = -math.Inf(1) // Start with negative infinity
	var bestError error
	attemptCount := 0

	logger.Debug(ctx, "Starting refinement with %d attempts, threshold: %.2f", r.config.N, r.config.Threshold)

	// Try up to N refinement attempts
	for i := 0; i < r.config.N && i < len(temperatures); i++ {
		attemptCount++
		temperature := temperatures[i]

		logger.Debug(ctx, "Refinement attempt %d/%d with temperature %.2f", i+1, r.config.N, temperature)

		// Configure options with current temperature
		attemptOpts := append(opts, core.WithGenerateOptions(core.WithTemperature(temperature)))

		// Run the module with current temperature
		outputs, err := r.module.Process(ctx, inputs, attemptOpts...)
		if err != nil {
			logger.Debug(ctx, "Attempt %d failed with error: %v", i+1, err)
			if bestError == nil {
				bestError = err // Keep first error if no successful attempts
			}
			continue
		}

		// Calculate reward for this prediction
		reward := r.config.RewardFn(inputs, outputs)
		logger.Debug(ctx, "Attempt %d reward: %.3f", i+1, reward)

		// Track the best prediction
		if reward > bestReward {
			bestReward = reward
			bestOutputs = outputs
			bestError = nil
			logger.Debug(ctx, "New best reward: %.3f (attempt %d)", reward, i+1)
		}

		// Early termination if threshold is met
		if reward >= r.config.Threshold {
			logger.Debug(ctx, "Threshold %.2f met with reward %.3f, stopping early", r.config.Threshold, reward)
			break
		}
	}

	// If no successful attempts, return the error
	if bestOutputs == nil {
		span.WithError(bestError)
		return nil, fmt.Errorf("all refinement attempts failed, last error: %w", bestError)
	}

	// Log final results
	logger.Debug(ctx, "Refinement completed: best reward %.3f from %d attempts", bestReward, attemptCount)
	span.WithAnnotation("best_reward", bestReward)
	span.WithAnnotation("attempts_made", attemptCount)
	span.WithAnnotation("outputs", bestOutputs)

	return bestOutputs, nil
}

// generateTemperatureSequence creates a sequence of temperatures for refinement attempts.
// This follows a similar strategy to the Python implementation.
func (r *Refine) generateTemperatureSequence() []float64 {
	if r.config.N == 1 {
		return []float64{0.7} // Single attempt with moderate temperature
	}

	temperatures := make([]float64, r.config.N)

	// First attempt: low temperature for consistency
	temperatures[0] = 0.3

	// If only 2 attempts, use low and high
	if r.config.N == 2 {
		temperatures[1] = 0.9
		return temperatures
	}

	// For more attempts, create a varied sequence
	// Second attempt: moderate temperature
	temperatures[1] = 0.7

	// Remaining attempts: varied temperatures with some randomness
	for i := 2; i < r.config.N; i++ {
		var base float64
		if r.config.N == 3 {
			// Special case for 3 attempts: third attempt gets high temperature
			base = 0.9
		} else {
			// For N > 3, spread temperatures across remaining range
			base = 0.5 + float64(i-2)*0.3/float64(r.config.N-3)
		}
		jitter := (r.rng.Float64() - 0.5) * 0.2 // ±0.1 jitter
		temp := base + jitter

		// Clamp to reasonable bounds
		if temp < 0.1 {
			temp = 0.1
		}
		if temp > 1.0 {
			temp = 1.0
		}

		temperatures[i] = temp
	}

	return temperatures
}

// GetSignature returns the module's signature.
func (r *Refine) GetSignature() core.Signature {
	return r.module.GetSignature()
}

// SetSignature updates the signature for both this module and the wrapped module.
func (r *Refine) SetSignature(signature core.Signature) {
	r.BaseModule.SetSignature(signature)
	r.module.SetSignature(signature)
}

// SetLLM sets the language model for the wrapped module.
func (r *Refine) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.module.SetLLM(llm)
}

// Clone creates a deep copy of the Refine module.
func (r *Refine) Clone() core.Module {
	return &Refine{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		module:     r.module.Clone(),
		config:     r.config, // Config is copied by value
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// WithDefaultOptions sets default options for the module.
func (r *Refine) WithDefaultOptions(opts ...core.Option) *Refine {
	// If the wrapped module supports default options, delegate to it
	if predictor, ok := r.module.(*Predict); ok {
		predictor.WithDefaultOptions(opts...)
	}
	return r
}

// GetWrappedModule returns the underlying module being refined.
func (r *Refine) GetWrappedModule() core.Module {
	return r.module
}

// UpdateConfig allows updating the refinement configuration.
func (r *Refine) UpdateConfig(config RefineConfig) *Refine {
	r.config = config
	return r
}

// GetConfig returns the current refinement configuration.
func (r *Refine) GetConfig() RefineConfig {
	return r.config
}

// OfferFeedback represents a module for generating advice to improve module performance.
// This is a simplified version of the Python implementation's OfferFeedback signature.
type OfferFeedback struct {
	core.BaseModule
	predict *Predict
}

// NewOfferFeedback creates a new OfferFeedback module.
func NewOfferFeedback() *OfferFeedback {
	// Create signature for feedback generation
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("program_inputs", core.WithDescription("The inputs to the program"))},
			{Field: core.NewField("program_outputs", core.WithDescription("The outputs from the program"))},
			{Field: core.NewField("reward_value", core.WithDescription("The reward score achieved"))},
			{Field: core.NewField("target_threshold", core.WithDescription("The target threshold to achieve"))},
		},
		[]core.OutputField{
			{Field: core.NewField("discussion", core.WithDescription("Analysis of the program's performance"))},
			{Field: core.NewField("advice", core.WithDescription("Specific advice for improvement"))},
		},
	).WithInstruction("Analyze the program's performance and provide specific advice for improvement.")

	return &OfferFeedback{
		BaseModule: *core.NewModule(signature),
		predict:    NewPredict(signature),
	}
}

// Process generates feedback and advice for improving module performance.
func (of *OfferFeedback) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	ctx, span := core.StartSpan(ctx, "OfferFeedback")
	defer core.EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)
	outputs, err := of.predict.Process(ctx, inputs, opts...)
	if err != nil {
		span.WithError(err)
	} else {
		span.WithAnnotation("outputs", outputs)
	}
	return outputs, err
}

// SetLLM sets the language model for feedback generation.
func (of *OfferFeedback) SetLLM(llm core.LLM) {
	of.BaseModule.SetLLM(llm)
	of.predict.SetLLM(llm)
}

// Clone creates a deep copy of the OfferFeedback module.
func (of *OfferFeedback) Clone() core.Module {
	return &OfferFeedback{
		BaseModule: *of.BaseModule.Clone().(*core.BaseModule),
		predict:    of.predict.Clone().(*Predict),
	}
}

// NewTypedRefine creates a new type-safe Refine module from a typed signature.
// Typed modules use text-based parsing by default since they typically rely on prefixes.
func NewTypedRefine[TInput, TOutput any](module core.Module, config RefineConfig) *Refine {
	// Note: We don't need to convert to legacy signature since NewRefine takes a Module directly

	refine := NewRefine(module, config)
	// Use clearer variable names for type display
	var i TInput
	var o TOutput
	refine.DisplayName = fmt.Sprintf("TypedRefine[%T,%T]", i, o)

	return refine
}
