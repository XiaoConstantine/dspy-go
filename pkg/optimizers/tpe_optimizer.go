package optimizers

import (
	"context"
	"fmt"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
	"math"
	"math/rand"
	"sort"
	"time"
)

// TPEConfig contains configuration for Tree-structured Parzen Estimators.
type TPEConfig struct {
	// Gamma is the percentile split between good and bad observations (default: 0.25)
	Gamma float64
	// Seed is used for random number generation
	Seed int64
	// NumEIGenerations is the number of random points to evaluate EI on
	NumEIGenerations int
	// Prior distributions for each parameter (optional)
	PriorWeight float64
	// Kernel bandwidth factor
	BandwidthFactor float64
}

// TPEOptimizer implements the Tree-structured Parzen Estimator for Bayesian optimization.
type TPEOptimizer struct {
	// Configuration
	gamma            float64
	seed             int64
	numEIGenerations int
	priorWeight      float64
	bandwidthFactor  float64

	// Internal state
	rng           *rand.Rand
	paramSpace    map[string][]interface{}
	observations  []observation
	bestParams    map[string]interface{}
	bestScore     float64
	maxTrials     int
	currentTrials int
}

// observation represents a single observation of parameters and resulting score.
type observation struct {
	params map[string]interface{}
	score  float64
}

// NewTPEOptimizer creates a new TPE optimizer instance.
func NewTPEOptimizer(config TPEConfig) SearchStrategy {
	// Set default values if not provided
	if config.Gamma <= 0 || config.Gamma >= 1 {
		config.Gamma = 0.25 // Default gamma
	}

	if config.NumEIGenerations <= 0 {
		config.NumEIGenerations = 24 // Default number of EI generations
	}

	if config.PriorWeight <= 0 {
		config.PriorWeight = 1.0 // Default prior weight
	}

	if config.BandwidthFactor <= 0 {
		config.BandwidthFactor = 1.0 // Default bandwidth factor
	}

	// Create RNG with seed
	var seed int64
	if config.Seed <= 0 {
		seed = time.Now().UnixNano()
	} else {
		seed = config.Seed
	}

	return &TPEOptimizer{
		gamma:            config.Gamma,
		seed:             seed,
		numEIGenerations: config.NumEIGenerations,
		priorWeight:      config.PriorWeight,
		bandwidthFactor:  config.BandwidthFactor,
		rng:              rand.New(rand.NewSource(seed)),
		observations:     make([]observation, 0),
		bestParams:       make(map[string]interface{}),
		bestScore:        -math.MaxFloat64,
	}
}

// Initialize sets up the search space and constraints.
func (t *TPEOptimizer) Initialize(config SearchConfig) error {
	if len(config.ParamSpace) == 0 {
		return fmt.Errorf("parameter space cannot be empty")
	}

	t.paramSpace = config.ParamSpace
	t.maxTrials = config.MaxTrials
	t.currentTrials = 0

	return nil
}

// SuggestParams suggests the next set of parameters to try.
func (t *TPEOptimizer) SuggestParams(ctx context.Context) (map[string]interface{}, error) {
	t.currentTrials++

	// If we don't have enough observations yet, use random sampling
	if len(t.observations) < max(5, int(float64(t.maxTrials)*0.1)) {
		return t.randomSample(), nil
	}

	// Otherwise use TPE to suggest parameters
	return t.suggestTPE(), nil
}

// UpdateResults updates the internal state with the results of the last trial.
func (t *TPEOptimizer) UpdateResults(params map[string]interface{}, score float64) error {
	t.observations = append(t.observations, observation{
		params: params,
		score:  score,
	})

	// Update best parameters if score is better
	if score > t.bestScore {
		t.bestScore = score
		t.bestParams = cloneParams(params)
	}

	return nil
}

// GetBestParams returns the best parameters found so far and their score.
func (t *TPEOptimizer) GetBestParams() (map[string]interface{}, float64) {
	return t.bestParams, t.bestScore
}

// randomSample generates a random set of parameters.
func (t *TPEOptimizer) randomSample() map[string]interface{} {
	params := make(map[string]interface{})

	for param, values := range t.paramSpace {
		if len(values) > 0 {
			// For now, handle numerical (float64) and categorical (any type) parameters
			idx := t.rng.Intn(len(values))
			params[param] = values[idx]
		}
	}

	return params
}

// suggestTPE generates parameters using the TPE algorithm.
func (t *TPEOptimizer) suggestTPE() map[string]interface{} {
	// Sort observations by score
	sort.Slice(t.observations, func(i, j int) bool {
		return t.observations[i].score > t.observations[j].score
	})

	// Determine the split point between good and bad observations
	nGood := max(1, int(float64(len(t.observations))*t.gamma))

	// Create lists of good and bad parameters
	goodParams := make([]map[string]interface{}, nGood)
	badParams := make([]map[string]interface{}, len(t.observations)-nGood)

	for i := 0; i < nGood; i++ {
		goodParams[i] = t.observations[i].params
	}

	for i := nGood; i < len(t.observations); i++ {
		badParams[i-nGood] = t.observations[i].params
	}

	// Generate candidates and compute EI for each one
	bestEI := -math.MaxFloat64
	bestParams := t.randomSample() // Fallback to random if something goes wrong

	// Generate candidate points and evaluate expected improvement
	for i := 0; i < t.numEIGenerations; i++ {
		candidate := t.generateTPECandidate(goodParams, badParams)
		ei := t.expectedImprovement(candidate, goodParams, badParams)

		if ei > bestEI {
			bestEI = ei
			bestParams = candidate
		}
	}

	return bestParams
}

// generateTPECandidate generates a candidate using the TPE approach.
func (t *TPEOptimizer) generateTPECandidate(
	goodParams []map[string]interface{},
	badParams []map[string]interface{},
) map[string]interface{} {
	candidate := make(map[string]interface{})

	// For each parameter, sample from the KDE of the good parameters
	for param, values := range t.paramSpace {
		// Handle categorical parameters (assuming all parameters are categorical in our implementation)
		// Get value counts for the parameter in good and bad sets
		goodCounts := t.countValues(goodParams, param, values)
		badCounts := t.countValues(badParams, param, values)

		// Apply smoothing and prior weight
		smoothedGoodCounts := t.smoothCounts(goodCounts, len(values))
		smoothedBadCounts := t.smoothCounts(badCounts, len(values))

		// Compute ratio of densities: p(x|y<γ) / p(x|y≥γ)
		ratios := make([]float64, len(values))
		for i := range values {
			if smoothedBadCounts[i] > 0 {
				ratios[i] = smoothedGoodCounts[i] / smoothedBadCounts[i]
			} else {
				ratios[i] = smoothedGoodCounts[i] * 1000 // Large value if not in bad set
			}
		}

		// Sample based on the ratios
		totalRatio := 0.0
		for _, r := range ratios {
			totalRatio += r
		}

		if totalRatio <= 0 {
			// If all ratios are zero, sample randomly
			idx := t.rng.Intn(len(values))
			candidate[param] = values[idx]
		} else {
			// Otherwise, sample based on the ratios
			threshold := t.rng.Float64() * totalRatio
			cumulative := 0.0
			for i, r := range ratios {
				cumulative += r
				if cumulative >= threshold {
					candidate[param] = values[i]
					break
				}
			}
		}
	}

	return candidate
}

// countValues counts the occurrences of each value for a given parameter.
func (t *TPEOptimizer) countValues(
	params []map[string]interface{},
	param string,
	possibleValues []interface{},
) []float64 {
	counts := make([]float64, len(possibleValues))

	for _, p := range params {
		if val, ok := p[param]; ok {
			for i, possVal := range possibleValues {
				if val == possVal {
					counts[i]++
					break
				}
			}
		}
	}

	return counts
}

// smoothCounts applies Laplace smoothing to the counts.
func (t *TPEOptimizer) smoothCounts(counts []float64, numValues int) []float64 {
	smoothed := make([]float64, len(counts))
	total := 0.0

	for _, c := range counts {
		total += c
	}

	// Apply Laplace smoothing and prior weight
	alpha := t.priorWeight
	for i, c := range counts {
		smoothed[i] = (c + alpha/float64(numValues)) / (total + alpha)
	}

	return smoothed
}

// expectedImprovement computes the expected improvement for a candidate.
func (t *TPEOptimizer) expectedImprovement(
	candidate map[string]interface{},
	goodParams []map[string]interface{},
	badParams []map[string]interface{},
) float64 {
	// Compute the likelihood under the good and bad distributions
	goodLikelihood := t.computeLikelihood(candidate, goodParams)
	badLikelihood := t.computeLikelihood(candidate, badParams)

	// Avoid division by zero
	if badLikelihood <= 0 {
		return goodLikelihood * 1000 // Large value if not in bad set
	}

	// Return the ratio of likelihoods
	return goodLikelihood / badLikelihood
}

// computeLikelihood computes the likelihood of a candidate under a set of parameters.
func (t *TPEOptimizer) computeLikelihood(
	candidate map[string]interface{},
	params []map[string]interface{},
) float64 {
	if len(params) == 0 {
		return 0
	}

	// For categorical parameters, we use a simple frequency-based approach
	likelihood := 1.0

	for param, candidateVal := range candidate {
		// Count how many times this value appears in the params
		count := 0
		for _, p := range params {
			if val, ok := p[param]; ok && val == candidateVal {
				count++
			}
		}

		// Compute probability with smoothing
		prob := (float64(count) + t.priorWeight/float64(len(t.paramSpace[param]))) /
			(float64(len(params)) + t.priorWeight)

		likelihood *= prob
	}

	return likelihood
}

func max(a, b int) int {
	return utils.Max(a, b)
}

// cloneParams delegates to the utils.cloneParams function.
func cloneParams(params map[string]interface{}) map[string]interface{} {
	return utils.CloneParams(params)
}
