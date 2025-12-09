package ace

// QualityWeights configures the relative importance of quality signals.
type QualityWeights struct {
	Outcome    float64
	Efficiency float64
	ToolSuccess float64
	ErrorFree  float64
}

// DefaultWeights returns balanced quality weights.
func DefaultWeights() QualityWeights {
	return QualityWeights{
		Outcome:    0.4,
		Efficiency: 0.2,
		ToolSuccess: 0.2,
		ErrorFree:  0.2,
	}
}

// QualityCalculator computes trajectory quality from multiple signals.
type QualityCalculator struct {
	weights           QualityWeights
	expectedSteps     int
	maxReasonableSteps int
}

// NewQualityCalculator creates a calculator with default parameters.
func NewQualityCalculator() *QualityCalculator {
	return &QualityCalculator{
		weights:           DefaultWeights(),
		expectedSteps:     5,
		maxReasonableSteps: 15,
	}
}

// WithWeights sets custom quality weights.
func (qc *QualityCalculator) WithWeights(w QualityWeights) *QualityCalculator {
	qc.weights = w
	return qc
}

// WithExpectedSteps sets the baseline for efficiency scoring.
func (qc *QualityCalculator) WithExpectedSteps(expected, max int) *QualityCalculator {
	qc.expectedSteps = expected
	qc.maxReasonableSteps = max
	return qc
}

// Calculate computes quality score for a trajectory.
func (qc *QualityCalculator) Calculate(t *Trajectory) float64 {
	if t == nil {
		return 0
	}

	outcomeScore := qc.outcomeScore(t.FinalOutcome)
	efficiencyScore := qc.efficiencyScore(len(t.Steps))
	toolSuccessScore := qc.toolSuccessScore(t.Steps)
	errorFreeScore := qc.errorFreeScore(t.Steps)

	return outcomeScore*qc.weights.Outcome +
		efficiencyScore*qc.weights.Efficiency +
		toolSuccessScore*qc.weights.ToolSuccess +
		errorFreeScore*qc.weights.ErrorFree
}

func (qc *QualityCalculator) outcomeScore(outcome Outcome) float64 {
	switch outcome {
	case OutcomeSuccess:
		return 1.0
	case OutcomePartial:
		return 0.5
	default:
		return 0.0
	}
}

func (qc *QualityCalculator) efficiencyScore(stepCount int) float64 {
	if stepCount == 0 {
		return 0
	}
	if stepCount <= qc.expectedSteps {
		return 1.0
	}
	if stepCount >= qc.maxReasonableSteps {
		return 0.2
	}
	// Linear interpolation between expected and max
	ratio := float64(stepCount-qc.expectedSteps) / float64(qc.maxReasonableSteps-qc.expectedSteps)
	return 1.0 - 0.8*ratio
}

func (qc *QualityCalculator) toolSuccessScore(steps []Step) float64 {
	toolSteps := 0
	successfulTools := 0

	for _, step := range steps {
		if step.Tool != "" {
			toolSteps++
			if step.Error == "" {
				successfulTools++
			}
		}
	}

	if toolSteps == 0 {
		return 1.0
	}
	return float64(successfulTools) / float64(toolSteps)
}

func (qc *QualityCalculator) errorFreeScore(steps []Step) float64 {
	for _, step := range steps {
		if step.Error != "" {
			return 0.0
		}
	}
	return 1.0
}
