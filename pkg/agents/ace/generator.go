package ace

import (
	"regexp"
	"sync"
	"time"

	"github.com/google/uuid"
)

// TrajectoryRecorder is a handle for recording a single trajectory.
// It is concurrency-safe and isolates trajectory state from other concurrent executions.
type TrajectoryRecorder struct {
	trajectory *Trajectory
	mu         sync.Mutex
}

// RecordStep captures a single action in the trajectory.
func (r *TrajectoryRecorder) RecordStep(action, tool, reasoning string, input, output map[string]any, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.trajectory == nil {
		return
	}

	step := Step{
		Index:     len(r.trajectory.Steps),
		Action:    action,
		Tool:      tool,
		Reasoning: reasoning,
		Input:     input,
		Output:    output,
		Timestamp: time.Now(),
	}

	if err != nil {
		step.Error = err.Error()
	}

	// Detect citations in reasoning
	cited := DetectCitations(reasoning)
	if len(cited) > 0 {
		r.trajectory.Context.CitedLearnings = appendUnique(r.trajectory.Context.CitedLearnings, cited...)
	}

	r.trajectory.Steps = append(r.trajectory.Steps, step)
}

// SetInjectedLearnings records which learnings were available in context.
func (r *TrajectoryRecorder) SetInjectedLearnings(learningIDs []string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.trajectory != nil {
		r.trajectory.Context.InjectedLearnings = learningIDs
	}
}

// End finalizes the trajectory with outcome and quality.
func (r *TrajectoryRecorder) End(outcome Outcome, quality float64) *Trajectory {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.trajectory == nil {
		return nil
	}

	r.trajectory.FinalOutcome = outcome
	r.trajectory.Quality = quality
	r.trajectory.CompletedAt = time.Now()
	r.trajectory.Duration = r.trajectory.CompletedAt.Sub(r.trajectory.StartedAt)

	result := r.trajectory
	r.trajectory = nil
	return result
}

// Current returns the in-progress trajectory (for inspection).
func (r *TrajectoryRecorder) Current() *Trajectory {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.trajectory
}

// Generator creates trajectory recorders for agent executions.
// It is stateless regarding active trajectories, making it safe for concurrent use.
type Generator struct{}

// NewGenerator creates a new trajectory generator.
func NewGenerator() *Generator {
	return &Generator{}
}

// Start begins recording a new trajectory and returns a handle for it.
// Each call returns an independent TrajectoryRecorder, safe for concurrent use.
func (g *Generator) Start(agentID, taskType, query string) *TrajectoryRecorder {
	trajectory := &Trajectory{
		ID:        uuid.New().String(),
		AgentID:   agentID,
		TaskType:  taskType,
		Query:     query,
		Steps:     make([]Step, 0),
		StartedAt: time.Now(),
		Context:   TrajectoryContext{},
		Metadata:  make(map[string]any),
	}

	return &TrajectoryRecorder{
		trajectory: trajectory,
	}
}

var citationRegex = regexp.MustCompile(`\[([LMP]\d{3})\]`)

// DetectCitations finds learning references in text.
func DetectCitations(text string) []string {
	matches := citationRegex.FindAllStringSubmatch(text, -1)
	var citations []string
	seen := make(map[string]bool)

	for _, match := range matches {
		if len(match) > 1 && !seen[match[1]] {
			citations = append(citations, match[1])
			seen[match[1]] = true
		}
	}
	return citations
}

func appendUnique(slice []string, items ...string) []string {
	seen := make(map[string]bool)
	for _, s := range slice {
		seen[s] = true
	}
	for _, item := range items {
		if !seen[item] {
			slice = append(slice, item)
			seen[item] = true
		}
	}
	return slice
}
