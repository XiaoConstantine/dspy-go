package ace

import (
	"regexp"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Generator records trajectories during agent execution.
type Generator struct {
	trajectory *Trajectory
	mu         sync.Mutex
}

// NewGenerator creates a new trajectory generator.
func NewGenerator() *Generator {
	return &Generator{}
}

// Start begins recording a new trajectory.
func (g *Generator) Start(agentID, taskType, query string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.trajectory = &Trajectory{
		ID:        uuid.New().String(),
		AgentID:   agentID,
		TaskType:  taskType,
		Query:     query,
		Steps:     make([]Step, 0),
		StartedAt: time.Now(),
		Context:   TrajectoryContext{},
		Metadata:  make(map[string]any),
	}
}

// SetInjectedLearnings records which learnings were available in context.
func (g *Generator) SetInjectedLearnings(learningIDs []string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.trajectory != nil {
		g.trajectory.Context.InjectedLearnings = learningIDs
	}
}

// RecordStep captures a single action in the trajectory.
func (g *Generator) RecordStep(action, tool, reasoning string, input, output map[string]any, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.trajectory == nil {
		return
	}

	step := Step{
		Index:     len(g.trajectory.Steps),
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
		g.trajectory.Context.CitedLearnings = appendUnique(g.trajectory.Context.CitedLearnings, cited...)
	}

	g.trajectory.Steps = append(g.trajectory.Steps, step)
}

// End finalizes the trajectory with outcome and quality.
func (g *Generator) End(outcome Outcome, quality float64) *Trajectory {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.trajectory == nil {
		return nil
	}

	g.trajectory.FinalOutcome = outcome
	g.trajectory.Quality = quality
	g.trajectory.CompletedAt = time.Now()
	g.trajectory.Duration = g.trajectory.CompletedAt.Sub(g.trajectory.StartedAt)

	result := g.trajectory
	g.trajectory = nil
	return result
}

// Current returns the in-progress trajectory (for inspection).
func (g *Generator) Current() *Trajectory {
	g.mu.Lock()
	defer g.mu.Unlock()

	return g.trajectory
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
