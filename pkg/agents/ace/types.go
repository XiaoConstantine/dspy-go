// Package ace implements Agentic Context Engineering for self-improving agents.
// Based on the ACE paper (arXiv:2510.04618).
package ace

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Learning represents a single learned strategy or pattern.
type Learning struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Content  string `json:"content"`
	Helpful  int    `json:"helpful"`
	Harmful  int    `json:"harmful"`
}

// SuccessRate returns the ratio of helpful to total uses.
func (l *Learning) SuccessRate() float64 {
	total := l.Helpful + l.Harmful
	if total == 0 {
		return 0.5
	}
	return float64(l.Helpful) / float64(total)
}

// TotalUses returns the total number of times this learning has been evaluated.
func (l *Learning) TotalUses() int {
	return l.Helpful + l.Harmful
}

// ShouldPrune returns true if this learning should be removed.
func (l *Learning) ShouldPrune(minRatio float64, minUsage int) bool {
	if l.TotalUses() < minUsage {
		return false
	}
	return l.SuccessRate() < minRatio
}

// String formats the learning for file storage.
func (l *Learning) String() string {
	return fmt.Sprintf("[%s] helpful=%d harmful=%d :: %s", l.ID, l.Helpful, l.Harmful, l.Content)
}

// ShortCode returns a compact reference ID for context injection.
func (l *Learning) ShortCode() string {
	prefix := "L"
	switch l.Category {
	case "mistakes":
		prefix = "M"
	case "patterns":
		prefix = "P"
	}
	// Parse number from ID more robustly using strings.LastIndex
	var num int
	if i := strings.LastIndex(l.ID, "-"); i != -1 && i < len(l.ID)-1 {
		num, _ = strconv.Atoi(l.ID[i+1:])
	}
	return fmt.Sprintf("%s%03d", prefix, num)
}

// Outcome represents the result of a task execution.
type Outcome string

const (
	OutcomeSuccess Outcome = "success"
	OutcomePartial Outcome = "partial"
	OutcomeFailure Outcome = "failure"
)

// Trajectory captures a complete task execution path.
type Trajectory struct {
	ID           string            `json:"id"`
	AgentID      string            `json:"agent_id"`
	TaskType     string            `json:"task_type"`
	Query        string            `json:"query"`
	Steps        []Step            `json:"steps"`
	FinalOutcome Outcome           `json:"outcome"`
	Quality      float64           `json:"quality"`
	Duration     time.Duration     `json:"duration"`
	StartedAt    time.Time         `json:"started_at"`
	CompletedAt  time.Time         `json:"completed_at"`
	Context      TrajectoryContext `json:"context"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
}

// TrajectoryContext tracks which learnings were available and used.
type TrajectoryContext struct {
	InjectedLearnings []string `json:"injected_learnings,omitempty"`
	CitedLearnings    []string `json:"cited_learnings,omitempty"`
	AppliedLearnings  []string `json:"applied_learnings,omitempty"`
}

// Step represents a single action in a trajectory.
type Step struct {
	Index     int            `json:"index"`
	Action    string         `json:"action"`
	Tool      string         `json:"tool,omitempty"`
	Reasoning string         `json:"reasoning"`
	Input     map[string]any `json:"input,omitempty"`
	Output    map[string]any `json:"output,omitempty"`
	Quality   float64        `json:"quality"`
	Duration  time.Duration  `json:"duration"`
	Timestamp time.Time      `json:"timestamp"`
	Error     string         `json:"error,omitempty"`
}

// IsSuccessful returns true if the step completed without error.
func (s *Step) IsSuccessful() bool {
	return s.Error == ""
}

// InsightCandidate represents a potential new learning from reflection.
type InsightCandidate struct {
	Content    string  `json:"content"`
	Category   string  `json:"category"`
	Confidence float64 `json:"confidence"`
	Evidence   []int   `json:"evidence"`
	Source     string  `json:"source"`
}

// LearningUpdate represents feedback to apply to an existing learning.
type LearningUpdate struct {
	LearningID string `json:"learning_id"`
	Delta      Delta  `json:"delta"`
	Reason     string `json:"reason"`
}

// Delta represents the type of feedback to apply.
type Delta string

const (
	DeltaHelpful Delta = "helpful"
	DeltaHarmful Delta = "harmful"
)

// ReflectionResult contains extracted insights from analyzing a trajectory.
type ReflectionResult struct {
	TrajectoryID    string             `json:"trajectory_id"`
	SuccessPatterns []InsightCandidate `json:"success_patterns"`
	FailurePatterns []InsightCandidate `json:"failure_patterns"`
	LearningUpdates []LearningUpdate   `json:"learning_updates"`
	ProcessedAt     time.Time          `json:"processed_at"`
}

// CurationResult describes changes made to the learnings file.
type CurationResult struct {
	FilePath     string     `json:"file_path"`
	Added        []Learning `json:"added"`
	Updated      []Learning `json:"updated"`
	Merged       []string   `json:"merged"`
	Pruned       []string   `json:"pruned"`
	TokensBefore int        `json:"tokens_before"`
	TokensAfter  int        `json:"tokens_after"`
	ProcessedAt  time.Time  `json:"processed_at"`
}

// Config configures the ACE manager.
type Config struct {
	Enabled             bool    `json:"enabled"`
	LearningsPath       string  `json:"learnings_path"`
	AsyncReflection     bool    `json:"async_reflection"`
	CurationFrequency   int     `json:"curation_frequency"`
	MinConfidence       float64 `json:"min_confidence"`
	MaxTokens           int     `json:"max_tokens"`
	PruneMinRatio       float64 `json:"prune_min_ratio"`
	PruneMinUsage       int     `json:"prune_min_usage"`
	SimilarityThreshold float64 `json:"similarity_threshold"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Enabled:             false,
		LearningsPath:       ".learnings/default.md",
		AsyncReflection:     true,
		CurationFrequency:   10,
		MinConfidence:       0.7,
		MaxTokens:           80000,
		PruneMinRatio:       0.3,
		PruneMinUsage:       5,
		SimilarityThreshold: 0.85,
	}
}

// Validate checks that the config has valid values.
func (c *Config) Validate() error {
	if c.LearningsPath == "" {
		return fmt.Errorf("learnings_path cannot be empty")
	}
	if c.MinConfidence < 0 || c.MinConfidence > 1 {
		return fmt.Errorf("min_confidence must be between 0 and 1")
	}
	if c.PruneMinRatio < 0 || c.PruneMinRatio > 1 {
		return fmt.Errorf("prune_min_ratio must be between 0 and 1")
	}
	if c.SimilarityThreshold < 0 || c.SimilarityThreshold > 1 {
		return fmt.Errorf("similarity_threshold must be between 0 and 1")
	}
	if c.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be positive")
	}
	if c.CurationFrequency <= 0 {
		return fmt.Errorf("curation_frequency must be positive")
	}
	return nil
}
