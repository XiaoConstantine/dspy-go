package tools

import (
	"context"
	"math"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// BayesianToolSelector implements intelligent tool selection using Bayesian scoring.
type BayesianToolSelector struct {
	// Weights for different scoring components
	MatchWeight       float64 `json:"match_weight"`
	PerformanceWeight float64 `json:"performance_weight"`
	CapabilityWeight  float64 `json:"capability_weight"`

	// Prior probabilities for tool selection
	PriorProbabilities map[string]float64 `json:"prior_probabilities"`
}

// NewBayesianToolSelector creates a new Bayesian tool selector with default weights.
func NewBayesianToolSelector() *BayesianToolSelector {
	return &BayesianToolSelector{
		MatchWeight:        0.4,
		PerformanceWeight:  0.35,
		CapabilityWeight:   0.25,
		PriorProbabilities: make(map[string]float64),
	}
}

// SelectBest selects the best tool from scored candidates.
func (s *BayesianToolSelector) SelectBest(ctx context.Context, intent string, candidates []ToolScore) (core.Tool, error) {
	if len(candidates) == 0 {
		return nil, errors.New(errors.ResourceNotFound, "no tool candidates provided")
	}

	// Find the candidate with the highest final score
	bestCandidate := candidates[0]
	for _, candidate := range candidates[1:] {
		if candidate.FinalScore > bestCandidate.FinalScore {
			bestCandidate = candidate
		}
	}

	return bestCandidate.Tool, nil
}

// ScoreTools scores all tools for a given intent using Bayesian inference.
func (s *BayesianToolSelector) ScoreTools(ctx context.Context, intent string, tools []core.Tool) ([]ToolScore, error) {
	if len(tools) == 0 {
		return nil, errors.New(errors.InvalidInput, "no tools to score")
	}

	scores := make([]ToolScore, len(tools))
	intentTokens := s.tokenizeIntent(intent)

	for i, tool := range tools {
		scores[i] = ToolScore{
			Tool:             tool,
			MatchScore:       s.calculateMatchScore(intentTokens, tool),
			PerformanceScore: s.calculatePerformanceScore(tool),
			CapabilityScore:  s.calculateCapabilityScore(intentTokens, tool),
		}

		// Calculate final score using weighted Bayesian combination
		scores[i].FinalScore = s.calculateFinalScore(scores[i])
	}

	return scores, nil
}

// calculateMatchScore computes how well a tool matches the intent.
func (s *BayesianToolSelector) calculateMatchScore(intentTokens []string, tool core.Tool) float64 {
	toolName := strings.ToLower(tool.Name())
	toolDesc := ""

	if metadata := tool.Metadata(); metadata != nil {
		toolDesc = strings.ToLower(metadata.Description)
	}

	nameScore := s.calculateTextMatch(intentTokens, []string{toolName})
	descScore := s.calculateTextMatch(intentTokens, s.tokenizeText(toolDesc))

	// Weight name matches higher than description matches
	return nameScore*0.7 + descScore*0.3
}

// calculatePerformanceScore computes the performance-based score.
func (s *BayesianToolSelector) calculatePerformanceScore(tool core.Tool) float64 {
	// This would typically access the performance metrics from the registry
	// For now, we'll use a placeholder that considers tool metadata

	metadata := tool.Metadata()
	if metadata == nil {
		return 0.5 // Neutral score
	}

	// Factors that might indicate better performance:
	// 1. Version (newer versions might be better)
	// 2. Number of capabilities (more versatile tools)
	// 3. Schema complexity (well-defined tools)

	score := 0.5 // Base score

	// Version scoring (simple heuristic)
	if metadata.Version != "" {
		score += 0.1
	}

	// Capability count scoring
	capabilityCount := len(metadata.Capabilities)
	if capabilityCount > 0 {
		// Normalize to 0-0.3 range
		score += math.Min(0.3, float64(capabilityCount)*0.05)
	}

	// Schema complexity scoring
	if len(metadata.InputSchema.Properties) > 0 {
		score += 0.1
	}

	return math.Min(1.0, score)
}

// calculateCapabilityScore computes how well tool capabilities match the intent.
func (s *BayesianToolSelector) calculateCapabilityScore(intentTokens []string, tool core.Tool) float64 {
	metadata := tool.Metadata()
	if metadata == nil || len(metadata.Capabilities) == 0 {
		return 0.3 // Low but not zero for tools without explicit capabilities
	}

	capabilities := make([]string, len(metadata.Capabilities))
	for i, cap := range metadata.Capabilities {
		capabilities[i] = strings.ToLower(cap)
	}

	return s.calculateTextMatch(intentTokens, capabilities)
}

// calculateFinalScore combines all scores using Bayesian weights.
func (s *BayesianToolSelector) calculateFinalScore(score ToolScore) float64 {
	// Apply Bayesian combination with configured weights
	finalScore := score.MatchScore*s.MatchWeight +
		score.PerformanceScore*s.PerformanceWeight +
		score.CapabilityScore*s.CapabilityWeight

	// Apply prior probability if available
	if prior, exists := s.PriorProbabilities[score.Tool.Name()]; exists {
		// Bayesian update: P(tool|intent) ∝ P(intent|tool) * P(tool)
		finalScore = finalScore * prior
	}

	return math.Min(1.0, finalScore)
}

// calculateTextMatch computes similarity between intent tokens and target text.
func (s *BayesianToolSelector) calculateTextMatch(intentTokens []string, targetTokens []string) float64 {
	if len(intentTokens) == 0 || len(targetTokens) == 0 {
		return 0.0
	}

	matches := 0
	for _, intentToken := range intentTokens {
		for _, targetToken := range targetTokens {
			if s.tokensMatch(intentToken, targetToken) {
				matches++
				break // Count each intent token at most once
			}
		}
	}

	// Jaccard similarity with bias toward recall
	return float64(matches) / float64(len(intentTokens))
}

// tokensMatch determines if two tokens match (with fuzzy matching).
func (s *BayesianToolSelector) tokensMatch(token1, token2 string) bool {
	// Normalize to lowercase for comparison
	token1 = strings.ToLower(token1)
	token2 = strings.ToLower(token2)

	// Exact match
	if token1 == token2 {
		return true
	}

	// Substring match
	if strings.Contains(token1, token2) || strings.Contains(token2, token1) {
		return true
	}

	// Fuzzy matching for common variations
	return s.fuzzyMatch(token1, token2)
}

// fuzzyMatch implements simple fuzzy matching for common variations.
func (s *BayesianToolSelector) fuzzyMatch(token1, token2 string) bool {
	// Common synonyms and variations
	synonyms := map[string][]string{
		"search": {"find", "query", "lookup", "discover"},
		"create": {"make", "generate", "build", "construct"},
		"read":   {"get", "fetch", "retrieve", "load"},
		"write":  {"save", "store", "persist", "update"},
		"delete": {"remove", "destroy", "erase"},
		"list":   {"show", "display", "enumerate"},
		"parse":  {"analyze", "process", "interpret"},
		"format": {"transform", "convert", "structure"},
	}

	// Check if tokens are synonyms
	for word, syns := range synonyms {
		if (token1 == word && s.contains(syns, token2)) ||
			(token2 == word && s.contains(syns, token1)) ||
			(s.contains(syns, token1) && s.contains(syns, token2)) {
			return true
		}
	}

	return false
}

// contains checks if a slice contains a string.
func (s *BayesianToolSelector) contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// tokenizeIntent breaks down the intent into meaningful tokens.
func (s *BayesianToolSelector) tokenizeIntent(intent string) []string {
	// Simple tokenization - split on common delimiters and normalize
	intent = strings.ToLower(intent)
	intent = strings.ReplaceAll(intent, "_", " ")
	intent = strings.ReplaceAll(intent, "-", " ")

	tokens := strings.Fields(intent)

	// Filter out common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "i": true,
		"you": true, "it": true, "that": true, "this": true, "is": true,
		"are": true, "was": true, "were": true, "be": true, "been": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
	}

	filteredTokens := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if !stopWords[token] && len(token) > 1 {
			filteredTokens = append(filteredTokens, token)
		}
	}

	return filteredTokens
}

// tokenizeText tokenizes arbitrary text.
func (s *BayesianToolSelector) tokenizeText(text string) []string {
	return s.tokenizeIntent(text) // Reuse the same logic
}

// UpdatePriorProbabilities updates the prior probabilities based on tool usage.
func (s *BayesianToolSelector) UpdatePriorProbabilities(toolUsageStats map[string]int) {
	total := 0
	for _, count := range toolUsageStats {
		total += count
	}

	if total == 0 {
		return
	}

	for toolName, count := range toolUsageStats {
		s.PriorProbabilities[toolName] = float64(count) / float64(total)
	}
}

// Ensure BayesianToolSelector implements the ToolSelector interface.
var _ ToolSelector = (*BayesianToolSelector)(nil)
