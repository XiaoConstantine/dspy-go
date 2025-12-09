package ace

import (
	"context"
	"sort"
	"strings"
	"unicode"
)

// Curator manages the learnings file with tiered deduplication.
type Curator struct {
	config            Config
	similarityThreshold float64
}

// NewCurator creates a curator with the given config.
func NewCurator(config Config) *Curator {
	threshold := config.SimilarityThreshold
	if threshold <= 0 {
		threshold = 0.85
	}
	return &Curator{
		config:            config,
		similarityThreshold: threshold,
	}
}

// Curate applies reflection results to the learnings file.
func (c *Curator) Curate(ctx context.Context, file *LearningsFile, results *ReflectionResult) (*CurationResult, error) {
	learnings, err := file.Load()
	if err != nil {
		return nil, err
	}

	tokensBefore, _ := file.EstimateTokens()

	result := &CurationResult{
		FilePath:     file.Path,
		TokensBefore: tokensBefore,
	}

	if results == nil {
		result.TokensAfter = tokensBefore
		return result, nil
	}

	// Apply learning updates (credit assignment)
	for _, update := range results.LearningUpdates {
		l := FindByID(learnings, update.LearningID)
		if l == nil {
			l = FindByShortCode(learnings, update.LearningID)
		}
		if l != nil {
			switch update.Delta {
			case DeltaHelpful:
				l.Helpful++
			case DeltaHarmful:
				l.Harmful++
			}
			result.Updated = append(result.Updated, *l)
		}
	}

	// Add new insights with deduplication
	for _, insight := range results.SuccessPatterns {
		if insight.Confidence < c.config.MinConfidence {
			continue
		}
		if added := c.addOrMerge(&learnings, insight, result); added {
			result.Added = append(result.Added, learnings[len(learnings)-1])
		}
	}

	for _, insight := range results.FailurePatterns {
		if insight.Confidence < c.config.MinConfidence {
			continue
		}
		if added := c.addOrMerge(&learnings, insight, result); added {
			result.Added = append(result.Added, learnings[len(learnings)-1])
		}
	}

	// Prune low-performing learnings
	learnings, pruned := c.prune(learnings)
	result.Pruned = pruned

	// Check token budget
	content := FormatLearnings(learnings)
	tokensAfter := len(content) / 4

	if tokensAfter > c.config.MaxTokens {
		// Aggressive pruning before LLM compaction
		learnings, extraPruned := c.aggressivePrune(learnings, tokensAfter-c.config.MaxTokens)
		result.Pruned = append(result.Pruned, extraPruned...)
		content = FormatLearnings(learnings)
		tokensAfter = len(content) / 4
	}

	result.TokensAfter = tokensAfter

	if err := file.Save(learnings); err != nil {
		return nil, err
	}

	return result, nil
}

// addOrMerge adds a new insight or merges with existing similar learning.
func (c *Curator) addOrMerge(learnings *[]Learning, insight InsightCandidate, result *CurationResult) bool {
	// Tier 1: Exact match
	for i := range *learnings {
		if (*learnings)[i].Content == insight.Content {
			return false
		}
	}

	// Tier 1: Normalized match
	normalizedNew := normalize(insight.Content)
	for i := range *learnings {
		if normalize((*learnings)[i].Content) == normalizedNew {
			return false
		}
	}

	// Tier 1: Token set similarity
	newTokens := tokenize(insight.Content)
	for i := range *learnings {
		existingTokens := tokenize((*learnings)[i].Content)
		if jaccardSimilarity(newTokens, existingTokens) >= c.similarityThreshold {
			// Merge: keep existing, increment helpful
			(*learnings)[i].Helpful++
			result.Merged = append(result.Merged, (*learnings)[i].ID)
			return false
		}
	}

	// No duplicate found, add new learning
	category := insight.Category
	if category == "" {
		category = "strategies"
	}

	newLearning := Learning{
		ID:       GetNextID(*learnings, category),
		Category: category,
		Content:  insight.Content,
		Helpful:  1,
		Harmful:  0,
	}

	*learnings = append(*learnings, newLearning)
	return true
}

// prune removes learnings with poor success rates.
func (c *Curator) prune(learnings []Learning) ([]Learning, []string) {
	var kept []Learning
	var pruned []string

	for _, l := range learnings {
		if l.ShouldPrune(c.config.PruneMinRatio, c.config.PruneMinUsage) {
			pruned = append(pruned, l.ID)
		} else {
			kept = append(kept, l)
		}
	}

	return kept, pruned
}

// aggressivePrune removes lowest-performing learnings to reduce token count.
func (c *Curator) aggressivePrune(learnings []Learning, tokensToRemove int) ([]Learning, []string) {
	if len(learnings) == 0 {
		return learnings, nil
	}

	// Sort by success rate (lowest first)
	type scored struct {
		idx   int
		score float64
		tokens int
	}

	var candidates []scored
	for i, l := range learnings {
		tokens := len(l.String()) / 4
		candidates = append(candidates, scored{i, l.SuccessRate(), tokens})
	}

	// Sort by score ascending (lowest first for removal)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score < candidates[j].score
	})

	// Remove until we've freed enough tokens
	removed := make(map[int]bool)
	var pruned []string
	tokensRemoved := 0

	for _, c := range candidates {
		if tokensRemoved >= tokensToRemove {
			break
		}
		removed[c.idx] = true
		pruned = append(pruned, learnings[c.idx].ID)
		tokensRemoved += c.tokens
	}

	var kept []Learning
	for i, l := range learnings {
		if !removed[i] {
			kept = append(kept, l)
		}
	}

	return kept, pruned
}

// normalize converts text to a canonical form for comparison.
func normalize(s string) string {
	s = strings.ToLower(s)
	s = strings.TrimSpace(s)

	// Collapse whitespace
	var b strings.Builder
	prevSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteRune(' ')
				prevSpace = true
			}
		} else {
			b.WriteRune(r)
			prevSpace = false
		}
	}
	return b.String()
}

// tokenize splits text into word tokens.
func tokenize(s string) map[string]bool {
	tokens := make(map[string]bool)
	s = strings.ToLower(s)

	var word strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			word.WriteRune(r)
		} else if word.Len() > 0 {
			tokens[word.String()] = true
			word.Reset()
		}
	}
	if word.Len() > 0 {
		tokens[word.String()] = true
	}

	return tokens
}

// jaccardSimilarity computes the Jaccard index between two token sets.
func jaccardSimilarity(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	intersection := 0
	for token := range a {
		if b[token] {
			intersection++
		}
	}

	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}
