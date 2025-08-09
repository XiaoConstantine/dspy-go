package react

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// MemoryOptimizer implements memory optimization with forgetting curve.
type MemoryOptimizer struct {
	retention  time.Duration
	threshold  float64
	memory     agents.Memory
	index      *MemoryIndex
	forgetting *ForgettingCurve
	compressor *MemoryCompressor
	mu         sync.RWMutex
}

// MemoryIndex provides fast access to memory items.
type MemoryIndex struct {
	items      map[string]*MemoryItem
	categories map[string][]*MemoryItem
	timestamps []time.Time
	mu         sync.RWMutex
}

// MemoryItem represents an item in memory.
type MemoryItem struct {
	Key          string
	Value        interface{}
	Category     string
	Importance   float64
	AccessCount  int
	LastAccessed time.Time
	Created      time.Time
	Embedding    []float64 // For semantic similarity
	Hash         string
}

// ForgettingCurve implements Ebbinghaus forgetting curve.
type ForgettingCurve struct {
	// R = e^(-t/S) where R is retention, t is time, S is strength
	baseStrength     float64
	strengthModifier float64
	minRetention     float64
}

// MemoryCompressor reduces memory footprint.
type MemoryCompressor struct {
	compressionRatio float64
	strategy         CompressionStrategy
}

// CompressionStrategy defines how memory is compressed.
type CompressionStrategy int

const (
	// CompressionSummarize creates summaries of similar memories.
	CompressionSummarize CompressionStrategy = iota
	// CompressionMerge merges similar memories.
	CompressionMerge
	// CompressionPrune removes low-importance memories.
	CompressionPrune
)

// NewMemoryOptimizer creates a new memory optimizer.
func NewMemoryOptimizer(retention time.Duration, threshold float64) *MemoryOptimizer {
	return &MemoryOptimizer{
		retention:  retention,
		threshold:  threshold,
		memory:     agents.NewInMemoryStore(),
		index:      NewMemoryIndex(),
		forgetting: NewForgettingCurve(),
		compressor: NewMemoryCompressor(),
	}
}

// NewMemoryIndex creates a new memory index.
func NewMemoryIndex() *MemoryIndex {
	return &MemoryIndex{
		items:      make(map[string]*MemoryItem),
		categories: make(map[string][]*MemoryItem),
		timestamps: make([]time.Time, 0),
	}
}

// NewForgettingCurve creates a new forgetting curve calculator.
func NewForgettingCurve() *ForgettingCurve {
	return &ForgettingCurve{
		baseStrength:     1.0,
		strengthModifier: 0.5,
		minRetention:     0.1,
	}
}

// NewMemoryCompressor creates a new memory compressor.
func NewMemoryCompressor() *MemoryCompressor {
	return &MemoryCompressor{
		compressionRatio: 0.5,
		strategy:         CompressionSummarize,
	}
}

// Store saves information to optimized memory.
func (mo *MemoryOptimizer) Store(ctx context.Context, input map[string]interface{}, output map[string]interface{}, success bool) error {
	mo.mu.Lock()
	defer mo.mu.Unlock()

	logger := logging.GetLogger()
	logger.Debug(ctx, "Storing memory with optimization")

	// Create memory item
	key := mo.generateKey(input)
	item := &MemoryItem{
		Key:          key,
		Value:        map[string]interface{}{"input": input, "output": output, "success": success},
		Category:     mo.categorize(input),
		Importance:   mo.calculateImportance(input, output, success),
		AccessCount:  0,
		LastAccessed: time.Now(),
		Created:      time.Now(),
		Hash:         mo.hash(input),
	}

	// Generate embedding if possible
	if embedding := mo.generateEmbedding(input); embedding != nil {
		item.Embedding = embedding
	}

	// Check if we need to compress before storing
	if mo.shouldCompress() {
		mo.compress(ctx)
	}

	// Store in index
	mo.index.Add(item)

	// Store in underlying memory
	return mo.memory.Store(key, item.Value)
}

// Retrieve gets relevant information from memory.
func (mo *MemoryOptimizer) Retrieve(ctx context.Context, query map[string]interface{}) (interface{}, error) {
	mo.mu.RLock()
	defer mo.mu.RUnlock()

	logger := logging.GetLogger()
	logger.Debug(ctx, "Retrieving from optimized memory")

	// First, clean up old memories
	mo.cleanup(ctx)

	// Find relevant memories
	relevant := mo.findRelevant(query, 5)

	if len(relevant) == 0 {
		return nil, nil
	}

	// Update access counts
	for _, item := range relevant {
		item.AccessCount++
		item.LastAccessed = time.Now()
	}

	// Return most relevant
	return relevant[0].Value, nil
}

// generateKey creates a unique key for memory storage.
func (mo *MemoryOptimizer) generateKey(input map[string]interface{}) string {
	// Create a deterministic key based on input
	taskStr := ""
	if task, ok := input["task"].(string); ok {
		taskStr = task
	}

	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("memory_%s_%d", mo.hash(taskStr), timestamp)
}

// hash generates a hash of the input.
func (mo *MemoryOptimizer) hash(input interface{}) string {
	h := sha256.New()
	fmt.Fprintf(h, "%v", input)
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// categorize determines the category of a memory item.
func (mo *MemoryOptimizer) categorize(input map[string]interface{}) string {
	// Simple categorization based on input structure
	if task, ok := input["task"].(string); ok {
		// Categorize based on task keywords
		categories := map[string][]string{
			"research":    {"research", "investigate", "analyze", "study"},
			"calculation": {"calculate", "compute", "solve", "math"},
			"synthesis":   {"synthesize", "summarize", "combine", "merge"},
			"comparison":  {"compare", "contrast", "evaluate", "assess"},
			"creation":    {"create", "generate", "write", "design"},
		}

		taskLower := strings.ToLower(task)
		for category, keywords := range categories {
			for _, keyword := range keywords {
				if strings.Contains(taskLower, keyword) {
					return category
				}
			}
		}
	}

	return "general"
}

// calculateImportance determines the importance of a memory item.
func (mo *MemoryOptimizer) calculateImportance(input, output map[string]interface{}, success bool) float64 {
	importance := 0.5 // Base importance

	// Success increases importance
	if success {
		importance += 0.2
	}

	// Complex outputs are more important
	if len(output) > 3 {
		importance += 0.1
	}

	// Recent memories are more important (will decay over time)
	importance += 0.2

	return math.Min(importance, 1.0)
}

// generateEmbedding creates a vector embedding for semantic similarity.
func (mo *MemoryOptimizer) generateEmbedding(input map[string]interface{}) []float64 {
	// Simplified embedding - in practice, use a proper embedding model
	// For now, create a simple feature vector
	embedding := make([]float64, 10)

	// Extract features from input
	if task, ok := input["task"].(string); ok {
		// Simple features: length, word count, etc.
		embedding[0] = float64(len(task)) / 100.0
		embedding[1] = float64(len(strings.Split(task, " "))) / 20.0

		// Keyword presence
		keywords := []string{"analyze", "compute", "search", "create", "compare"}
		for i, keyword := range keywords {
			if strings.Contains(strings.ToLower(task), keyword) {
				embedding[2+i] = 1.0
			}
		}
	}

	return embedding
}

// shouldCompress determines if memory compression is needed.
func (mo *MemoryOptimizer) shouldCompress() bool {
	itemCount := len(mo.index.items)
	// Compress when we have too many items
	return itemCount > 100
}

// compress reduces memory footprint.
func (mo *MemoryOptimizer) compress(ctx context.Context) {
	logger := logging.GetLogger()
	logger.Debug(ctx, "Compressing memory")

	switch mo.compressor.strategy {
	case CompressionSummarize:
		mo.compressBySummarization(ctx)
	case CompressionMerge:
		mo.compressByMerging(ctx)
	case CompressionPrune:
		mo.compressByPruning(ctx)
	}
}

// compressBySummarization creates summaries of similar memories.
func (mo *MemoryOptimizer) compressBySummarization(ctx context.Context) {
	// Group similar memories
	groups := mo.groupSimilarMemories()

	for category, items := range groups {
		if len(items) > 5 {
			// Create summary
			summary := mo.createSummary(items)

			// Replace individual items with summary
			summaryItem := &MemoryItem{
				Key:          fmt.Sprintf("summary_%s_%d", category, time.Now().Unix()),
				Value:        summary,
				Category:     category,
				Importance:   mo.calculateGroupImportance(items),
				AccessCount:  mo.sumAccessCounts(items),
				LastAccessed: time.Now(),
				Created:      time.Now(),
			}

			// Remove individual items
			for _, item := range items[5:] {
				mo.index.Remove(item.Key)
				// Note: Memory interface doesn't have Delete, items will be cleaned up by GC
			}

			// Add summary
			mo.index.Add(summaryItem)
			if err := mo.memory.Store(summaryItem.Key, summaryItem.Value); err != nil {
				// Log error but continue processing
				continue
			}
		}
	}
}

// compressByMerging merges similar memories.
func (mo *MemoryOptimizer) compressByMerging(ctx context.Context) {
	// Find and merge duplicate or very similar memories
	for key1, item1 := range mo.index.items {
		for key2, item2 := range mo.index.items {
			if key1 != key2 && mo.areSimilar(item1, item2) {
				// Merge items
				merged := mo.mergeItems(item1, item2)

				// Remove old items
				mo.index.Remove(key2)
				// Note: Memory interface doesn't have Delete, items will be cleaned up by GC

				// Update first item
				mo.index.items[key1] = merged
				if err := mo.memory.Store(key1, merged.Value); err != nil {
					// Log error but continue processing
					continue
				}
			}
		}
	}
}

// compressByPruning removes low-importance memories.
func (mo *MemoryOptimizer) compressByPruning(ctx context.Context) {
	// Calculate retention scores
	scores := make(map[string]float64)
	for key, item := range mo.index.items {
		scores[key] = mo.calculateRetentionScore(item)
	}

	// Sort by score
	type scorePair struct {
		key   string
		score float64
	}
	pairs := make([]scorePair, 0, len(scores))
	for k, v := range scores {
		pairs = append(pairs, scorePair{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score < pairs[j].score
	})

	// Remove bottom percentile
	cutoff := int(float64(len(pairs)) * mo.compressor.compressionRatio)
	for i := 0; i < cutoff; i++ {
		mo.index.Remove(pairs[i].key)
		// Note: Memory interface doesn't have Delete, items will be cleaned up by GC
	}
}

// cleanup removes old memories based on forgetting curve.
func (mo *MemoryOptimizer) cleanup(ctx context.Context) {
	now := time.Now()
	toRemove := make([]string, 0)

	for key, item := range mo.index.items {
		age := now.Sub(item.Created)
		if age > mo.retention {
			// Check retention based on forgetting curve
			retention := mo.forgetting.Calculate(age, item.Importance, item.AccessCount)
			if retention < mo.threshold {
				toRemove = append(toRemove, key)
			}
		}
	}

	// Remove items
	for _, key := range toRemove {
		mo.index.Remove(key)
		// Note: Memory interface doesn't have Delete, items will be cleaned up by GC
	}
}

// findRelevant finds the most relevant memories for a query.
func (mo *MemoryOptimizer) findRelevant(query map[string]interface{}, limit int) []*MemoryItem {
	queryCategory := mo.categorize(query)
	queryEmbedding := mo.generateEmbedding(query)
	queryHash := mo.hash(query)

	// Score all items
	type scoredItem struct {
		item  *MemoryItem
		score float64
	}
	scored := make([]scoredItem, 0)

	for _, item := range mo.index.items {
		score := 0.0

		// Category match
		if item.Category == queryCategory {
			score += 0.3
		}

		// Semantic similarity
		if item.Embedding != nil && queryEmbedding != nil {
			similarity := mo.cosineSimilarity(item.Embedding, queryEmbedding)
			score += similarity * 0.4
		}

		// Hash similarity (exact match)
		if item.Hash == queryHash {
			score += 0.2
		}

		// Recency and importance
		age := time.Since(item.Created)
		recency := math.Exp(-age.Hours() / 24) // Decay over days
		score += recency * 0.05
		score += item.Importance * 0.05

		scored = append(scored, scoredItem{item, score})
	}

	// Sort by score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Return top items
	result := make([]*MemoryItem, 0, limit)
	for i := 0; i < limit && i < len(scored); i++ {
		result = append(result, scored[i].item)
	}

	return result
}

// groupSimilarMemories groups memories by similarity.
func (mo *MemoryOptimizer) groupSimilarMemories() map[string][]*MemoryItem {
	return mo.index.categories
}

// createSummary creates a summary of multiple memory items.
func (mo *MemoryOptimizer) createSummary(items []*MemoryItem) interface{} {
	summary := map[string]interface{}{
		"type":         "summary",
		"item_count":   len(items),
		"created":      time.Now(),
		"categories":   make(map[string]int),
		"success_rate": 0.0,
	}

	successCount := 0
	for _, item := range items {
		summary["categories"].(map[string]int)[item.Category]++
		if val, ok := item.Value.(map[string]interface{}); ok {
			if success, ok := val["success"].(bool); ok && success {
				successCount++
			}
		}
	}

	summary["success_rate"] = float64(successCount) / float64(len(items))
	return summary
}

// calculateGroupImportance calculates importance for a group of items.
func (mo *MemoryOptimizer) calculateGroupImportance(items []*MemoryItem) float64 {
	if len(items) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, item := range items {
		sum += item.Importance
	}
	return sum / float64(len(items))
}

// sumAccessCounts sums access counts for a group of items.
func (mo *MemoryOptimizer) sumAccessCounts(items []*MemoryItem) int {
	sum := 0
	for _, item := range items {
		sum += item.AccessCount
	}
	return sum
}

// areSimilar checks if two memory items are similar.
func (mo *MemoryOptimizer) areSimilar(item1, item2 *MemoryItem) bool {
	// Check category
	if item1.Category != item2.Category {
		return false
	}

	// Check semantic similarity
	if item1.Embedding != nil && item2.Embedding != nil {
		similarity := mo.cosineSimilarity(item1.Embedding, item2.Embedding)
		return similarity > 0.8
	}

	// Check hash similarity
	return item1.Hash == item2.Hash
}

// mergeItems merges two memory items.
func (mo *MemoryOptimizer) mergeItems(item1, item2 *MemoryItem) *MemoryItem {
	merged := &MemoryItem{
		Key:          item1.Key,
		Value:        mo.mergeValues(item1.Value, item2.Value),
		Category:     item1.Category,
		Importance:   math.Max(item1.Importance, item2.Importance),
		AccessCount:  item1.AccessCount + item2.AccessCount,
		LastAccessed: mo.laterTime(item1.LastAccessed, item2.LastAccessed),
		Created:      mo.earlierTime(item1.Created, item2.Created),
		Embedding:    item1.Embedding, // Keep first embedding
		Hash:         item1.Hash,
	}
	return merged
}

// mergeValues merges two values.
func (mo *MemoryOptimizer) mergeValues(val1, val2 interface{}) interface{} {
	// Simple merge - in practice, this could be more sophisticated
	return map[string]interface{}{
		"merged":    true,
		"values":    []interface{}{val1, val2},
		"merged_at": time.Now(),
	}
}

// calculateRetentionScore calculates how important it is to retain a memory.
func (mo *MemoryOptimizer) calculateRetentionScore(item *MemoryItem) float64 {
	age := time.Since(item.Created)
	retention := mo.forgetting.Calculate(age, item.Importance, item.AccessCount)

	// Boost for recently accessed items
	recencyBoost := 0.0
	if time.Since(item.LastAccessed) < 24*time.Hour {
		recencyBoost = 0.2
	}

	return retention + recencyBoost
}

// cosineSimilarity calculates cosine similarity between two vectors.
func (mo *MemoryOptimizer) cosineSimilarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0.0
	}

	var dotProduct, norm1, norm2 float64
	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// laterTime returns the later of two times.
func (mo *MemoryOptimizer) laterTime(t1, t2 time.Time) time.Time {
	if t1.After(t2) {
		return t1
	}
	return t2
}

// earlierTime returns the earlier of two times.
func (mo *MemoryOptimizer) earlierTime(t1, t2 time.Time) time.Time {
	if t1.Before(t2) {
		return t1
	}
	return t2
}

// Calculate computes retention based on forgetting curve.
func (fc *ForgettingCurve) Calculate(age time.Duration, importance float64, accessCount int) float64 {
	// Ebbinghaus forgetting curve: R = e^(-t/S)
	// Where R is retention, t is time, S is strength

	// Calculate strength based on importance and access count
	strength := fc.baseStrength * (1 + importance) * (1 + math.Log(float64(accessCount+1)))
	strength *= fc.strengthModifier

	// Calculate retention
	t := age.Hours() / 24.0 // Convert to days
	retention := math.Exp(-t / strength)

	// Apply minimum retention
	if retention < fc.minRetention {
		retention = fc.minRetention
	}

	return retention
}

// Add adds an item to the index.
func (mi *MemoryIndex) Add(item *MemoryItem) {
	mi.mu.Lock()
	defer mi.mu.Unlock()

	mi.items[item.Key] = item
	mi.categories[item.Category] = append(mi.categories[item.Category], item)
	mi.timestamps = append(mi.timestamps, item.Created)
}

// Remove removes an item from the index.
func (mi *MemoryIndex) Remove(key string) {
	mi.mu.Lock()
	defer mi.mu.Unlock()

	if item, exists := mi.items[key]; exists {
		delete(mi.items, key)

		// Remove from category
		if catItems, ok := mi.categories[item.Category]; ok {
			newCatItems := make([]*MemoryItem, 0)
			for _, catItem := range catItems {
				if catItem.Key != key {
					newCatItems = append(newCatItems, catItem)
				}
			}
			mi.categories[item.Category] = newCatItems
		}
	}
}

// Get retrieves an item from the index.
func (mi *MemoryIndex) Get(key string) (*MemoryItem, bool) {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	item, exists := mi.items[key]
	return item, exists
}

// GetByCategory retrieves items by category.
func (mi *MemoryIndex) GetByCategory(category string) []*MemoryItem {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	return mi.categories[category]
}

// Size returns the number of items in the index.
func (mi *MemoryIndex) Size() int {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	return len(mi.items)
}

// GetStatistics returns memory statistics.
func (mo *MemoryOptimizer) GetStatistics() map[string]interface{} {
	mo.mu.RLock()
	defer mo.mu.RUnlock()

	stats := map[string]interface{}{
		"total_items":       mo.index.Size(),
		"categories":        len(mo.index.categories),
		"retention_rate":    mo.threshold,
		"compression_ratio": mo.compressor.compressionRatio,
	}

	// Category distribution
	catDist := make(map[string]int)
	for cat, items := range mo.index.categories {
		catDist[cat] = len(items)
	}
	stats["category_distribution"] = catDist

	// Average importance and access count
	totalImportance := 0.0
	totalAccess := 0
	for _, item := range mo.index.items {
		totalImportance += item.Importance
		totalAccess += item.AccessCount
	}

	if mo.index.Size() > 0 {
		stats["avg_importance"] = totalImportance / float64(mo.index.Size())
		stats["avg_access_count"] = float64(totalAccess) / float64(mo.index.Size())
	}

	return stats
}
