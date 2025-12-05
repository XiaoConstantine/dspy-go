package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// ErrorRetainer implements Manus's error retention pattern for implicit learning.
// Instead of hiding errors, we keep them in context so agents learn from failures.
// This prevents repeated mistakes and improves agent reliability over time.
type ErrorRetainer struct {
	mu sync.RWMutex

	memory    *FileSystemMemory
	errors    []ErrorRecord
	successes []SuccessRecord

	// Configuration
	config        Config
	maxErrors     int
	maxSuccesses  int
	retentionTime time.Duration

	// Metrics
	totalErrors       int64
	totalSuccesses    int64
	learnedPatterns   int64
	preventedRetries  int64
}

// ErrorRecord captures detailed information about failures for learning.
type ErrorRecord struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	ErrorType   string                 `json:"error_type"`
	Message     string                 `json:"message"`
	Context     map[string]interface{} `json:"context"`
	StackTrace  string                 `json:"stack_trace,omitempty"`
	Input       string                 `json:"input,omitempty"`
	Attempt     int                    `json:"attempt"`
	Resolution  string                 `json:"resolution,omitempty"`
	Severity    ErrorSeverity          `json:"severity"`
	Pattern     string                 `json:"pattern,omitempty"`
	Learned     bool                   `json:"learned"`
}

// SuccessRecord captures successful resolutions for pattern reinforcement.
type SuccessRecord struct {
	ID           string                 `json:"id"`
	Timestamp    time.Time              `json:"timestamp"`
	SuccessType  string                 `json:"success_type"`
	Description  string                 `json:"description"`
	Context      map[string]interface{} `json:"context"`
	Input        string                 `json:"input,omitempty"`
	Output       string                 `json:"output,omitempty"`
	PrevErrors   []string               `json:"previous_errors,omitempty"`
	Pattern      string                 `json:"pattern,omitempty"`
	Confidence   float64                `json:"confidence"`
}

// ErrorSeverity categorizes errors for prioritized learning.
type ErrorSeverity string

const (
	SeverityCritical ErrorSeverity = "critical" // System failures, data corruption
	SeverityHigh     ErrorSeverity = "high"     // Major functionality broken
	SeverityMedium   ErrorSeverity = "medium"   // Degraded performance
	SeverityLow      ErrorSeverity = "low"      // Minor issues, warnings
	SeverityInfo     ErrorSeverity = "info"     // Informational only
)

// NewErrorRetainer creates a retainer that learns from errors instead of hiding them.
func NewErrorRetainer(memory *FileSystemMemory, config Config) *ErrorRetainer {
	return &ErrorRetainer{
		memory:        memory,
		errors:        make([]ErrorRecord, 0),
		successes:     make([]SuccessRecord, 0),
		config:        config,
		maxErrors:     config.MaxErrorRetention,
		maxSuccesses:  config.MaxErrorRetention / 2, // Keep fewer successes
		retentionTime: config.Memory.RetentionPeriod,
	}
}

// RecordError captures an error for learning and pattern recognition.
// CRITICAL: This makes errors visible to the agent for implicit learning.
func (er *ErrorRetainer) RecordError(ctx context.Context, errorType, message string, severity ErrorSeverity, errorCtx map[string]interface{}) string {
	er.mu.Lock()
	defer er.mu.Unlock()

	logger := logging.GetLogger()

	// Create detailed error record
	record := ErrorRecord{
		ID:        fmt.Sprintf("error_%d", time.Now().UnixNano()),
		Timestamp: time.Now(),
		ErrorType: errorType,
		Message:   message,
		Context:   errorCtx,
		Attempt:   er.getAttemptCount(errorType, message),
		Severity:  severity,
		Pattern:   er.detectErrorPattern(errorType, message),
		Learned:   false,
	}

	// Add to in-memory storage
	er.errors = append(er.errors, record)
	er.totalErrors++

	// Trim if exceeding max retention
	if len(er.errors) > er.maxErrors {
		er.errors = er.errors[len(er.errors)-er.maxErrors:]
	}

	// Store to filesystem for persistence
	if err := er.persistErrors(ctx); err != nil {
		logger.Warn(ctx, "Failed to persist error records: %v", err)
	}

	logger.Debug(ctx, "Recorded error for learning: %s (pattern: %s, attempt: %d)",
		record.ID, record.Pattern, record.Attempt)

	// Return formatted error summary for context inclusion
	return er.formatErrorForContext(record)
}

// RecordSuccess captures successful resolutions to reinforce positive patterns.
func (er *ErrorRetainer) RecordSuccess(ctx context.Context, successType, description string, successCtx map[string]interface{}) string {
	er.mu.Lock()
	defer er.mu.Unlock()

	logger := logging.GetLogger()

	// Create success record
	record := SuccessRecord{
		ID:          fmt.Sprintf("success_%d", time.Now().UnixNano()),
		Timestamp:   time.Now(),
		SuccessType: successType,
		Description: description,
		Context:     successCtx,
		Pattern:     er.detectSuccessPattern(successType, description),
		Confidence:  er.calculateConfidence(successType),
		PrevErrors:  er.getRelatedErrors(successType),
	}

	// Add to in-memory storage
	er.successes = append(er.successes, record)
	er.totalSuccesses++

	// Trim if exceeding max retention
	if len(er.successes) > er.maxSuccesses {
		er.successes = er.successes[len(er.successes)-er.maxSuccesses:]
	}

	// Mark related errors as potentially resolved
	er.markErrorsAsLearned(successType, record.Pattern)

	// Store to filesystem
	if err := er.persistSuccesses(ctx); err != nil {
		logger.Warn(ctx, "Failed to persist success records: %v", err)
	}

	logger.Debug(ctx, "Recorded success for pattern reinforcement: %s (pattern: %s)",
		record.ID, record.Pattern)

	return er.formatSuccessForContext(record)
}

// GetErrorContext returns error history formatted for agent context inclusion.
// This is the CORE pattern - making errors visible for learning.
func (er *ErrorRetainer) GetErrorContext() string {
	er.mu.RLock()
	defer er.mu.RUnlock()

	if len(er.errors) == 0 && len(er.successes) == 0 {
		return ""
	}

	var content strings.Builder
	content.WriteString("## Recent Learning History\n\n")

	// Recent errors for pattern recognition
	recentErrors := er.getRecentErrors(5) // Show last 5 errors
	if len(recentErrors) > 0 {
		content.WriteString("### Recent Errors (Learn from these):\n")
		for _, err := range recentErrors {
			content.WriteString(fmt.Sprintf("- **%s** (%s): %s\n",
				err.ErrorType, err.Severity, err.Message))
			if err.Pattern != "" {
				content.WriteString(fmt.Sprintf("  *Pattern: %s, Attempt: %d*\n", err.Pattern, err.Attempt))
			}
		}
		content.WriteString("\n")
	}

	// Recent successes for pattern reinforcement
	recentSuccesses := er.getRecentSuccesses(3) // Show last 3 successes
	if len(recentSuccesses) > 0 {
		content.WriteString("### Recent Successes (Reinforce these patterns):\n")
		for _, success := range recentSuccesses {
			content.WriteString(fmt.Sprintf("- **%s**: %s\n",
				success.SuccessType, success.Description))
			if success.Pattern != "" {
				content.WriteString(fmt.Sprintf("  *Pattern: %s, Confidence: %.2f*\n",
					success.Pattern, success.Confidence))
			}
		}
		content.WriteString("\n")
	}

	// Learning insights
	insights := er.generateLearningInsights()
	if len(insights) > 0 {
		content.WriteString("### Learning Insights:\n")
		for _, insight := range insights {
			content.WriteString(fmt.Sprintf("- %s\n", insight))
		}
		content.WriteString("\n")
	}

	return content.String()
}

// GetErrorPattern analyzes if a new error matches known patterns.
func (er *ErrorRetainer) GetErrorPattern(errorType, message string) (string, int, bool) {
	er.mu.RLock()
	defer er.mu.RUnlock()

	pattern := er.detectErrorPattern(errorType, message)
	attempts := er.getAttemptCount(errorType, message)
	isRepeated := attempts > 1

	return pattern, attempts, isRepeated
}

// ShouldPreventRetry determines if an error pattern suggests stopping retries.
func (er *ErrorRetainer) ShouldPreventRetry(errorType, message string, currentAttempt int) bool {
	er.mu.RLock()
	defer er.mu.RUnlock()

	// Check if this error pattern has failed multiple times
	attempts := er.getAttemptCount(errorType, message)

	// Prevent retry if:
	// 1. We've seen this exact pattern fail 3+ times
	// 2. It's a critical error that typically doesn't resolve with retries
	// 3. The pattern shows consistent failure

	if attempts >= 3 {
		er.preventedRetries++
		return true
	}

	// Check for critical error patterns that shouldn't be retried
	criticalPatterns := []string{"auth_failure", "permission_denied", "not_found", "invalid_input"}
	pattern := er.detectErrorPattern(errorType, message)

	for _, critical := range criticalPatterns {
		if strings.Contains(pattern, critical) {
			er.preventedRetries++
			return true
		}
	}

	return false
}

// GetLearningMetrics returns metrics about error learning effectiveness.
func (er *ErrorRetainer) GetLearningMetrics() map[string]interface{} {
	er.mu.RLock()
	defer er.mu.RUnlock()

	learnedErrors := 0
	for _, err := range er.errors {
		if err.Learned {
			learnedErrors++
		}
	}

	return map[string]interface{}{
		"total_errors":       er.totalErrors,
		"total_successes":    er.totalSuccesses,
		"current_errors":     len(er.errors),
		"current_successes":  len(er.successes),
		"learned_errors":     learnedErrors,
		"learned_patterns":   er.learnedPatterns,
		"prevented_retries":  er.preventedRetries,
		"learning_rate":      float64(learnedErrors) / float64(len(er.errors)+1), // +1 to avoid division by zero
	}
}

// Helper methods

func (er *ErrorRetainer) getAttemptCount(errorType, message string) int {
	count := 1
	pattern := er.detectErrorPattern(errorType, message)

	for _, err := range er.errors {
		if err.Pattern == pattern || (err.ErrorType == errorType && strings.Contains(err.Message, message[:minInt(50, len(message))])) {
			count++
		}
	}

	return count
}

func (er *ErrorRetainer) detectErrorPattern(errorType, message string) string {
	// Simple pattern detection based on error type and message keywords
	patterns := map[string][]string{
		"timeout":        {"timeout", "deadline", "connection"},
		"auth_failure":   {"auth", "authentication", "unauthorized", "forbidden"},
		"rate_limit":     {"rate", "limit", "throttle", "quota"},
		"not_found":      {"not found", "404", "missing", "does not exist"},
		"invalid_input":  {"invalid", "malformed", "parse", "format"},
		"network":        {"network", "connection", "dns", "host"},
		"permission":     {"permission", "access", "denied", "forbidden"},
	}

	lowerMessage := strings.ToLower(message)
	lowerType := strings.ToLower(errorType)

	for pattern, keywords := range patterns {
		for _, keyword := range keywords {
			if strings.Contains(lowerMessage, keyword) || strings.Contains(lowerType, keyword) {
				return pattern
			}
		}
	}

	// Fallback to error type
	return strings.ToLower(errorType)
}

func (er *ErrorRetainer) detectSuccessPattern(successType, description string) string {
	patterns := map[string][]string{
		"retry_success":    {"retry", "attempt", "recovered"},
		"fallback_success": {"fallback", "alternative", "backup"},
		"auth_recovered":   {"auth", "login", "token", "credential"},
		"optimization":     {"optimized", "improved", "faster", "efficient"},
		"validation":       {"validated", "verified", "checked", "confirmed"},
	}

	lowerDescription := strings.ToLower(description)
	lowerType := strings.ToLower(successType)

	for pattern, keywords := range patterns {
		for _, keyword := range keywords {
			if strings.Contains(lowerDescription, keyword) || strings.Contains(lowerType, keyword) {
				return pattern
			}
		}
	}

	return strings.ToLower(successType)
}

func (er *ErrorRetainer) calculateConfidence(successType string) float64 {
	// Calculate confidence based on historical success rate for this type
	count := 0
	for _, success := range er.successes {
		if success.SuccessType == successType {
			count++
		}
	}

	// Simple confidence calculation - more successes = higher confidence
	confidence := float64(count) / float64(len(er.successes)+1)
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (er *ErrorRetainer) getRelatedErrors(successType string) []string {
	var related []string

	// Find errors that might be related to this success
	for _, err := range er.errors {
		if strings.Contains(err.ErrorType, successType) ||
		   strings.Contains(successType, err.ErrorType) {
			related = append(related, err.ID)
		}
	}

	return related
}

func (er *ErrorRetainer) markErrorsAsLearned(successType, pattern string) {
	for i := range er.errors {
		if er.errors[i].Pattern == pattern ||
		   strings.Contains(er.errors[i].ErrorType, successType) {
			if !er.errors[i].Learned {
				er.errors[i].Learned = true
				er.learnedPatterns++
			}
		}
	}
}

func (er *ErrorRetainer) getRecentErrors(limit int) []ErrorRecord {
	if len(er.errors) == 0 {
		return []ErrorRecord{}
	}

	start := len(er.errors) - limit
	if start < 0 {
		start = 0
	}

	return er.errors[start:]
}

func (er *ErrorRetainer) getRecentSuccesses(limit int) []SuccessRecord {
	if len(er.successes) == 0 {
		return []SuccessRecord{}
	}

	start := len(er.successes) - limit
	if start < 0 {
		start = 0
	}

	return er.successes[start:]
}

func (er *ErrorRetainer) generateLearningInsights() []string {
	var insights []string

	// Analyze error patterns
	patternCounts := make(map[string]int)
	for _, err := range er.errors {
		if err.Pattern != "" {
			patternCounts[err.Pattern]++
		}
	}

	// Generate insights based on patterns
	for pattern, count := range patternCounts {
		if count >= 3 {
			insights = append(insights, fmt.Sprintf("Frequent %s errors (%d occurrences) - consider pattern-specific handling", pattern, count))
		}
	}

	// Check retry effectiveness
	if er.preventedRetries > 0 {
		insights = append(insights, fmt.Sprintf("Prevented %d futile retries based on error patterns", er.preventedRetries))
	}

	// Learning rate insight
	metrics := er.GetLearningMetrics()
	if rate, ok := metrics["learning_rate"].(float64); ok && rate > 0.5 {
		insights = append(insights, fmt.Sprintf("Good learning rate: %.1f%% of errors have been learned from", rate*100))
	}

	return insights
}

func (er *ErrorRetainer) formatErrorForContext(record ErrorRecord) string {
	return fmt.Sprintf("[Error recorded: %s - %s (attempt %d, pattern: %s)]",
		record.ErrorType, record.Message, record.Attempt, record.Pattern)
}

func (er *ErrorRetainer) formatSuccessForContext(record SuccessRecord) string {
	return fmt.Sprintf("[Success recorded: %s - %s (pattern: %s, confidence: %.2f)]",
		record.SuccessType, record.Description, record.Pattern, record.Confidence)
}

func (er *ErrorRetainer) persistErrors(ctx context.Context) error {
	data, err := json.MarshalIndent(er.errors, "", "  ")
	if err != nil {
		return err
	}

	_, err = er.memory.StoreFile(ctx, "errors", "current", data, map[string]interface{}{
		"count":     len(er.errors),
		"timestamp": time.Now(),
	})

	return err
}

func (er *ErrorRetainer) persistSuccesses(ctx context.Context) error {
	data, err := json.MarshalIndent(er.successes, "", "  ")
	if err != nil {
		return err
	}

	_, err = er.memory.StoreFile(ctx, "successes", "current", data, map[string]interface{}{
		"count":     len(er.successes),
		"timestamp": time.Now(),
	})

	return err
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
