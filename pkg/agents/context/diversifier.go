package context

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// ContextDiversifier implements Manus's context diversity patterns to prevent few-shot traps.
// The key insight: agents get stuck in patterns when they see the same context repeatedly.
// By rotating context presentations, we maintain performance while preventing overfitting.
type ContextDiversifier struct {
	mu sync.RWMutex

	memory           *FileSystemMemory
	contextTemplates []ContextTemplate
	usageHistory     map[string]ContextUsage
	diversityMetrics DiversityMetrics

	// Rotation settings
	rotationInterval   time.Duration
	maxTemplateReuse   int
	diversityThreshold float64

	// Configuration
	config Config
}

// ContextTemplate defines different ways to present the same information.
type ContextTemplate struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Structure     TemplateStructure      `json:"structure"`
	Style         PresentationStyle      `json:"style"`
	UsageCount    int64                  `json:"usage_count"`
	LastUsed      time.Time              `json:"last_used"`
	Effectiveness float64                `json:"effectiveness"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// TemplateStructure defines how content is organized in the template.
type TemplateStructure string

const (
	StructureLinear       TemplateStructure = "linear"        // Traditional top-to-bottom
	StructureHierarchical TemplateStructure = "hierarchical"  // Nested sections
	StructureBulletPoints TemplateStructure = "bullet_points" // Bulleted lists
	StructureTimeline     TemplateStructure = "timeline"      // Chronological order
	StructureMatrix       TemplateStructure = "matrix"        // Table/grid format
	StructureNarrative    TemplateStructure = "narrative"     // Story-like flow
)

// PresentationStyle defines how content is stylistically presented.
type PresentationStyle string

const (
	StyleFormal     PresentationStyle = "formal"     // Professional tone
	StyleCasual     PresentationStyle = "casual"     // Conversational tone
	StyleTechnical  PresentationStyle = "technical"  // Developer-focused
	StyleExecutive  PresentationStyle = "executive"  // High-level summary
	StyleDetailed   PresentationStyle = "detailed"   // Comprehensive info
	StyleMinimalist PresentationStyle = "minimalist" // Essential info only
)

// ContextUsage tracks how each template performs.
type ContextUsage struct {
	TemplateID      string    `json:"template_id"`
	UsageCount      int64     `json:"usage_count"`
	LastUsed        time.Time `json:"last_used"`
	SuccessRate     float64   `json:"success_rate"`
	AverageLatency  float64   `json:"average_latency"`
	TokenEfficiency float64   `json:"token_efficiency"`
}

// DiversityMetrics tracks overall context diversity health.
type DiversityMetrics struct {
	TotalTemplates      int     `json:"total_templates"`
	ActiveTemplates     int     `json:"active_templates"`
	TemplateEntropy     float64 `json:"template_entropy"`
	DiversityScore      float64 `json:"diversity_score"`
	RecentRotations     int64   `json:"recent_rotations"`
	PreventedStagnation int64   `json:"prevented_stagnation"`
}

// DiversificationResult provides information about context transformation.
type DiversificationResult struct {
	OriginalTemplate string                 `json:"original_template"`
	NewTemplate      string                 `json:"new_template"`
	Transformation   string                 `json:"transformation"`
	DiversityGain    float64                `json:"diversity_gain"`
	EstimatedImpact  string                 `json:"estimated_impact"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// NewContextDiversifier creates a diversifier to prevent few-shot pattern traps.
func NewContextDiversifier(memory *FileSystemMemory, config Config) *ContextDiversifier {
	diversifier := &ContextDiversifier{
		memory:             memory,
		contextTemplates:   make([]ContextTemplate, 0),
		usageHistory:       make(map[string]ContextUsage),
		rotationInterval:   15 * time.Minute, // Rotate every 15 minutes
		maxTemplateReuse:   5,                // Max times to reuse same template
		diversityThreshold: 0.3,              // Minimum diversity score
		config:             config,
	}

	// Initialize default templates
	diversifier.initializeDefaultTemplates()

	return diversifier
}

// DiversifyContext applies diversity patterns to prevent few-shot traps.
// CRITICAL: This is the core method that prevents agent stagnation.
func (cd *ContextDiversifier) DiversifyContext(ctx context.Context, content string, contentType string) (string, DiversificationResult, error) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	logger := logging.GetLogger()

	// Analyze current context patterns
	currentTemplate := cd.detectCurrentTemplate(content)

	// Check if we need to rotate template
	shouldRotate, reason := cd.shouldRotateTemplate(currentTemplate)

	if !shouldRotate {
		// No rotation needed, return original content
		return content, DiversificationResult{
			OriginalTemplate: currentTemplate.ID,
			NewTemplate:      currentTemplate.ID,
			Transformation:   "none",
			DiversityGain:    0.0,
			EstimatedImpact:  "no_change",
		}, nil
	}

	// Select new template based on diversity needs
	newTemplate := cd.selectDiverseTemplate(currentTemplate, contentType)

	// Transform content using new template
	diversifiedContent, err := cd.transformContent(content, currentTemplate, newTemplate)
	if err != nil {
		return content, DiversificationResult{}, err
	}

	// Update usage tracking
	cd.updateTemplateUsage(newTemplate.ID, true)
	cd.diversityMetrics.RecentRotations++

	logger.Debug(ctx, "Diversified context: %s -> %s (reason: %s)",
		currentTemplate.ID, newTemplate.ID, reason)

	result := DiversificationResult{
		OriginalTemplate: currentTemplate.ID,
		NewTemplate:      newTemplate.ID,
		Transformation:   cd.describeTransformation(currentTemplate, newTemplate),
		DiversityGain:    cd.calculateDiversityGain(currentTemplate, newTemplate),
		EstimatedImpact:  cd.estimateImpact(currentTemplate, newTemplate),
		Metadata: map[string]interface{}{
			"rotation_reason": reason,
			"content_type":    contentType,
			"timestamp":       time.Now(),
		},
	}

	return diversifiedContent, result, nil
}

// RotateTemplateIfNeeded checks if template rotation is needed and performs it.
func (cd *ContextDiversifier) RotateTemplateIfNeeded(ctx context.Context, currentContent string) (string, bool) {
	rotated, result, err := cd.DiversifyContext(ctx, currentContent, "general")
	if err != nil {
		return currentContent, false
	}

	return rotated, result.Transformation != "none"
}

// GetDiversityHealth returns current diversity metrics.
func (cd *ContextDiversifier) GetDiversityHealth() DiversityMetrics {
	cd.mu.RLock()
	defer cd.mu.RUnlock()

	// Calculate template entropy
	entropy := cd.calculateTemplateEntropy()

	// Calculate overall diversity score
	diversityScore := cd.calculateDiversityScore()

	cd.diversityMetrics.TemplateEntropy = entropy
	cd.diversityMetrics.DiversityScore = diversityScore
	cd.diversityMetrics.TotalTemplates = len(cd.contextTemplates)
	cd.diversityMetrics.ActiveTemplates = cd.countActiveTemplates()

	return cd.diversityMetrics
}

// PreventStagnation explicitly forces template rotation to break patterns.
func (cd *ContextDiversifier) PreventStagnation(ctx context.Context, content string) string {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	logger := logging.GetLogger()

	// Force selection of least recently used template
	leastUsedTemplate := cd.getLeastUsedTemplate()
	currentTemplate := cd.detectCurrentTemplate(content)

	if leastUsedTemplate.ID == currentTemplate.ID {
		// If already using least used, pick random different one
		leastUsedTemplate = cd.getRandomDifferentTemplate(currentTemplate)
	}

	// Transform content
	transformedContent, err := cd.transformContent(content, currentTemplate, leastUsedTemplate)
	if err != nil {
		logger.Warn(ctx, "Failed to transform content for stagnation prevention: %v", err)
		return content
	}

	cd.updateTemplateUsage(leastUsedTemplate.ID, true)
	cd.diversityMetrics.PreventedStagnation++

	logger.Debug(ctx, "Forced template rotation for stagnation prevention: %s -> %s",
		currentTemplate.ID, leastUsedTemplate.ID)

	return transformedContent
}

// AddCustomTemplate allows adding domain-specific templates.
func (cd *ContextDiversifier) AddCustomTemplate(template ContextTemplate) {
	cd.mu.Lock()
	defer cd.mu.Unlock()

	cd.contextTemplates = append(cd.contextTemplates, template)
}

// Private helper methods

func (cd *ContextDiversifier) initializeDefaultTemplates() {
	templates := []ContextTemplate{
		{
			ID:            "linear_formal",
			Name:          "Linear Formal",
			Structure:     StructureLinear,
			Style:         StyleFormal,
			Effectiveness: 0.8,
		},
		{
			ID:            "hierarchical_technical",
			Name:          "Hierarchical Technical",
			Structure:     StructureHierarchical,
			Style:         StyleTechnical,
			Effectiveness: 0.85,
		},
		{
			ID:            "bullets_casual",
			Name:          "Bullet Points Casual",
			Structure:     StructureBulletPoints,
			Style:         StyleCasual,
			Effectiveness: 0.75,
		},
		{
			ID:            "timeline_detailed",
			Name:          "Timeline Detailed",
			Structure:     StructureTimeline,
			Style:         StyleDetailed,
			Effectiveness: 0.7,
		},
		{
			ID:            "executive_minimalist",
			Name:          "Executive Minimalist",
			Structure:     StructureMatrix,
			Style:         StyleMinimalist,
			Effectiveness: 0.9,
		},
		{
			ID:            "narrative_casual",
			Name:          "Narrative Casual",
			Structure:     StructureNarrative,
			Style:         StyleCasual,
			Effectiveness: 0.65,
		},
	}

	cd.contextTemplates = templates

	// Initialize usage tracking
	for _, template := range templates {
		cd.usageHistory[template.ID] = ContextUsage{
			TemplateID:      template.ID,
			UsageCount:      0,
			SuccessRate:     template.Effectiveness,
			TokenEfficiency: 1.0,
		}
	}
}

func (cd *ContextDiversifier) detectCurrentTemplate(content string) ContextTemplate {
	// Simple heuristic-based template detection
	content = strings.ToLower(content)

	// Check for bullet points
	if strings.Count(content, "-") > 3 || strings.Count(content, "*") > 3 {
		return cd.getTemplateByID("bullets_casual")
	}

	// Check for formal language
	formalWords := []string{"therefore", "furthermore", "consequently", "moreover"}
	formalCount := 0
	for _, word := range formalWords {
		if strings.Contains(content, word) {
			formalCount++
		}
	}
	if formalCount >= 2 {
		return cd.getTemplateByID("linear_formal")
	}

	// Check for technical content
	techWords := []string{"function", "method", "class", "interface", "implementation"}
	techCount := 0
	for _, word := range techWords {
		if strings.Contains(content, word) {
			techCount++
		}
	}
	if techCount >= 2 {
		return cd.getTemplateByID("hierarchical_technical")
	}

	// Check for timeline indicators
	timeWords := []string{"first", "then", "next", "finally", "after", "before"}
	timeCount := 0
	for _, word := range timeWords {
		if strings.Contains(content, word) {
			timeCount++
		}
	}
	if timeCount >= 2 {
		return cd.getTemplateByID("timeline_detailed")
	}

	// Default to linear formal
	return cd.getTemplateByID("linear_formal")
}

func (cd *ContextDiversifier) shouldRotateTemplate(currentTemplate ContextTemplate) (bool, string) {
	usage, exists := cd.usageHistory[currentTemplate.ID]
	if !exists {
		return false, "no_usage_data"
	}

	// Check usage count
	if usage.UsageCount >= int64(cd.maxTemplateReuse) {
		return true, "max_reuse_exceeded"
	}

	// Check time since last usage
	if time.Since(usage.LastUsed) >= cd.rotationInterval {
		return true, "rotation_interval_reached"
	}

	// Check diversity threshold
	diversityScore := cd.calculateDiversityScore()
	if diversityScore < cd.diversityThreshold {
		return true, "diversity_threshold_breach"
	}

	return false, "no_rotation_needed"
}

func (cd *ContextDiversifier) selectDiverseTemplate(current ContextTemplate, contentType string) ContextTemplate {
	// Score all templates based on diversity benefit
	var bestTemplate ContextTemplate
	var bestScore float64 = -1

	for _, template := range cd.contextTemplates {
		if template.ID == current.ID {
			continue // Skip current template
		}

		score := cd.calculateDiversityScore() * template.Effectiveness

		// Bonus for least recently used
		usage := cd.usageHistory[template.ID]
		timeSinceUse := time.Since(usage.LastUsed).Hours()
		score += timeSinceUse / 24.0 // Bonus increases with days since last use

		// Bonus for structural/style differences
		if template.Structure != current.Structure {
			score += 0.3
		}
		if template.Style != current.Style {
			score += 0.2
		}

		if score > bestScore {
			bestScore = score
			bestTemplate = template
		}
	}

	return bestTemplate
}

func (cd *ContextDiversifier) transformContent(content string, from, to ContextTemplate) (string, error) {
	// This is a simplified transformation - in practice, you'd want more sophisticated
	// content transformation based on the specific template characteristics

	var transformed strings.Builder

	// Add template-specific header
	transformed.WriteString(cd.generateTemplateHeader(to))
	transformed.WriteString("\n\n")

	// Transform content based on target template structure
	switch to.Structure {
	case StructureBulletPoints:
		transformed.WriteString(cd.convertToBulletPoints(content))

	case StructureHierarchical:
		transformed.WriteString(cd.convertToHierarchical(content))

	case StructureTimeline:
		transformed.WriteString(cd.convertToTimeline(content))

	case StructureMatrix:
		transformed.WriteString(cd.convertToMatrix(content))

	case StructureNarrative:
		transformed.WriteString(cd.convertToNarrative(content, to.Style))

	default: // StructureLinear
		transformed.WriteString(cd.convertToLinear(content, to.Style))
	}

	return transformed.String(), nil
}

func (cd *ContextDiversifier) updateTemplateUsage(templateID string, success bool) {
	usage := cd.usageHistory[templateID]
	usage.UsageCount++
	usage.LastUsed = time.Now()

	if success {
		// Update success rate with exponential smoothing
		alpha := 0.1
		usage.SuccessRate = alpha*1.0 + (1-alpha)*usage.SuccessRate
	}

	cd.usageHistory[templateID] = usage
}

func (cd *ContextDiversifier) calculateTemplateEntropy() float64 {
	if len(cd.usageHistory) == 0 {
		return 0.0
	}

	var totalUsage int64
	for _, usage := range cd.usageHistory {
		totalUsage += usage.UsageCount
	}

	if totalUsage == 0 {
		return math.Log2(float64(len(cd.usageHistory))) // Maximum entropy
	}

	entropy := 0.0
	for _, usage := range cd.usageHistory {
		if usage.UsageCount > 0 {
			probability := float64(usage.UsageCount) / float64(totalUsage)
			entropy -= probability * math.Log2(probability)
		}
	}

	return entropy
}

func (cd *ContextDiversifier) calculateDiversityScore() float64 {
	maxEntropy := math.Log2(float64(len(cd.contextTemplates)))
	if maxEntropy == 0 {
		return 1.0
	}

	currentEntropy := cd.calculateTemplateEntropy()
	return currentEntropy / maxEntropy
}

func (cd *ContextDiversifier) countActiveTemplates() int {
	active := 0
	cutoff := time.Now().Add(-24 * time.Hour)

	for _, usage := range cd.usageHistory {
		if usage.LastUsed.After(cutoff) {
			active++
		}
	}

	return active
}

func (cd *ContextDiversifier) getLeastUsedTemplate() ContextTemplate {
	var leastUsed ContextTemplate
	var minUsage int64 = math.MaxInt64
	var oldestTime = time.Now()

	for _, template := range cd.contextTemplates {
		usage := cd.usageHistory[template.ID]
		if usage.UsageCount < minUsage ||
			(usage.UsageCount == minUsage && usage.LastUsed.Before(oldestTime)) {
			minUsage = usage.UsageCount
			oldestTime = usage.LastUsed
			leastUsed = template
		}
	}

	return leastUsed
}

func (cd *ContextDiversifier) getRandomDifferentTemplate(exclude ContextTemplate) ContextTemplate {
	var candidates []ContextTemplate
	for _, template := range cd.contextTemplates {
		if template.ID != exclude.ID {
			candidates = append(candidates, template)
		}
	}

	if len(candidates) == 0 {
		return exclude // Fallback
	}

	// Crypto-safe random selection
	randomIndex, _ := rand.Int(rand.Reader, big.NewInt(int64(len(candidates))))
	return candidates[randomIndex.Int64()]
}

func (cd *ContextDiversifier) getTemplateByID(id string) ContextTemplate {
	for _, template := range cd.contextTemplates {
		if template.ID == id {
			return template
		}
	}
	// Return first template as fallback
	if len(cd.contextTemplates) > 0 {
		return cd.contextTemplates[0]
	}
	return ContextTemplate{}
}

func (cd *ContextDiversifier) describeTransformation(from, to ContextTemplate) string {
	if from.Structure != to.Structure && from.Style != to.Style {
		return fmt.Sprintf("structure_%s_to_%s_style_%s_to_%s",
			from.Structure, to.Structure, from.Style, to.Style)
	} else if from.Structure != to.Structure {
		return fmt.Sprintf("structure_%s_to_%s", from.Structure, to.Structure)
	} else if from.Style != to.Style {
		return fmt.Sprintf("style_%s_to_%s", from.Style, to.Style)
	}
	return "minimal_variation"
}

func (cd *ContextDiversifier) calculateDiversityGain(from, to ContextTemplate) float64 {
	gain := 0.0

	if from.Structure != to.Structure {
		gain += 0.5
	}
	if from.Style != to.Style {
		gain += 0.3
	}

	// Factor in usage difference
	fromUsage := cd.usageHistory[from.ID].UsageCount
	toUsage := cd.usageHistory[to.ID].UsageCount

	if fromUsage > toUsage {
		gain += 0.2 * (float64(fromUsage-toUsage) / float64(fromUsage+1))
	}

	return math.Min(gain, 1.0)
}

func (cd *ContextDiversifier) estimateImpact(from, to ContextTemplate) string {
	gain := cd.calculateDiversityGain(from, to)

	if gain >= 0.7 {
		return "high_diversity_boost"
	} else if gain >= 0.4 {
		return "moderate_refresh"
	} else if gain >= 0.2 {
		return "subtle_variation"
	}
	return "minimal_change"
}

// Content transformation helpers - simplified implementations

func (cd *ContextDiversifier) generateTemplateHeader(template ContextTemplate) string {
	switch template.Style {
	case StyleFormal:
		return "# Context Information"
	case StyleCasual:
		return "## Here's what's happening:"
	case StyleTechnical:
		return "## Technical Context"
	case StyleExecutive:
		return "# Executive Summary"
	case StyleDetailed:
		return "## Detailed Context Analysis"
	case StyleMinimalist:
		return "## Context"
	default:
		return "## Context"
	}
}

func (cd *ContextDiversifier) convertToBulletPoints(content string) string {
	lines := strings.Split(content, "\n")
	var bullets []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && !strings.HasPrefix(trimmed, "-") && !strings.HasPrefix(trimmed, "*") {
			bullets = append(bullets, "- "+trimmed)
		} else if trimmed != "" {
			bullets = append(bullets, trimmed)
		}
	}

	return strings.Join(bullets, "\n")
}

func (cd *ContextDiversifier) convertToHierarchical(content string) string {
	// Simple hierarchical conversion
	lines := strings.Split(content, "\n")
	var hierarchical []string
	level := 0

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			indent := strings.Repeat("  ", level)
			hierarchical = append(hierarchical, indent+"- "+trimmed)
			if strings.Contains(trimmed, ":") {
				level = 1 // Increase indentation after headers
			}
		}
	}

	return strings.Join(hierarchical, "\n")
}

func (cd *ContextDiversifier) convertToTimeline(content string) string {
	lines := strings.Split(content, "\n")
	var timeline []string
	step := 1

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			timeline = append(timeline, fmt.Sprintf("%d. %s", step, trimmed))
			step++
		}
	}

	return strings.Join(timeline, "\n")
}

func (cd *ContextDiversifier) convertToMatrix(content string) string {
	// Simple matrix/table format
	return "| Context | Details |\n|---------|----------|\n" + content
}

func (cd *ContextDiversifier) convertToNarrative(content string, style PresentationStyle) string {
	connector := cd.getConnectorForStyle(style)
	lines := strings.Split(content, "\n")
	var narrative []string

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			if i == 0 {
				narrative = append(narrative, trimmed)
			} else {
				narrative = append(narrative, connector+" "+trimmed)
			}
		}
	}

	return strings.Join(narrative, " ")
}

func (cd *ContextDiversifier) convertToLinear(content string, style PresentationStyle) string {
	// Apply style-specific formatting to linear content
	switch style {
	case StyleFormal:
		return cd.applyFormalTone(content)
	case StyleCasual:
		return cd.applyCasualTone(content)
	case StyleTechnical:
		return cd.applyTechnicalTone(content)
	default:
		return content
	}
}

func (cd *ContextDiversifier) getConnectorForStyle(style PresentationStyle) string {
	connectors := map[PresentationStyle][]string{
		StyleFormal:    {"Furthermore", "Moreover", "Additionally", "Subsequently"},
		StyleCasual:    {"Also", "Plus", "And then", "Next up"},
		StyleTechnical: {"Then", "Next", "Following this", "Additionally"},
	}

	options := connectors[style]
	if len(options) == 0 {
		return "Additionally"
	}

	// Simple rotation through connectors
	randomIndex, _ := rand.Int(rand.Reader, big.NewInt(int64(len(options))))
	return options[randomIndex.Int64()]
}

func (cd *ContextDiversifier) applyFormalTone(content string) string {
	// Simple formal tone application
	content = strings.ReplaceAll(content, " can't ", " cannot ")
	content = strings.ReplaceAll(content, " won't ", " will not ")
	content = strings.ReplaceAll(content, " it's ", " it is ")
	return content
}

func (cd *ContextDiversifier) applyCasualTone(content string) string {
	// Simple casual tone application
	content = strings.ReplaceAll(content, " cannot ", " can't ")
	content = strings.ReplaceAll(content, " will not ", " won't ")
	content = strings.ReplaceAll(content, " it is ", " it's ")
	return content
}

func (cd *ContextDiversifier) applyTechnicalTone(content string) string {
	// Technical tone - more precise language
	content = strings.ReplaceAll(content, " uses ", " utilizes ")
	content = strings.ReplaceAll(content, " makes ", " generates ")
	content = strings.ReplaceAll(content, " shows ", " demonstrates ")
	return content
}
