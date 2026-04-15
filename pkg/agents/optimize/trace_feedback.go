package optimize

import (
	"context"
	"fmt"
	"maps"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

const (
	maxTraceFeedbackEvidence = 3
	maxTraceFeedbackChars    = 160
)

func (o *GEPAAgentOptimizer) optimizationTargetIDs(artifacts AgentArtifacts) map[string]string {
	if o == nil || o.baseAgent == nil {
		return nil
	}

	descriptors, _ := optimizationTargetsForAgent(o.baseAgent, artifacts)
	if len(descriptors) == 0 {
		return nil
	}

	targetIDs := make(map[string]string, len(descriptors))
	for _, descriptor := range descriptors {
		switch descriptor.Kind {
		case OptimizationTargetText:
			if descriptor.ArtifactKey != "" {
				targetIDs[string(descriptor.ArtifactKey)] = descriptor.ID
			}
		case OptimizationTargetInt:
			if descriptor.IntKey != "" {
				targetIDs[intArtifactModuleName(descriptor.IntKey)] = descriptor.ID
			}
		}
	}
	if len(targetIDs) == 0 {
		return nil
	}
	return targetIDs
}

func optimizationTargetIDForModule(moduleName string, targetIDs map[string]string) string {
	moduleName = strings.TrimSpace(moduleName)
	if moduleName == "" || len(targetIDs) == 0 {
		return ""
	}
	return strings.TrimSpace(targetIDs[moduleName])
}

func (o *GEPAAgentOptimizer) defaultFeedbackEvaluator(seed AgentArtifacts) optimizers.GEPAFeedbackEvaluator {
	targetIDs := o.optimizationTargetIDs(seed)
	if len(targetIDs) == 0 {
		return nil
	}

	return optimizers.GEPAFeedbackEvaluatorFunc(func(ctx context.Context, expected, actual map[string]interface{}, info *optimizers.GEPAFeedbackContext) *optimizers.GEPAFeedback {
		return deterministicTraceFeedback(ctx, targetIDs, expected, actual, info)
	})
}

func deterministicTraceFeedback(_ context.Context, targetIDs map[string]string, _ map[string]interface{}, actual map[string]interface{}, info *optimizers.GEPAFeedbackContext) *optimizers.GEPAFeedback {
	if info == nil {
		return nil
	}

	target := feedbackTargetComponent(info.Candidate, targetIDs)
	score, scoreErr := metricScoreValue(actual[gepaProgramOutputScoreKey])
	errorMessage := firstNonEmpty(
		stringValue(actual["error"]),
		errString(info.Err),
	)
	traceStatus := stringValue(actual["trace_status"])
	termination := stringValue(actual["termination"])
	failedTests := stringSliceValue(actual["failed_tests"])
	traceSummary := stringValue(actual["trace_summary"])
	traceEvidence := stringSliceValue(actual["trace_evidence"])

	if scoreErr != nil && errorMessage == "" && traceStatus == "" && termination == "" && len(failedTests) == 0 && traceSummary == "" && len(traceEvidence) == 0 {
		return nil
	}
	if scoreErr != nil {
		score = 0
	}
	if score >= 1.0 && errorMessage == "" && len(failedTests) == 0 && (traceStatus == "" || traceStatus == string(agents.TraceStatusSuccess)) {
		return nil
	}

	lines := make([]string, 0, 6)
	lines = append(lines, fmt.Sprintf("score=%.2f", score))
	if errorMessage != "" {
		lines = append(lines, "issue="+truncateTraceFeedback(errorMessage))
	}
	if traceStatus != "" && traceStatus != string(agents.TraceStatusSuccess) {
		lines = append(lines, "trace_status="+traceStatus)
	}
	if termination != "" {
		lines = append(lines, "termination="+termination)
	}
	if len(failedTests) > 0 {
		lines = append(lines, "failed_tests="+strings.Join(limitStrings(failedTests, maxTraceFeedbackEvidence), ", "))
	}
	if fraction, ok := numericValue(actual["verifier_pass_fraction"]); ok && fraction < 1 {
		lines = append(lines, fmt.Sprintf("verifier_pass_fraction=%.2f", fraction))
	}
	if len(traceEvidence) > 0 {
		lines = append(lines, "evidence="+strings.Join(limitStrings(traceEvidence, maxTraceFeedbackEvidence), "; "))
	} else if traceSummary != "" {
		lines = append(lines, "trace="+truncateTraceFeedback(traceSummary))
	}

	metadata := map[string]interface{}{
		"source": "deterministic_trace_feedback",
	}
	if target != "" {
		metadata["optimization_target_id"] = target
	}
	if traceStatus != "" {
		metadata["trace_status"] = traceStatus
	}
	if len(failedTests) > 0 {
		metadata["failed_tests"] = append([]string(nil), failedTests...)
	}

	return &optimizers.GEPAFeedback{
		Feedback:        strings.Join(lines, "\n"),
		TargetComponent: target,
		Metadata:        metadata,
	}
}

func feedbackTargetComponent(candidate *optimizers.GEPACandidate, fallback map[string]string) string {
	if candidate == nil {
		return ""
	}
	if target := stringValue(metadataValue(candidate.Metadata, gepaMetadataOptimizationTargetKey)); target != "" {
		return target
	}
	if target := optimizationTargetIDForModule(candidate.ModuleName, stringMapValue(metadataValue(candidate.Metadata, gepaMetadataOptimizationTargetsKey))); target != "" {
		return target
	}
	if target := optimizationTargetIDForModule(candidate.ModuleName, fallback); target != "" {
		return target
	}
	return strings.TrimSpace(candidate.ModuleName)
}

func metadataValue(metadata map[string]interface{}, key string) interface{} {
	if len(metadata) == 0 {
		return nil
	}
	return metadata[key]
}

func stringMapValue(raw interface{}) map[string]string {
	switch value := raw.(type) {
	case map[string]string:
		return maps.Clone(value)
	case map[string]interface{}:
		converted := make(map[string]string, len(value))
		for key, entry := range value {
			if text := stringValue(entry); text != "" {
				converted[key] = text
			}
		}
		if len(converted) > 0 {
			return converted
		}
	}
	return nil
}

func stringValue(raw interface{}) string {
	if text, ok := raw.(string); ok {
		return strings.TrimSpace(text)
	}
	return ""
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return strings.TrimSpace(err.Error())
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func numericValue(raw interface{}) (float64, bool) {
	value, err := metricScoreValue(raw)
	if err != nil {
		return 0, false
	}
	return value, true
}

func stringSliceValue(raw interface{}) []string {
	switch value := raw.(type) {
	case []string:
		return cleanStrings(value)
	case []interface{}:
		items := make([]string, 0, len(value))
		for _, item := range value {
			if text := stringValue(item); text != "" {
				items = append(items, text)
			}
		}
		return cleanStrings(items)
	default:
		return nil
	}
}

func cleanStrings(values []string) []string {
	items := make([]string, 0, len(values))
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			items = append(items, text)
		}
	}
	if len(items) == 0 {
		return nil
	}
	return items
}

func limitStrings(values []string, limit int) []string {
	if len(values) <= limit || limit <= 0 {
		return values
	}
	return append([]string(nil), values[:limit]...)
}

func truncateTraceFeedback(value string) string {
	value = strings.TrimSpace(value)
	if len(value) <= maxTraceFeedbackChars {
		return value
	}
	return strings.TrimSpace(value[:maxTraceFeedbackChars-3]) + "..."
}
