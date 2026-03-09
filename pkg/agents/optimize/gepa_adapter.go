package optimize

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

const (
	gepaMetadataArtifactsKey       = "artifacts"
	gepaMetadataArtifactKeysKey    = "artifact_keys"
	gepaMetadataPrimaryArtifactKey = "primary_artifact"
	gepaMetadataTraceSummaryKey    = "rich_trace_summary"
	gepaMetadataTraceEvidenceKey   = "rich_trace_evidence"
	maxTraceSummarySteps           = 5
	maxTraceEvidenceItems          = 12
	maxFailedStepEvidence          = 4
)

// GEPAAdapterConfig configures the agent-to-GEPA bridge layer.
type GEPAAdapterConfig struct {
	PopulationSize  int
	MaxGenerations  int
	ReflectionFreq  int
	ValidationSplit float64
	ArtifactKeys    []ArtifactKey
	EvalConcurrency int
	PassThreshold   float64
	PrimaryArtifact ArtifactKey
}

// DefaultGEPAAdapterConfig returns a conservative default adapter config.
func DefaultGEPAAdapterConfig() GEPAAdapterConfig {
	return GEPAAdapterConfig{
		PopulationSize:  8,
		MaxGenerations:  4,
		ReflectionFreq:  1,
		ValidationSplit: 0.2,
		EvalConcurrency: 1,
		PassThreshold:   1.0,
		PrimaryArtifact: ArtifactSkillPack,
	}
}

// GEPAAgentOptimizer holds the shared translation logic between agent evaluation
// and GEPA's candidate, trace, and fitness types.
type GEPAAgentOptimizer struct {
	config    GEPAAdapterConfig
	evaluator AgentEvaluator
	baseAgent OptimizableAgent
	factory   func(AgentArtifacts) (OptimizableAgent, error)
}

// GEPACandidateEvaluation captures the GEPA-shaped output of evaluating one agent candidate.
type GEPACandidateEvaluation struct {
	Candidate    *optimizers.GEPACandidate
	Artifacts    AgentArtifacts
	Run          *HarnessRunResult
	Fitness      *optimizers.MultiObjectiveFitness
	Traces       []optimizers.ExecutionTrace
	AverageScore float64
}

// NewGEPAAgentOptimizer creates a new adapter scaffold around an optimizable agent and evaluator.
func NewGEPAAgentOptimizer(baseAgent OptimizableAgent, evaluator AgentEvaluator, cfg GEPAAdapterConfig) *GEPAAgentOptimizer {
	cfg = cfg.withDefaults()
	return &GEPAAgentOptimizer{
		config:    cfg,
		evaluator: evaluator,
		baseAgent: baseAgent,
	}
}

// WithFactory registers a fallback constructor used when clone-based materialization is unavailable.
func (o *GEPAAgentOptimizer) WithFactory(factory func(AgentArtifacts) (OptimizableAgent, error)) *GEPAAgentOptimizer {
	if o == nil {
		return nil
	}
	o.factory = factory
	return o
}

// SeedCandidate encodes an artifact set into a GEPA candidate record.
func (o *GEPAAgentOptimizer) SeedCandidate(seed AgentArtifacts) (*optimizers.GEPACandidate, error) {
	primary, err := o.config.primaryArtifact(seed)
	if err != nil {
		return nil, err
	}

	artifactKeys := o.config.resolveArtifactKeys(seed)
	candidate := &optimizers.GEPACandidate{
		ID:          fmt.Sprintf("agent-candidate-%d", time.Now().UnixNano()),
		ModuleName:  string(primary),
		Instruction: seed.Text[primary],
		Generation:  0,
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			gepaMetadataArtifactsKey:       serializeArtifacts(seed),
			gepaMetadataArtifactKeysKey:    serializeArtifactKeys(artifactKeys),
			gepaMetadataPrimaryArtifactKey: string(primary),
		},
	}

	return candidate, nil
}

// CandidateArtifacts reconstructs the full artifact set represented by a GEPA candidate.
func (o *GEPAAgentOptimizer) CandidateArtifacts(candidate *optimizers.GEPACandidate) (AgentArtifacts, error) {
	if candidate == nil {
		return AgentArtifacts{}, fmt.Errorf("optimize: nil GEPA candidate")
	}

	artifacts := AgentArtifacts{
		Text: make(map[ArtifactKey]string),
		Int:  make(map[string]int),
		Bool: make(map[string]bool),
	}

	if candidate.Metadata != nil {
		if rawArtifacts, ok := candidate.Metadata[gepaMetadataArtifactsKey]; ok {
			decoded, err := deserializeArtifacts(rawArtifacts)
			if err != nil {
				return AgentArtifacts{}, err
			}
			artifacts = decoded
		}
	}

	primary := ArtifactKey(candidate.ModuleName)
	if primary == "" {
		primary = o.config.PrimaryArtifact
	}
	if artifacts.Text == nil {
		artifacts.Text = make(map[ArtifactKey]string)
	}
	artifacts.Text[primary] = candidate.Instruction

	return artifacts, nil
}

// MaterializeAgent creates a concrete agent instance for the provided artifact set.
func (o *GEPAAgentOptimizer) MaterializeAgent(artifacts AgentArtifacts) (OptimizableAgent, error) {
	if o == nil {
		return nil, fmt.Errorf("optimize: nil GEPA agent optimizer")
	}
	if o.baseAgent != nil {
		cloned, err := o.baseAgent.Clone()
		if err == nil {
			if err := cloned.SetArtifacts(artifacts); err != nil {
				return nil, fmt.Errorf("optimize: apply candidate artifacts to cloned agent: %w", err)
			}
			return cloned, nil
		}
		if o.factory == nil {
			return nil, fmt.Errorf("optimize: clone base agent: %w", err)
		}
	}
	if o.factory == nil {
		return nil, fmt.Errorf("optimize: no base agent or factory available")
	}

	agent, err := o.factory(artifacts.Clone())
	if err != nil {
		return nil, fmt.Errorf("optimize: construct agent from factory: %w", err)
	}
	return agent, nil
}

// EvaluateCandidate runs the adapter's evaluator over examples and converts the result
// into GEPA fitness and trace records.
func (o *GEPAAgentOptimizer) EvaluateCandidate(ctx context.Context, candidate *optimizers.GEPACandidate, examples []AgentExample) (*GEPACandidateEvaluation, error) {
	if o == nil {
		return nil, fmt.Errorf("optimize: nil GEPA agent optimizer")
	}
	if o.evaluator == nil {
		return nil, fmt.Errorf("optimize: nil agent evaluator")
	}
	if candidate == nil {
		return nil, fmt.Errorf("optimize: nil GEPA candidate")
	}

	artifacts, err := o.CandidateArtifacts(candidate)
	if err != nil {
		return nil, err
	}

	agent, err := o.MaterializeAgent(artifacts)
	if err != nil {
		return nil, err
	}

	harness := &Harness{
		Evaluator:     o.evaluator,
		PassThreshold: o.config.PassThreshold,
	}

	run, err := harness.Run(ctx, agent, examples)
	if err != nil {
		return nil, err
	}

	fitness := buildMultiObjectiveFitness(run)
	candidate.Fitness = fitness.WeightedScore

	traces := make([]optimizers.ExecutionTrace, 0, len(run.Results))
	for idx, result := range run.Results {
		if idx >= len(examples) {
			break
		}
		traces = append(traces, buildGEPATrace(candidate, examples[idx], result.Result, harness.PassThreshold))
	}

	return &GEPACandidateEvaluation{
		Candidate:    candidate,
		Artifacts:    artifacts,
		Run:          run,
		Fitness:      fitness,
		Traces:       traces,
		AverageScore: run.AverageScore,
	}, nil
}

func (cfg GEPAAdapterConfig) withDefaults() GEPAAdapterConfig {
	defaults := DefaultGEPAAdapterConfig()
	if cfg.PopulationSize <= 0 {
		cfg.PopulationSize = defaults.PopulationSize
	}
	if cfg.MaxGenerations <= 0 {
		cfg.MaxGenerations = defaults.MaxGenerations
	}
	if cfg.ReflectionFreq < 0 {
		cfg.ReflectionFreq = defaults.ReflectionFreq
	}
	if cfg.ValidationSplit <= 0 {
		cfg.ValidationSplit = defaults.ValidationSplit
	}
	if cfg.EvalConcurrency <= 0 {
		cfg.EvalConcurrency = defaults.EvalConcurrency
	}
	if cfg.PassThreshold <= 0 {
		cfg.PassThreshold = defaults.PassThreshold
	}
	if cfg.PrimaryArtifact == "" {
		cfg.PrimaryArtifact = defaults.PrimaryArtifact
	}
	return cfg
}

func (cfg GEPAAdapterConfig) resolveArtifactKeys(seed AgentArtifacts) []ArtifactKey {
	if len(cfg.ArtifactKeys) > 0 {
		keys := append([]ArtifactKey(nil), cfg.ArtifactKeys...)
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		return keys
	}

	keys := make([]ArtifactKey, 0, len(seed.Text))
	for key, value := range seed.Text {
		if strings.TrimSpace(value) != "" {
			keys = append(keys, key)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func (cfg GEPAAdapterConfig) primaryArtifact(seed AgentArtifacts) (ArtifactKey, error) {
	cfg = cfg.withDefaults()
	if cfg.PrimaryArtifact != "" && strings.TrimSpace(seed.Text[cfg.PrimaryArtifact]) != "" {
		return cfg.PrimaryArtifact, nil
	}

	keys := cfg.resolveArtifactKeys(seed)
	if len(keys) == 0 {
		return "", fmt.Errorf("optimize: seed artifacts must include at least one non-empty text artifact")
	}
	return keys[0], nil
}

func serializeArtifacts(artifacts AgentArtifacts) map[string]interface{} {
	text := make(map[string]string, len(artifacts.Text))
	for key, value := range artifacts.Text {
		text[string(key)] = value
	}

	return map[string]interface{}{
		"text": text,
		"int":  core.ShallowCopyMap(artifacts.Int),
		"bool": core.ShallowCopyMap(artifacts.Bool),
	}
}

func deserializeArtifacts(raw interface{}) (AgentArtifacts, error) {
	switch value := raw.(type) {
	case AgentArtifacts:
		return value.Clone(), nil
	case map[string]interface{}:
		artifacts := AgentArtifacts{
			Text: make(map[ArtifactKey]string),
			Int:  make(map[string]int),
			Bool: make(map[string]bool),
		}

		if rawText, ok := value["text"]; ok {
			text, err := decodeTextArtifacts(rawText)
			if err != nil {
				return AgentArtifacts{}, err
			}
			artifacts.Text = text
		}
		if rawInt, ok := value["int"]; ok {
			ints, err := decodeIntArtifacts(rawInt)
			if err != nil {
				return AgentArtifacts{}, err
			}
			artifacts.Int = ints
		}
		if rawBool, ok := value["bool"]; ok {
			bools, err := decodeBoolArtifacts(rawBool)
			if err != nil {
				return AgentArtifacts{}, err
			}
			artifacts.Bool = bools
		}
		return artifacts, nil
	default:
		return AgentArtifacts{}, fmt.Errorf("optimize: unsupported artifact metadata type %T", raw)
	}
}

func decodeTextArtifacts(raw interface{}) (map[ArtifactKey]string, error) {
	switch value := raw.(type) {
	case map[ArtifactKey]string:
		text := make(map[ArtifactKey]string, len(value))
		for key, v := range value {
			text[key] = v
		}
		return text, nil
	case map[string]string:
		text := make(map[ArtifactKey]string, len(value))
		for key, v := range value {
			text[ArtifactKey(key)] = v
		}
		return text, nil
	case map[string]interface{}:
		text := make(map[ArtifactKey]string, len(value))
		for key, v := range value {
			str, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("optimize: artifact text value for %q is %T, want string", key, v)
			}
			text[ArtifactKey(key)] = str
		}
		return text, nil
	default:
		return nil, fmt.Errorf("optimize: unsupported text artifact type %T", raw)
	}
}

func decodeIntArtifacts(raw interface{}) (map[string]int, error) {
	switch value := raw.(type) {
	case map[string]int:
		return core.ShallowCopyMap(value), nil
	case map[string]interface{}:
		ints := make(map[string]int, len(value))
		for key, v := range value {
			n, err := toInt(v)
			if err != nil {
				return nil, fmt.Errorf("optimize: decode int artifact %q: %w", key, err)
			}
			ints[key] = n
		}
		return ints, nil
	default:
		return nil, fmt.Errorf("optimize: unsupported int artifact type %T", raw)
	}
}

func decodeBoolArtifacts(raw interface{}) (map[string]bool, error) {
	switch value := raw.(type) {
	case map[string]bool:
		return core.ShallowCopyMap(value), nil
	case map[string]interface{}:
		bools := make(map[string]bool, len(value))
		for key, v := range value {
			b, ok := v.(bool)
			if !ok {
				return nil, fmt.Errorf("optimize: artifact bool value for %q is %T, want bool", key, v)
			}
			bools[key] = b
		}
		return bools, nil
	default:
		return nil, fmt.Errorf("optimize: unsupported bool artifact type %T", raw)
	}
}

func toInt(v interface{}) (int, error) {
	switch n := v.(type) {
	case int:
		return n, nil
	case int32:
		return int(n), nil
	case int64:
		return int(n), nil
	case float64:
		return int(n), nil
	case string:
		parsed, err := strconv.Atoi(n)
		if err != nil {
			return 0, err
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("unsupported integer type %T", v)
	}
}

func serializeArtifactKeys(keys []ArtifactKey) []string {
	if len(keys) == 0 {
		return nil
	}
	serialized := make([]string, 0, len(keys))
	for _, key := range keys {
		serialized = append(serialized, string(key))
	}
	return serialized
}

func buildMultiObjectiveFitness(run *HarnessRunResult) *optimizers.MultiObjectiveFitness {
	fitness := &optimizers.MultiObjectiveFitness{}
	if run == nil || run.CompletedExamples == 0 {
		return fitness
	}

	total := float64(run.CompletedExamples)
	fitness.SuccessRate = float64(run.PassedExamples) / total
	fitness.OutputQuality = clamp01(run.AverageScore)
	fitness.Efficiency = clamp01(averageEfficiency(run.Results))
	fitness.Robustness = clamp01(1.0 - float64(countRobustnessFailures(run.Results))/total)
	fitness.Generalization = clamp01(scoreConsistency(run.Results))
	// These objectives are intentionally neutral placeholders until the adapter
	// computes real cross-candidate diversity and novelty signals.
	fitness.Diversity = 0.5
	fitness.Innovation = 0.5
	fitness.WeightedScore = fitness.ComputeWeightedScore(optimizers.DefaultMultiObjectiveWeights())

	return fitness
}

func averageEfficiency(results []HarnessExampleResult) float64 {
	if len(results) == 0 {
		return 0
	}

	total := 0.0
	for _, result := range results {
		total += singleResultEfficiency(result.Result)
	}
	return total / float64(len(results))
}

func singleResultEfficiency(result *EvalResult) float64 {
	if result == nil || result.SideInfo == nil {
		return 0
	}

	components := make([]float64, 0, 3)
	if result.SideInfo.LatencyMS >= 0 {
		components = append(components, 1.0/(1.0+result.SideInfo.LatencyMS/1000.0))
	}

	if tokens := tokenCount(result.SideInfo.Tokens); tokens > 0 {
		components = append(components, 1.0/(1.0+float64(tokens)/1000.0))
	}

	if calls := toolCallCount(result.SideInfo); calls >= 0 {
		components = append(components, 1.0/(1.0+float64(calls)))
	}

	if len(components) == 0 {
		return 1.0
	}

	sum := 0.0
	for _, component := range components {
		sum += component
	}
	return sum / float64(len(components))
}

func tokenCount(tokens map[string]int64) int64 {
	var total int64
	for _, count := range tokens {
		total += count
	}
	return total
}

func toolCallCount(sideInfo *SideInfo) int {
	if sideInfo == nil {
		return -1
	}
	if sideInfo.Trace != nil {
		total := 0
		for _, count := range sideInfo.Trace.ToolUsageCount {
			total += count
		}
		return total
	}
	if raw, ok := sideInfo.Diagnostics["tool_usage_count"]; ok {
		switch value := raw.(type) {
		case map[string]int:
			total := 0
			for _, count := range value {
				total += count
			}
			return total
		case map[string]interface{}:
			total := 0
			for _, count := range value {
				n, err := toInt(count)
				if err == nil {
					total += n
				}
			}
			return total
		}
	}
	return -1
}

func countRobustnessFailures(results []HarnessExampleResult) int {
	failures := 0
	for _, result := range results {
		if result.Result == nil || result.Result.SideInfo == nil {
			failures++
			continue
		}
		diagnostics := result.Result.SideInfo.Diagnostics
		if diagnostics == nil {
			continue
		}
		if diagnostics["evaluation_error"] != nil || diagnostics["execution_error"] != nil || diagnostics["comparison_error"] != nil {
			failures++
		}
	}
	return failures
}

func scoreConsistency(results []HarnessExampleResult) float64 {
	if len(results) == 0 {
		return 0
	}
	if len(results) == 1 {
		if results[0].Result == nil {
			return 0
		}
		return clamp01(results[0].Result.Score)
	}

	mean := 0.0
	for _, result := range results {
		if result.Result != nil {
			mean += result.Result.Score
		}
	}
	mean /= float64(len(results))

	variance := 0.0
	for _, result := range results {
		score := 0.0
		if result.Result != nil {
			score = result.Result.Score
		}
		diff := score - mean
		variance += diff * diff
	}
	variance /= float64(len(results))
	return clamp01(1.0 - variance)
}

func buildGEPATrace(candidate *optimizers.GEPACandidate, example AgentExample, result *EvalResult, passThreshold float64) optimizers.ExecutionTrace {
	trace := optimizers.ExecutionTrace{
		CandidateID: candidate.ID,
		ModuleName:  candidate.ModuleName,
		Inputs:      core.ShallowCopyMap(example.Inputs),
		Outputs:     nil,
		Success:     false,
		Timestamp:   time.Now(),
		ContextData: map[string]interface{}{
			"example_id": example.ID,
		},
	}

	if result == nil || result.SideInfo == nil {
		trace.Error = fmt.Errorf("missing evaluation result")
		return trace
	}

	sideInfo := result.SideInfo
	if sideInfo.Trace != nil {
		trace.Inputs = core.ShallowCopyMap(sideInfo.Trace.Input)
		trace.Outputs = core.ShallowCopyMap(sideInfo.Trace.Output)
		trace.Duration = sideInfo.Trace.ProcessingTime
		if !sideInfo.Trace.CompletedAt.IsZero() {
			trace.Timestamp = sideInfo.Trace.CompletedAt
		}
		trace.ContextData["agent_type"] = sideInfo.Trace.AgentType
		trace.ContextData["agent_id"] = sideInfo.Trace.AgentID
		trace.ContextData["task"] = sideInfo.Trace.Task
		trace.ContextData["trace_status"] = string(sideInfo.Trace.Status)
		trace.ContextData["termination_cause"] = sideInfo.Trace.TerminationCause
		trace.ContextData["step_count"] = len(sideInfo.Trace.Steps)
		trace.ContextData["token_usage"] = core.ShallowCopyMap(sideInfo.Trace.TokenUsage)
		trace.ContextData["tool_usage_count"] = core.ShallowCopyMap(sideInfo.Trace.ToolUsageCount)
		trace.ContextData[gepaMetadataTraceSummaryKey] = summarizeAgentTrace(sideInfo.Trace)
	}
	if example.Outputs != nil {
		trace.ContextData["expected_outputs"] = core.ShallowCopyMap(example.Outputs)
	}
	if trace.Duration == 0 && sideInfo.LatencyMS > 0 {
		trace.Duration = time.Duration(sideInfo.LatencyMS * float64(time.Millisecond))
	}

	trace.ContextData["scores"] = core.ShallowCopyMap(sideInfo.Scores)
	trace.ContextData["diagnostics"] = core.ShallowCopyMap(sideInfo.Diagnostics)
	trace.ContextData["passed_tests"] = append([]string(nil), sideInfo.PassedTests...)
	trace.ContextData["failed_tests"] = append([]string(nil), sideInfo.FailedTests...)
	trace.ContextData["tokens"] = core.ShallowCopyMap(sideInfo.Tokens)
	if evidence := buildRichTraceEvidence(sideInfo.Trace, sideInfo); len(evidence) > 0 {
		trace.ContextData[gepaMetadataTraceEvidenceKey] = evidence
	}

	evalErr := extractEvalError(sideInfo)
	trace.Success = result.Score >= passThreshold && evalErr == nil
	trace.Error = evalErr
	return trace
}

func extractEvalError(sideInfo *SideInfo) error {
	if sideInfo == nil || sideInfo.Diagnostics == nil {
		return nil
	}
	for _, key := range []string{"evaluation_error", "execution_error", "comparison_error"} {
		if raw, ok := sideInfo.Diagnostics[key]; ok {
			if message, ok := raw.(string); ok && message != "" {
				return fmt.Errorf("%s", message)
			}
		}
	}
	if sideInfo.Trace != nil && sideInfo.Trace.Error != "" {
		return fmt.Errorf("%s", sideInfo.Trace.Error)
	}
	return nil
}

func summarizeAgentTrace(trace *agents.ExecutionTrace) string {
	if trace == nil {
		return ""
	}

	var builder strings.Builder
	if trace.Task != "" {
		builder.WriteString("task=")
		builder.WriteString(truncateString(trace.Task, 120))
	}
	if trace.Status != "" {
		if builder.Len() > 0 {
			builder.WriteString("; ")
		}
		builder.WriteString("status=")
		builder.WriteString(string(trace.Status))
	}
	if trace.TerminationCause != "" {
		if builder.Len() > 0 {
			builder.WriteString("; ")
		}
		builder.WriteString("termination=")
		builder.WriteString(trace.TerminationCause)
	}
	if trace.ContextMetadata != nil {
		appendSummaryCount(&builder, "subllm", trace.ContextMetadata[modrlm.TraceMetadataSubLLMCallCount])
		appendSummaryCount(&builder, "subrlm", trace.ContextMetadata[modrlm.TraceMetadataSubRLMCallCount])
		appendSummaryCount(&builder, "signals", trace.ContextMetadata[modrlm.TraceMetadataConfidenceSignals])
		appendSummaryCount(&builder, "compressions", trace.ContextMetadata[modrlm.TraceMetadataHistoryCompressions])
	}

	stepLimit := len(trace.Steps)
	if stepLimit > maxTraceSummarySteps {
		stepLimit = maxTraceSummarySteps
	}
	for i := 0; i < stepLimit; i++ {
		step := trace.Steps[i]
		builder.WriteString("; step")
		builder.WriteString(strconv.Itoa(step.Index))
		builder.WriteString("[tool=")
		builder.WriteString(step.Tool)
		builder.WriteString(",success=")
		builder.WriteString(strconv.FormatBool(step.Success))
		builder.WriteString("]")
		if step.Error != "" {
			builder.WriteString(" error=")
			builder.WriteString(truncateString(step.Error, 80))
		} else if step.Observation != "" {
			builder.WriteString(" obs=")
			builder.WriteString(truncateString(step.Observation, 80))
		}
	}

	return builder.String()
}

func buildRichTraceEvidence(trace *agents.ExecutionTrace, sideInfo *SideInfo) []string {
	evidence := make([]string, 0, maxTraceEvidenceItems)

	if trace != nil {
		if trace.TerminationCause != "" {
			evidence = append(evidence, "termination="+trace.TerminationCause)
		}
		if trace.Status != "" {
			evidence = append(evidence, "trace_status="+string(trace.Status))
		}
		if len(trace.Steps) > 0 {
			evidence = append(evidence, fmt.Sprintf("step_count=%d", len(trace.Steps)))
			evidence = append(evidence, detectToolLoopEvidence(trace.Steps)...)
			failedStepCount := 0
			for _, step := range trace.Steps {
				if step.Success && step.Error == "" {
					continue
				}
				if failedStepCount >= maxFailedStepEvidence {
					break
				}
				entry := fmt.Sprintf("step%d tool=%s success=%t", step.Index, step.Tool, step.Success)
				if step.Error != "" {
					entry += " error=" + truncateString(step.Error, 80)
				} else if step.Observation != "" {
					entry += " obs=" + truncateString(step.Observation, 80)
				}
				evidence = append(evidence, entry)
				failedStepCount++
			}
		}
		evidence = append(evidence, contextMetadataEvidence(trace.ContextMetadata)...)
	}

	if sideInfo != nil {
		for _, failedTest := range sideInfo.FailedTests {
			evidence = append(evidence, "failed_test="+failedTest)
		}
		if sideInfo.Diagnostics != nil {
			for _, key := range []string{"evaluation_error", "execution_error", "comparison_error"} {
				if raw, ok := sideInfo.Diagnostics[key]; ok {
					if message, ok := raw.(string); ok && message != "" {
						evidence = append(evidence, key+"="+truncateString(message, 80))
					}
				}
			}
			if mismatches, ok := sideInfo.Diagnostics["mismatches"].(map[string]interface{}); ok && len(mismatches) > 0 {
				keys := make([]string, 0, len(mismatches))
				for key := range mismatches {
					keys = append(keys, key)
				}
				sort.Strings(keys)
				evidence = append(evidence, "mismatch_keys="+strings.Join(keys, ","))
			}
		}
	}

	return utils.DedupeStrings(evidence, maxTraceEvidenceItems)
}

func contextMetadataEvidence(metadata map[string]interface{}) []string {
	if len(metadata) == 0 {
		return nil
	}

	evidence := make([]string, 0, 8)
	if enabled, ok := metadata[modrlm.TraceMetadataAdaptiveIterationEnabled].(bool); ok && enabled {
		evidence = append(evidence, "adaptive_iteration=enabled")
	}
	if value := nonNegativeIntEvidence("sub_llm_calls", metadata[modrlm.TraceMetadataSubLLMCallCount]); value != "" {
		evidence = append(evidence, value)
	}
	if value := nonNegativeIntEvidence("sub_rlm_calls", metadata[modrlm.TraceMetadataSubRLMCallCount]); value != "" {
		evidence = append(evidence, value)
	}
	if value := nonNegativeIntEvidence("confidence_signals", metadata[modrlm.TraceMetadataConfidenceSignals]); value != "" {
		evidence = append(evidence, value)
	}
	if value := nonNegativeIntEvidence("history_compressions", metadata[modrlm.TraceMetadataHistoryCompressions]); value != "" {
		evidence = append(evidence, value)
	}
	if value := nonNegativeIntEvidence("root_prompt_mean_tokens", metadata[modrlm.TraceMetadataRootPromptMeanTokens]); value != "" {
		evidence = append(evidence, value)
	}
	if value := nonNegativeIntEvidence("root_prompt_max_tokens", metadata[modrlm.TraceMetadataRootPromptMaxTokens]); value != "" {
		evidence = append(evidence, value)
	}
	if base, ok := extractPositiveInt(metadata[modrlm.TraceMetadataAdaptiveBaseIterations]); ok {
		if maxIterations, ok := extractPositiveInt(metadata[modrlm.TraceMetadataAdaptiveMaxIterations]); ok {
			evidence = append(evidence, fmt.Sprintf("adaptive_window=%d/%d", base, maxIterations))
		}
	}
	if threshold, ok := extractPositiveInt(metadata[modrlm.TraceMetadataAdaptiveConfidenceThreshold]); ok {
		evidence = append(evidence, fmt.Sprintf("adaptive_confidence_threshold=%d", threshold))
	}

	return evidence
}

func appendSummaryCount(builder *strings.Builder, label string, raw interface{}) {
	if builder == nil {
		return
	}
	value, ok := extractNonNegativeInt(raw)
	if !ok {
		return
	}
	if builder.Len() > 0 {
		builder.WriteString("; ")
	}
	builder.WriteString(label)
	builder.WriteString("=")
	builder.WriteString(strconv.Itoa(value))
}

func nonNegativeIntEvidence(label string, raw interface{}) string {
	value, ok := extractNonNegativeInt(raw)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%s=%d", label, value)
}

func extractNonNegativeInt(raw interface{}) (int, bool) {
	switch value := raw.(type) {
	case int:
		if value >= 0 {
			return value, true
		}
	case int64:
		if value >= 0 {
			return int(value), true
		}
	case float64:
		if value >= 0 {
			return int(value), true
		}
	}
	return 0, false
}

func extractPositiveInt(raw interface{}) (int, bool) {
	switch value := raw.(type) {
	case int:
		if value > 0 {
			return value, true
		}
	case int64:
		if value > 0 {
			return int(value), true
		}
	case float64:
		if value > 0 {
			return int(value), true
		}
	}
	return 0, false
}

// detectToolLoopEvidence intentionally captures only consecutive same-tool runs.
// More complex alternating loops are possible, but this cheap heuristic already
// surfaces a common failure pattern without full sequence mining.
func detectToolLoopEvidence(steps []agents.TraceStep) []string {
	if len(steps) < 2 {
		return nil
	}

	evidence := make([]string, 0, 2)
	currentTool := ""
	runLength := 0

	flush := func() {
		if currentTool != "" && runLength >= 2 {
			evidence = append(evidence, fmt.Sprintf("tool_loop=%sx%d", currentTool, runLength))
		}
	}

	for _, step := range steps {
		if step.Tool == "" {
			flush()
			currentTool = ""
			runLength = 0
			continue
		}
		if step.Tool != currentTool {
			flush()
			currentTool = step.Tool
			runLength = 1
			continue
		}
		runLength++
	}
	flush()

	return evidence
}

func truncateString(value string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
}

func clamp01(value float64) float64 {
	return math.Max(0, math.Min(1, value))
}
