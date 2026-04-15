package optimize

import (
	"context"
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

const (
	gepaProgramInputExampleIDKey = "__gepa_agent_example_id"
	gepaProgramOutputScoreKey    = "score"
)

var intArtifactInstructionPattern = regexp.MustCompile(`\b\d+\b`)

type artifactProgramKind int

const (
	artifactProgramKindText artifactProgramKind = iota
	artifactProgramKindInt
)

type artifactProgramSpec struct {
	moduleName string
	kind       artifactProgramKind
	textKey    ArtifactKey
	intKey     string
	intPlan    *IntMutationConfig
}

type artifactProgramAdapter struct {
	optimizer     *GEPAAgentOptimizer
	baseArtifacts AgentArtifacts
	specs         []artifactProgramSpec
	examples      map[string]AgentExample
}

type artifactProgramDataset struct {
	examples []core.Example
	index    int
}

type gepaArtifactModule struct {
	core.BaseModule
}

func newGEPAArtifactModule(name, instruction string) *gepaArtifactModule {
	return &gepaArtifactModule{
		BaseModule: core.BaseModule{
			Signature: core.Signature{
				Instruction: instruction,
				Inputs: []core.InputField{
					{Field: core.Field{Name: "artifact_input", Description: "Synthetic GEPA artifact input"}},
				},
				Outputs: []core.OutputField{
					{Field: core.Field{Name: "artifact_output", Description: "Synthetic GEPA artifact output"}},
				},
			},
			DisplayName: name,
			ModuleType:  "GEPAArtifactModule",
		},
	}
}

func (m *gepaArtifactModule) Process(context.Context, map[string]any, ...core.Option) (map[string]any, error) {
	return map[string]any{}, nil
}

func (m *gepaArtifactModule) Clone() core.Module {
	return &gepaArtifactModule{
		BaseModule: core.BaseModule{
			Signature:   m.Signature,
			LLM:         m.LLM,
			DisplayName: m.DisplayName,
			ModuleType:  m.ModuleType,
		},
	}
}

func (o *GEPAAgentOptimizer) buildOptimizationProgram(seed AgentArtifacts, examples []AgentExample) (core.Program, error) {
	specs, err := o.buildArtifactProgramSpecs(seed)
	if err != nil {
		return core.Program{}, err
	}

	modules := make(map[string]core.Module, len(specs))
	for _, spec := range specs {
		modules[spec.moduleName] = newGEPAArtifactModule(spec.moduleName, spec.instruction(seed))
	}

	adapter := &artifactProgramAdapter{
		optimizer:     o,
		baseArtifacts: seed.Clone(),
		specs:         specs,
		examples:      indexAgentExamples(examples),
	}

	return core.NewProgramWithForwardFactory(modules, adapter.forwardFactory()), nil
}

func (o *GEPAAgentOptimizer) buildOptimizationDataset(examples []AgentExample) core.Dataset {
	coreExamples := make([]core.Example, 0, len(examples))
	for idx, example := range examples {
		exampleID := strings.TrimSpace(example.ID)
		if exampleID == "" {
			exampleID = fmt.Sprintf("agent-example-%d", idx)
		}

		inputs := maps.Clone(example.Inputs)
		if inputs == nil {
			inputs = make(map[string]interface{}, 1)
		}
		inputs[gepaProgramInputExampleIDKey] = exampleID

		coreExamples = append(coreExamples, core.Example{
			Inputs:  inputs,
			Outputs: maps.Clone(example.Outputs),
		})
	}

	return &artifactProgramDataset{examples: coreExamples}
}

func (o *GEPAAgentOptimizer) buildOptimizationMetric() core.Metric {
	return func(expected, actual map[string]interface{}) float64 {
		if actual == nil {
			return 0
		}
		score, err := metricScoreValue(actual[gepaProgramOutputScoreKey])
		if err != nil {
			return 0
		}
		return score
	}
}

func (o *GEPAAgentOptimizer) buildArtifactProgramSpecs(seed AgentArtifacts) ([]artifactProgramSpec, error) {
	textKeys := o.config.resolveArtifactKeys(seed)
	intPlans := o.config.normalizedIntMutationPlans()
	intKeys := sortedIntArtifactKeys(intPlans)
	if len(textKeys) == 0 && len(intKeys) == 0 {
		return nil, fmt.Errorf("optimize: seed artifacts must include at least one optimizable artifact")
	}

	specs := make([]artifactProgramSpec, 0, len(textKeys)+len(intKeys))
	for _, key := range textKeys {
		specs = append(specs, artifactProgramSpec{
			moduleName: string(key),
			kind:       artifactProgramKindText,
			textKey:    key,
		})
	}
	for _, key := range intKeys {
		plan := intPlans[key]
		planCopy := plan
		specs = append(specs, artifactProgramSpec{
			moduleName: intArtifactModuleName(key),
			kind:       artifactProgramKindInt,
			intKey:     key,
			intPlan:    &planCopy,
		})
	}

	sort.Slice(specs, func(i, j int) bool { return specs[i].moduleName < specs[j].moduleName })
	return specs, nil
}

func (a *artifactProgramAdapter) forwardFactory() core.ForwardFactory {
	return func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			artifacts, err := a.artifactsFromModules(modules)
			if err != nil {
				return nil, err
			}

			example, err := a.lookupExample(inputs)
			if err != nil {
				return nil, err
			}

			result := a.optimizer.evaluateArtifactsOnExample(ctx, artifacts, example)
			candidate := a.candidateForArtifacts(ctx, artifacts)
			a.recordEvaluation(ctx, candidate, example, result)

			return buildProgramOutputs(result), nil
		}
	}
}

func (a *artifactProgramAdapter) artifactsFromModules(modules map[string]core.Module) (AgentArtifacts, error) {
	artifacts := a.baseArtifacts.Clone()

	for _, spec := range a.specs {
		module, exists := modules[spec.moduleName]
		if !exists {
			return AgentArtifacts{}, fmt.Errorf("optimize: GEPA program missing module %q", spec.moduleName)
		}

		instruction := module.GetSignature().Instruction
		switch spec.kind {
		case artifactProgramKindText:
			artifacts.Text[spec.textKey] = instruction
		case artifactProgramKindInt:
			artifacts.Int[spec.intKey] = parseIntArtifactInstruction(spec.intKey, instruction, artifacts.Int[spec.intKey], spec.intPlan)
		default:
			return AgentArtifacts{}, fmt.Errorf("optimize: unsupported artifact module kind %d", spec.kind)
		}
	}

	return artifacts, nil
}

func (a *artifactProgramAdapter) lookupExample(inputs map[string]interface{}) (AgentExample, error) {
	if inputs == nil {
		return AgentExample{}, fmt.Errorf("optimize: missing GEPA program inputs")
	}

	exampleID, _ := inputs[gepaProgramInputExampleIDKey].(string)
	if strings.TrimSpace(exampleID) == "" {
		return AgentExample{}, fmt.Errorf("optimize: missing GEPA agent example id in program inputs")
	}

	example, exists := a.examples[exampleID]
	if !exists {
		return AgentExample{}, fmt.Errorf("optimize: unknown GEPA agent example id %q", exampleID)
	}
	return cloneAgentExample(example), nil
}

func (a *artifactProgramAdapter) candidateForArtifacts(ctx context.Context, artifacts AgentArtifacts) *optimizers.GEPACandidate {
	componentTexts := make(map[string]string, len(a.specs))
	for _, spec := range a.specs {
		componentTexts[spec.moduleName] = spec.instruction(artifacts)
	}

	primary, err := a.optimizer.config.primaryArtifact(artifacts)
	if err != nil {
		primary = a.optimizer.config.PrimaryArtifact
	}

	metadata := map[string]interface{}{
		gepaMetadataArtifactsKey:       serializeArtifacts(artifacts),
		gepaMetadataArtifactKeysKey:    serializeArtifactKeys(a.optimizer.config.resolveArtifactKeys(artifacts)),
		gepaMetadataIntArtifactKeysKey: sortedIntArtifactKeys(a.optimizer.config.normalizedIntMutationPlans()),
		gepaMetadataPrimaryArtifactKey: string(primary),
	}
	if targetIDs := a.optimizer.optimizationTargetIDs(artifacts); len(targetIDs) > 0 {
		metadata[gepaMetadataOptimizationTargetsKey] = maps.Clone(targetIDs)
		if primaryTargetID := optimizationTargetIDForModule(string(primary), targetIDs); primaryTargetID != "" {
			metadata[gepaMetadataOptimizationTargetKey] = primaryTargetID
		}
	}

	return &optimizers.GEPACandidate{
		ID:             optimizers.CurrentCandidateID(ctx),
		ModuleName:     string(primary),
		Instruction:    artifacts.Text[primary],
		ComponentTexts: componentTexts,
		Metadata:       metadata,
	}
}

func (a *artifactProgramAdapter) recordEvaluation(ctx context.Context, candidate *optimizers.GEPACandidate, example AgentExample, result *EvalResult) {
	if candidate == nil || result == nil {
		return
	}

	state := optimizers.GetGEPAState(ctx)
	if state == nil {
		return
	}

	trace := buildGEPATrace(candidate, example, result, a.optimizer.config.PassThreshold)
	state.AddTrace(&trace)
	state.RecordCandidateCaseObservation(candidate.ID, optimizersCandidateCaseObservation(result, a.optimizer.config.PassThreshold))
}

func (o *GEPAAgentOptimizer) evaluateArtifactsOnExample(ctx context.Context, artifacts AgentArtifacts, example AgentExample) *EvalResult {
	agent, err := o.MaterializeAgent(artifacts)
	if err != nil {
		return evaluationFailureResult(err)
	}

	result, err := o.evaluator.Evaluate(ctx, agent, example)
	if err != nil {
		return evaluationFailureResult(err)
	}
	if result == nil {
		return evaluationFailureResult(fmt.Errorf("nil evaluation result"))
	}
	return result
}

func buildProgramOutputs(result *EvalResult) map[string]interface{} {
	outputs := map[string]interface{}{
		gepaProgramOutputScoreKey: 0.0,
	}
	if result == nil {
		return outputs
	}

	outputs[gepaProgramOutputScoreKey] = result.Score
	if result.SideInfo == nil {
		return outputs
	}

	if len(result.SideInfo.FailedTests) > 0 {
		outputs["failed_tests"] = append([]string(nil), result.SideInfo.FailedTests...)
	}
	if len(result.SideInfo.PassedTests) > 0 {
		outputs["passed_tests"] = append([]string(nil), result.SideInfo.PassedTests...)
	}
	if toolCalls := toolCallCount(result.SideInfo); toolCalls >= 0 {
		outputs["tool_calls"] = toolCalls
	}
	if result.SideInfo.LatencyMS > 0 {
		outputs["latency_ms"] = result.SideInfo.LatencyMS
	}
	if totalTokens := tokenCount(result.SideInfo.Tokens); totalTokens > 0 {
		outputs["total_tokens"] = totalTokens
	}
	if result.SideInfo.Diagnostics != nil {
		if answer, ok := result.SideInfo.Diagnostics["final_answer"].(string); ok && strings.TrimSpace(answer) != "" {
			outputs["final_answer"] = answer
		}
		if fraction, ok := result.SideInfo.Diagnostics["verifier_pass_fraction"]; ok {
			outputs["verifier_pass_fraction"] = fraction
		}
		if summary := diagnosticString(result.SideInfo.Diagnostics, "execution_error", "evaluation_error", "comparison_error"); summary != "" {
			outputs["error"] = summary
		}
	}
	if result.SideInfo.Trace != nil {
		outputs["trace_status"] = string(result.SideInfo.Trace.Status)
		outputs["termination"] = result.SideInfo.Trace.TerminationCause
		outputs["trace_summary"] = summarizeAgentTrace(result.SideInfo.Trace)
		if evidence := buildRichTraceEvidence(result.SideInfo.Trace, result.SideInfo); len(evidence) > 0 {
			outputs["trace_evidence"] = evidence
		}
	}
	return outputs
}

func (s artifactProgramSpec) instruction(artifacts AgentArtifacts) string {
	switch s.kind {
	case artifactProgramKindText:
		return artifacts.Text[s.textKey]
	case artifactProgramKindInt:
		return formatIntArtifactInstruction(s.intKey, artifacts.Int[s.intKey], s.intPlan)
	default:
		return ""
	}
}

func intArtifactModuleName(key string) string {
	return "__int__:" + strings.TrimSpace(key)
}

func formatIntArtifactInstruction(key string, value int, plan *IntMutationConfig) string {
	if value <= 0 && plan != nil {
		value = plan.Min
	}
	if value <= 0 {
		value = 1
	}
	return fmt.Sprintf("Set %s to %d.", key, value)
}

func parseIntArtifactInstruction(key, instruction string, fallback int, plan *IntMutationConfig) int {
	value := fallback
	if value <= 0 && plan != nil {
		value = plan.Min
	}
	if value <= 0 {
		value = 1
	}

	match := intArtifactInstructionPattern.FindString(strings.TrimSpace(instruction))
	if match == "" {
		return clampIntArtifactValue(value, plan)
	}

	parsed, err := toInt(match)
	if err != nil {
		return clampIntArtifactValue(value, plan)
	}
	return clampIntArtifactValue(parsed, plan)
}

func clampIntArtifactValue(value int, plan *IntMutationConfig) int {
	if plan == nil {
		if value <= 0 {
			return 1
		}
		return value
	}
	if value < plan.Min {
		value = plan.Min
	}
	if value > plan.Max {
		value = plan.Max
	}
	return value
}

func sortedIntArtifactKeys(plans map[string]IntMutationConfig) []string {
	if len(plans) == 0 {
		return nil
	}

	keys := make([]string, 0, len(plans))
	for key := range plans {
		if strings.TrimSpace(key) == "" {
			continue
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func indexAgentExamples(examples []AgentExample) map[string]AgentExample {
	indexed := make(map[string]AgentExample, len(examples))
	for idx, example := range examples {
		exampleID := strings.TrimSpace(example.ID)
		if exampleID == "" {
			exampleID = fmt.Sprintf("agent-example-%d", idx)
		}
		example.ID = exampleID
		indexed[exampleID] = cloneAgentExample(example)
	}
	return indexed
}

func cloneAgentExample(example AgentExample) AgentExample {
	return AgentExample{
		ID:       example.ID,
		Inputs:   maps.Clone(example.Inputs),
		Outputs:  maps.Clone(example.Outputs),
		Metadata: maps.Clone(example.Metadata),
	}
}

func optimizersCandidateCaseObservation(result *EvalResult, passThreshold float64) optimizers.CandidateCaseObservation {
	observation := optimizers.CandidateCaseObservation{
		ToolCalls: -1,
	}
	if result == nil {
		observation.RobustnessFailure = true
		return observation
	}

	observation.Score = result.Score
	observation.Passed = result.Score >= passThreshold
	if result.SideInfo == nil {
		observation.RobustnessFailure = true
		return observation
	}

	observation.LatencyMS = result.SideInfo.LatencyMS
	observation.TotalTokens = tokenCount(result.SideInfo.Tokens)
	observation.ToolCalls = toolCallCount(result.SideInfo)
	if diagnostics := result.SideInfo.Diagnostics; diagnostics != nil {
		if diagnostics["evaluation_error"] != nil || diagnostics["execution_error"] != nil || diagnostics["comparison_error"] != nil {
			observation.RobustnessFailure = true
		}
	}
	return observation
}

func diagnosticString(diagnostics map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if value, ok := diagnostics[key].(string); ok && strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func metricScoreValue(raw interface{}) (float64, error) {
	switch value := raw.(type) {
	case float64:
		return value, nil
	case float32:
		return float64(value), nil
	case int:
		return float64(value), nil
	case int32:
		return float64(value), nil
	case int64:
		return float64(value), nil
	default:
		return 0, fmt.Errorf("unsupported score type %T", raw)
	}
}

func (d *artifactProgramDataset) Next() (core.Example, bool) {
	if d == nil || d.index >= len(d.examples) {
		return core.Example{}, false
	}
	example := d.examples[d.index]
	d.index++
	return example, true
}

func (d *artifactProgramDataset) Reset() {
	if d == nil {
		return
	}
	d.index = 0
}
