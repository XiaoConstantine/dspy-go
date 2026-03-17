package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

const gepaFixtureOutputEnv = "GEPA_FIXTURE_OUTPUT"

type compatFixtureModule struct {
	signature core.Signature
}

func newCompatFixtureModule(instruction string) *compatFixtureModule {
	return &compatFixtureModule{
		signature: core.Signature{
			Instruction: instruction,
			Inputs: []core.InputField{
				{Field: core.Field{Name: "input", Description: "Fixture input"}},
			},
			Outputs: []core.OutputField{
				{Field: core.Field{Name: "output", Description: "Fixture output"}},
			},
		},
	}
}

func (m *compatFixtureModule) Process(context.Context, map[string]any, ...core.Option) (map[string]any, error) {
	return map[string]any{"output": m.signature.Instruction}, nil
}

func (m *compatFixtureModule) GetSignature() core.Signature { return m.signature }

func (m *compatFixtureModule) SetSignature(signature core.Signature) { m.signature = signature }

func (m *compatFixtureModule) SetLLM(core.LLM) {}

func (m *compatFixtureModule) Clone() core.Module {
	return &compatFixtureModule{signature: m.signature}
}

func (m *compatFixtureModule) GetDisplayName() string { return "CompatFixtureModule" }

func (m *compatFixtureModule) GetModuleType() string { return "fixture" }

type compatFixtureLLM struct{}

func (f *compatFixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, "Generate 1 diverse variations of the instruction for a classifier module."):
		return &core.LLMResponse{Content: "1. classifier improved"}, nil
	case strings.Contains(prompt, "Generate 1 diverse variations of the instruction for a generator module."):
		return &core.LLMResponse{Content: "1. generator improved"}, nil
	case strings.Contains(prompt, `Original: "classifier base"`):
		return &core.LLMResponse{Content: "classifier improved"}, nil
	case strings.Contains(prompt, `Original: "generator base"`):
		return &core.LLMResponse{Content: "generator improved"}, nil
	default:
		return &core.LLMResponse{Content: "fixture fallback response"}, nil
	}
}

func (f *compatFixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *compatFixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *compatFixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *compatFixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFixtureLLM) ProviderName() string { return "fixture" }

func (f *compatFixtureLLM) ModelID() string { return "fixture-gepa" }

func (f *compatFixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type compatFeedbackFixtureLLM struct{}

func (f *compatFeedbackFixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, "Generate 1 diverse variations of the instruction for a classifier module."):
		return &core.LLMResponse{Content: "1. classifier base"}, nil
	case strings.Contains(prompt, "As an expert prompt engineer, critically analyze this instruction"):
		return &core.LLMResponse{
			Content: `STRENGTHS:
- Clear task framing

WEAKNESSES:
- Uses the wrong terminology

SUGGESTIONS:
- Use classifier terminology exactly

CONFIDENCE: 0.9`,
		}, nil
	case strings.Contains(prompt, "You are improving a GEPA instruction using reflection-guided evidence.") &&
		strings.Contains(prompt, "Use classifier terminology exactly"):
		return &core.LLMResponse{Content: "```feedback tuned classifier instruction```"}, nil
	default:
		return &core.LLMResponse{Content: "classifier base"}, nil
	}
}

func (f *compatFeedbackFixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFeedbackFixtureLLM) ProviderName() string { return "fixture" }

func (f *compatFeedbackFixtureLLM) ModelID() string { return "fixture-gepa-feedback" }

func (f *compatFeedbackFixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type compatFormatFixtureLLM struct{}

func (f *compatFormatFixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, "Generate 1 diverse variations of the instruction for a classifier module."):
		return &core.LLMResponse{Content: "1. classifier base"}, nil
	case strings.Contains(prompt, "As an expert prompt engineer, critically analyze this instruction"):
		return &core.LLMResponse{
			Content: `STRENGTHS:
- Clear task framing

WEAKNESSES:
- Output format is invalid

SUGGESTIONS:
- Fix the response format so the category field parses cleanly

CONFIDENCE: 0.9`,
		}, nil
	case strings.Contains(prompt, "You are improving a GEPA instruction using reflection-guided evidence.") &&
		strings.Contains(prompt, "Execution failure: couldn't parse category output"):
		// This fixture keys off dspy-go's current format-failure feedback text.
		// The companion Python fixture matches DSPy's built-in wording instead,
		// so the parity target here is the rewritten instruction, not identical
		// feedback strings across implementations.
		return &core.LLMResponse{Content: "```format tuned classifier instruction```"}, nil
	default:
		return &core.LLMResponse{Content: "classifier base"}, nil
	}
}

func (f *compatFormatFixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatFormatFixtureLLM) ProviderName() string { return "fixture" }

func (f *compatFormatFixtureLLM) ModelID() string { return "fixture-gepa-format" }

func (f *compatFormatFixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type compatResumeFixtureLLM struct{}

func (f *compatResumeFixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, "classifier better"):
		return &core.LLMResponse{Content: "classifier best"}, nil
	case strings.Contains(prompt, "classifier base"):
		return &core.LLMResponse{Content: "classifier better"}, nil
	case strings.Contains(prompt, "Generate 1 diverse variations of the instruction for a classifier module."):
		return &core.LLMResponse{Content: "classifier better"}, nil
	default:
		return &core.LLMResponse{Content: "classifier better"}, nil
	}
}

func (f *compatResumeFixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatResumeFixtureLLM) ProviderName() string { return "fixture" }

func (f *compatResumeFixtureLLM) ModelID() string { return "fixture-gepa-resume" }

func (f *compatResumeFixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type compatSingleInstructionLLM struct {
	proposal string
	modelID  string
}

func (f *compatSingleInstructionLLM) Generate(_ context.Context, _ string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: f.proposal}, nil
}

func (f *compatSingleInstructionLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *compatSingleInstructionLLM) ProviderName() string { return "fixture" }

func (f *compatSingleInstructionLLM) ModelID() string {
	if f.modelID != "" {
		return f.modelID
	}
	return "fixture-gepa-single-instruction"
}

func (f *compatSingleInstructionLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type compatFixtureScenarioResult struct {
	Selector                        string            `json:"selector"`
	FirstCandidateUpdatedComponents []string          `json:"first_candidate_updated_components"`
	FirstCandidateInstructions      map[string]string `json:"first_candidate_instructions"`
	FinalProgramUpdatedComponents   []string          `json:"final_program_updated_components"`
	FinalProgramInstructions        map[string]string `json:"final_program_instructions"`
	CandidateCount                  int               `json:"candidate_count"`
}

type compatComponentSelectionReport struct {
	Scenarios map[string]compatFixtureScenarioResult `json:"scenarios"`
}

type compatValidationFrontierResult struct {
	FrontierWinnerLabelsByExample []string       `json:"frontier_winner_labels_by_example"`
	FrontierCoverageLabels        map[string]int `json:"frontier_coverage_labels"`
	CandidateCount                int            `json:"candidate_count"`
}

type compatMergeResult struct {
	MergedCandidatePresent           bool     `json:"merged_candidate_present"`
	MergedCandidateUpdatedComponents []string `json:"merged_candidate_updated_components"`
	MergedCandidateParentLabels      []string `json:"merged_candidate_parent_labels"`
	MergedCandidateParentCount       int      `json:"merged_candidate_parent_count"`
}

type compatFeedbackResult struct {
	OriginalInstruction     string `json:"original_instruction"`
	CandidateInstruction    string `json:"candidate_instruction"`
	FinalProgramInstruction string `json:"final_program_instruction"`
	CandidateCount          int    `json:"candidate_count"`
}

type compatResumeResult struct {
	CheckpointWritten              bool   `json:"checkpoint_written"`
	StoppedMetricCalls             int    `json:"stopped_metric_calls"`
	ResumedMetricCalls             int    `json:"resumed_metric_calls"`
	FreshMetricCalls               int    `json:"fresh_metric_calls"`
	ResumedCandidateCount          int    `json:"resumed_candidate_count"`
	FreshCandidateCount            int    `json:"fresh_candidate_count"`
	ResumedFinalProgramInstruction string `json:"resumed_final_program_instruction"`
	FreshFinalProgramInstruction   string `json:"fresh_final_program_instruction"`
}

type compatMinibatchScenarioResult struct {
	CandidateCount              int    `json:"candidate_count"`
	CandidateAdded              bool   `json:"candidate_added"`
	FinalProgramInstruction     string `json:"final_program_instruction"`
	WinningCandidateInstruction string `json:"winning_candidate_instruction"`
}

type compatMinibatchAcceptanceResult struct {
	AcceptedCase compatMinibatchScenarioResult `json:"accepted_case"`
	RejectedCase compatMinibatchScenarioResult `json:"rejected_case"`
}

type compatFixtureCollection struct {
	ComponentSelection  compatComponentSelectionReport  `json:"component_selection"`
	ValidationFrontier  compatValidationFrontierResult  `json:"validation_frontier"`
	AncestorMerge       compatMergeResult               `json:"ancestor_merge"`
	FeedbackGuided      compatFeedbackResult            `json:"feedback_guided"`
	FormatFailure       compatFeedbackResult            `json:"format_failure_feedback"`
	MinibatchAcceptance compatMinibatchAcceptanceResult `json:"minibatch_acceptance"`
	ResumeParity        compatResumeResult              `json:"resume_parity"`
}

type compatFixtureReport struct {
	Runner   string                  `json:"runner"`
	Fixtures compatFixtureCollection `json:"fixtures"`
}

func newCompatFixtureProgram(classifierInstruction, generatorInstruction string) core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"classifier": newCompatFixtureModule(classifierInstruction),
		"generator":  newCompatFixtureModule(generatorInstruction),
	}, func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			classifierInstruction := modules["classifier"].GetSignature().Instruction
			generatorInstruction := modules["generator"].GetSignature().Instruction
			return map[string]interface{}{
				"output":                 classifierInstruction + "|" + generatorInstruction,
				"classifier_instruction": classifierInstruction,
				"generator_instruction":  generatorInstruction,
			}, nil
		}
	})
}

func newCompatFeedbackFixtureProgram(classifierInstruction string) core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"classifier": newCompatFixtureModule(classifierInstruction),
	}, func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{
				"output": modules["classifier"].GetSignature().Instruction,
			}, nil
		}
	})
}

func newCompatFormatFixtureProgram(classifierInstruction string) core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"classifier": newCompatFixtureModule(classifierInstruction),
	}, func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			instruction := modules["classifier"].GetSignature().Instruction
			if strings.Contains(instruction, "format tuned classifier instruction") {
				return map[string]interface{}{"output": "format tuned classifier instruction"}, nil
			}
			return map[string]interface{}{"output": "not json at all"}, fmt.Errorf("couldn't parse category output")
		}
	})
}

func compatInstructionProgressMetric(_ map[string]interface{}, actual map[string]interface{}) float64 {
	output, _ := actual["output"].(string)
	score := 0.0
	for _, part := range strings.Split(output, "|") {
		if strings.Contains(part, "improved") {
			score += 0.5
		}
	}
	return score
}

func compatFrontierProgressMetric(expected, actual map[string]interface{}) float64 {
	kind, _ := expected["kind"].(string)
	classifierInstruction, _ := actual["classifier_instruction"].(string)
	generatorInstruction, _ := actual["generator_instruction"].(string)
	classifierImproved := strings.Contains(strings.ToLower(classifierInstruction), "improved")
	generatorImproved := strings.Contains(strings.ToLower(generatorInstruction), "improved")

	switch kind {
	case "train":
		return compatBoolToFloat(classifierImproved != generatorImproved)
	case "alpha":
		if classifierImproved && !generatorImproved {
			return 1.0
		}
		if !classifierImproved && !generatorImproved {
			return 0.5
		}
		return 0.0
	case "beta":
		if generatorImproved && !classifierImproved {
			return 1.0
		}
		if !classifierImproved && !generatorImproved {
			return 0.5
		}
		return 0.0
	default:
		return 0.0
	}
}

func compatBoolToFloat(value bool) float64 {
	if value {
		return 1.0
	}
	return 0.0
}

func compatResumeProgressMetric(_ map[string]interface{}, actual map[string]interface{}) float64 {
	output, _ := actual["output"].(string)
	lowered := strings.ToLower(output)
	switch {
	case strings.Contains(lowered, "classifier best"):
		return 1.0
	case strings.Contains(lowered, "classifier better"):
		return 0.5
	default:
		return 0.0
	}
}

func compatProgramInstructions(program core.Program) map[string]string {
	instructions := make(map[string]string, len(program.Modules))
	for moduleName, module := range program.Modules {
		instructions[moduleName] = module.GetSignature().Instruction
	}
	return instructions
}

func compatUpdatedComponents(original, current map[string]string) []string {
	updated := make([]string, 0)
	for moduleName, instruction := range current {
		if original[moduleName] != instruction {
			updated = append(updated, moduleName)
		}
	}
	sort.Strings(updated)
	return updated
}

func compatCandidateInstructions(candidate *GEPACandidate) map[string]string {
	if candidate == nil {
		return nil
	}

	instructions := make(map[string]string, len(candidate.ComponentTexts))
	for moduleName, instruction := range candidate.ComponentTexts {
		instructions[moduleName] = instruction
	}
	if len(instructions) > 0 {
		return instructions
	}
	if candidate.ModuleName != "" {
		instructions[candidate.ModuleName] = candidate.Instruction
	}
	return instructions
}

func compatCandidateLabel(instructions, original map[string]string) string {
	classifierImproved := instructions["classifier"] != original["classifier"]
	generatorImproved := instructions["generator"] != original["generator"]

	switch {
	case classifierImproved && !generatorImproved:
		return "classifier"
	case generatorImproved && !classifierImproved:
		return "generator"
	case classifierImproved && generatorImproved:
		return "merged"
	default:
		return "seed"
	}
}

func runCompatSelectorScenario(t *testing.T, ctx context.Context, selector string) compatFixtureScenarioResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatFixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := DefaultGEPAConfig()
	config.PopulationSize = 1
	config.MaxGenerations = 2
	config.MutationRate = 1.0
	config.ReflectionFreq = 0
	config.ComponentSelection = selector
	config.SelectionStrategy = "tournament"
	config.TournamentSize = 1
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 1

	gepa, err := NewGEPA(config)
	if err != nil {
		t.Fatalf("new GEPA: %v", err)
	}

	originalInstructions := map[string]string{
		"classifier": "classifier base",
		"generator":  "generator base",
	}
	program := newCompatFixtureProgram(originalInstructions["classifier"], originalInstructions["generator"])
	dataset := datasets.NewSimpleDataset([]core.Example{{Outputs: map[string]interface{}{"output": "fixture"}}})

	optimizedProgram, err := gepa.Compile(ctx, program, dataset, compatInstructionProgressMetric)
	if err != nil {
		t.Fatalf("compile selector fixture: %v", err)
	}

	population := gepa.CurrentPopulation()
	if population == nil || len(population.Candidates) == 0 {
		t.Fatalf("missing GEPA population after compile")
	}

	firstCandidate := population.Candidates[0]
	firstCandidateInstructions := compatCandidateInstructions(firstCandidate)
	finalProgramInstructions := compatProgramInstructions(optimizedProgram)
	return compatFixtureScenarioResult{
		Selector:                        selector,
		FirstCandidateUpdatedComponents: compatUpdatedComponents(originalInstructions, firstCandidateInstructions),
		FirstCandidateInstructions:      firstCandidateInstructions,
		FinalProgramUpdatedComponents:   compatUpdatedComponents(originalInstructions, finalProgramInstructions),
		FinalProgramInstructions:        finalProgramInstructions,
		CandidateCount:                  len(population.Candidates),
	}
}

func runCompatValidationFrontierScenario(t *testing.T, ctx context.Context) compatValidationFrontierResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatFixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := DefaultGEPAConfig()
	config.PopulationSize = 4
	config.MaxGenerations = 1
	config.ReflectionFreq = 0
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 1
	config.ValidationSplit = 0.8
	config.RandomSeed = 7

	gepa, err := NewGEPA(config)
	if err != nil {
		t.Fatalf("new GEPA: %v", err)
	}

	originalInstructions := map[string]string{
		"classifier": "classifier base",
		"generator":  "generator base",
	}
	program := newCompatFixtureProgram(originalInstructions["classifier"], originalInstructions["generator"])
	dataset := datasets.NewSimpleDataset([]core.Example{
		{Inputs: map[string]interface{}{"input": "train"}, Outputs: map[string]interface{}{"kind": "train"}},
		{Inputs: map[string]interface{}{"input": "alpha-1"}, Outputs: map[string]interface{}{"kind": "alpha"}},
		{Inputs: map[string]interface{}{"input": "alpha-2"}, Outputs: map[string]interface{}{"kind": "alpha"}},
		{Inputs: map[string]interface{}{"input": "beta-1"}, Outputs: map[string]interface{}{"kind": "beta"}},
		{Inputs: map[string]interface{}{"input": "beta-2"}, Outputs: map[string]interface{}{"kind": "beta"}},
	})

	if _, err := gepa.Compile(ctx, program, dataset, compatFrontierProgressMetric); err != nil {
		t.Fatalf("compile validation frontier fixture: %v", err)
	}

	population := gepa.CurrentPopulation()
	if population == nil || len(population.Candidates) == 0 {
		t.Fatalf("missing GEPA population after compile")
	}

	candidateStates := make(map[string]map[string]string, len(population.Candidates))
	for _, candidate := range population.Candidates {
		candidateStates[candidate.ID] = compatCandidateInstructions(candidate)
	}

	frontier, _ := gepa.GetOptimizationState().ValidationFrontierSnapshot()
	if len(frontier) == 0 {
		t.Fatalf("missing validation frontier after compile")
	}

	winnersByKind := make(map[string]string)
	for _, entry := range frontier {
		if entry == nil {
			continue
		}
		kind, _ := entry.Example.Outputs["kind"].(string)
		if kind == "" {
			continue
		}
		instructions := candidateStates[entry.CandidateID]
		if len(instructions) == 0 {
			continue
		}
		winnersByKind[kind] = compatCandidateLabel(instructions, originalInstructions)
	}

	winnerLabels := make([]string, 0, 2)
	coverage := make(map[string]int)
	for _, kind := range []string{"alpha", "beta"} {
		label, ok := winnersByKind[kind]
		if !ok {
			continue
		}
		winnerLabels = append(winnerLabels, label)
		coverage[label]++
	}

	return compatValidationFrontierResult{
		FrontierWinnerLabelsByExample: winnerLabels,
		FrontierCoverageLabels:        coverage,
		CandidateCount:                len(population.Candidates),
	}
}

func newCompatMergeFixtureProgram(classifierInstruction, generatorInstruction string) core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"classifier": newStaticCandidateTestModule(classifierInstruction),
		"generator":  newStaticCandidateTestModule(generatorInstruction),
	}, func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{
				"output": modules["classifier"].GetSignature().Instruction + "|" + modules["generator"].GetSignature().Instruction,
			}, nil
		}
	})
}

func runCompatMergeScenario(t *testing.T, ctx context.Context) compatMergeResult {
	t.Helper()

	generationLLM := &countingLLM{}
	gepa := &GEPA{
		config:        DefaultGEPAConfig(),
		state:         NewGEPAState(),
		generationLLM: generationLLM,
		rng:           rand.New(rand.NewSource(9)),
	}

	root := &GEPACandidate{
		ID:          "root",
		ModuleName:  "classifier",
		Instruction: "classifier base",
		ComponentTexts: map[string]string{
			"classifier": "classifier base",
			"generator":  "generator base",
		},
	}
	source := &GEPACandidate{
		ID:          "source",
		ModuleName:  "classifier",
		Instruction: "classifier improved",
		ComponentTexts: map[string]string{
			"classifier": "classifier improved",
			"generator":  "generator base",
		},
		ParentIDs: []string{"root"},
	}
	partner := &GEPACandidate{
		ID:          "partner",
		ModuleName:  "generator",
		Instruction: "generator improved",
		ComponentTexts: map[string]string{
			"classifier": "classifier base",
			"generator":  "generator improved",
		},
		ParentIDs: []string{"root"},
	}
	current := &Population{
		Generation: 1,
		Candidates: []*GEPACandidate{source, partner},
	}
	gepa.state.PopulationHistory = []*Population{
		{Generation: 0, Candidates: []*GEPACandidate{root}},
		current,
	}
	gepa.state.SetValidationFrontier(nil, map[string]int{
		"source":  2,
		"partner": 2,
	})

	dataset := newCountingDataset([]core.Example{
		{Outputs: map[string]interface{}{"output": "classifier improved|generator improved"}},
	})
	adapter := gepa.newEvaluationAdapter(
		newCompatMergeFixtureProgram("classifier base", "generator base"),
		dataset,
		exactOutputMetric,
	)
	gepa.setLatestEvaluationAdapter(adapter)

	proposed := gepa.proposeNextGenerationCandidate(ctx, current, source, 2)
	if proposed == nil {
		t.Fatalf("expected merge proposal")
	}
	if generationLLM.generateCalls != 0 {
		t.Fatalf("expected merge path to avoid generation LLM, got %d calls", generationLLM.generateCalls)
	}

	parentLabels := make([]string, 0, len(proposed.ParentIDs))
	parentIndex := map[string]*GEPACandidate{
		"source":  source,
		"partner": partner,
	}
	originalInstructions := compatCandidateInstructions(root)
	for _, parentID := range proposed.ParentIDs {
		parent := parentIndex[parentID]
		if parent == nil {
			continue
		}
		parentLabels = append(parentLabels, compatCandidateLabel(compatCandidateInstructions(parent), originalInstructions))
	}
	sort.Strings(parentLabels)

	return compatMergeResult{
		MergedCandidatePresent:           len(proposed.ParentIDs) == 2,
		MergedCandidateUpdatedComponents: compatUpdatedComponents(originalInstructions, compatCandidateInstructions(proposed)),
		MergedCandidateParentLabels:      parentLabels,
		MergedCandidateParentCount:       len(proposed.ParentIDs),
	}
}

func runCompatFeedbackScenario(t *testing.T, ctx context.Context) compatFeedbackResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatFeedbackFixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := DefaultGEPAConfig()
	config.PopulationSize = 1
	config.MaxGenerations = 2
	config.ReflectionFreq = 1
	config.SelectionStrategy = "tournament"
	config.TournamentSize = 1
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 1
	config.RandomSeed = 7
	config.FeedbackEvaluator = GEPAFeedbackEvaluatorFunc(func(_ context.Context, _, _ map[string]interface{}, info *GEPAFeedbackContext) *GEPAFeedback {
		return &GEPAFeedback{
			Feedback:        "Use classifier terminology exactly",
			TargetComponent: info.Candidate.ModuleName,
		}
	})

	gepa, err := NewGEPA(config)
	if err != nil {
		t.Fatalf("new GEPA: %v", err)
	}

	originalInstruction := "classifier base"
	program := newCompatFeedbackFixtureProgram(originalInstruction)
	dataset := datasets.NewSimpleDataset([]core.Example{
		{
			Inputs:  map[string]interface{}{"input": "feedback fixture input"},
			Outputs: map[string]interface{}{"output": "feedback tuned classifier instruction"},
		},
	})

	optimizedProgram, err := gepa.Compile(ctx, program, dataset, exactOutputMetric)
	if err != nil {
		t.Fatalf("compile feedback fixture: %v", err)
	}

	population := gepa.CurrentPopulation()
	if population == nil || len(population.Candidates) == 0 {
		t.Fatalf("missing GEPA population after feedback fixture compile")
	}

	return compatFeedbackResult{
		OriginalInstruction:     originalInstruction,
		CandidateInstruction:    candidateInstructionForModule(population.Candidates[0], "classifier"),
		FinalProgramInstruction: optimizedProgram.Modules["classifier"].GetSignature().Instruction,
		CandidateCount:          len(population.Candidates),
	}
}

func runCompatFormatFailureScenario(t *testing.T, ctx context.Context) compatFeedbackResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatFormatFixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := DefaultGEPAConfig()
	config.PopulationSize = 1
	config.MaxGenerations = 2
	config.ReflectionFreq = 1
	config.SelectionStrategy = "tournament"
	config.TournamentSize = 1
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 1
	config.RandomSeed = 7
	config.AddFormatFailureAsFeedback = true

	gepa, err := NewGEPA(config)
	if err != nil {
		t.Fatalf("new GEPA: %v", err)
	}

	originalInstruction := "classifier base"
	program := newCompatFormatFixtureProgram(originalInstruction)
	dataset := datasets.NewSimpleDataset([]core.Example{
		{
			Inputs:  map[string]interface{}{"input": "format fixture input"},
			Outputs: map[string]interface{}{"output": "format tuned classifier instruction"},
		},
	})

	optimizedProgram, err := gepa.Compile(ctx, program, dataset, exactOutputMetric)
	if err != nil {
		t.Fatalf("compile format fixture: %v", err)
	}

	population := gepa.CurrentPopulation()
	if population == nil || len(population.Candidates) == 0 {
		t.Fatalf("missing GEPA population after format fixture compile")
	}

	return compatFeedbackResult{
		OriginalInstruction:     originalInstruction,
		CandidateInstruction:    candidateInstructionForModule(population.Candidates[0], "classifier"),
		FinalProgramInstruction: optimizedProgram.Modules["classifier"].GetSignature().Instruction,
		CandidateCount:          len(population.Candidates),
	}
}

func runCompatMinibatchScenario(t *testing.T, ctx context.Context, proposal, datasetOutput, modelID string) compatMinibatchScenarioResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatSingleInstructionLLM{proposal: proposal, modelID: modelID}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := DefaultGEPAConfig()
	config.PopulationSize = 1
	config.MaxGenerations = 2
	config.MutationRate = 1.0
	config.ReflectionFreq = 0
	config.SelectionStrategy = "tournament"
	config.TournamentSize = 1
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 2
	config.RandomSeed = 7

	gepa, err := NewGEPA(config)
	if err != nil {
		t.Fatalf("new GEPA minibatch fixture: %v", err)
	}

	dataset := datasets.NewSimpleDataset([]core.Example{
		{Inputs: map[string]interface{}{"input": "mini-1"}, Outputs: map[string]interface{}{"output": datasetOutput}},
		{Inputs: map[string]interface{}{"input": "mini-2"}, Outputs: map[string]interface{}{"output": datasetOutput}},
		{Inputs: map[string]interface{}{"input": "mini-3"}, Outputs: map[string]interface{}{"output": datasetOutput}},
	})

	optimizedProgram, err := gepa.Compile(ctx, newCompatFeedbackFixtureProgram("alpha base"), dataset, exactOutputMetric)
	if err != nil {
		t.Fatalf("compile minibatch fixture: %v", err)
	}

	candidateCount := compatPopulationCandidateCount(gepa.state)
	winningInstruction := ""
	if gepa.state != nil && gepa.state.BestCandidate != nil {
		winningInstruction = candidateInstructionForModule(gepa.state.BestCandidate, "classifier")
	}

	return compatMinibatchScenarioResult{
		CandidateCount:              candidateCount,
		CandidateAdded:              candidateCount > 1,
		FinalProgramInstruction:     optimizedProgram.Modules["classifier"].GetSignature().Instruction,
		WinningCandidateInstruction: winningInstruction,
	}
}

func runCompatMinibatchAcceptanceScenario(t *testing.T, ctx context.Context) compatMinibatchAcceptanceResult {
	t.Helper()

	return compatMinibatchAcceptanceResult{
		AcceptedCase: runCompatMinibatchScenario(t, ctx, "alpha tuned", "alpha tuned", "fixture-gepa-minibatch-accept"),
		RejectedCase: runCompatMinibatchScenario(t, ctx, "alpha worse", "alpha base", "fixture-gepa-minibatch-reject"),
	}
}

func runCompatResumeScenario(t *testing.T, ctx context.Context) compatResumeResult {
	t.Helper()

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &compatResumeFixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	buildConfig := func(runDir string) *GEPAConfig {
		config := DefaultGEPAConfig()
		config.RunDir = runDir
		config.PopulationSize = 1
		config.MaxGenerations = 3
		config.MutationRate = 1.0
		config.ReflectionFreq = 0
		config.SelectionStrategy = "tournament"
		config.TournamentSize = 1
		config.ConcurrencyLevel = 1
		config.EvaluationBatchSize = 1
		config.RandomSeed = 7
		return config
	}

	stopRunDir := t.TempDir()
	stoppedConfig := buildConfig(stopRunDir)
	stoppedConfig.StopCallbacks = []GEPAStopper{
		func(_ context.Context, gepa *GEPA) *GEPAStopDecision {
			if gepa == nil || gepa.state == nil {
				return nil
			}
			if gepa.state.BestFitness < 0.5 || gepa.state.BestFitness >= 1.0 {
				return nil
			}
			return &GEPAStopDecision{Reason: "compat_resume_stop"}
		},
	}

	dataset := datasets.NewSimpleDataset([]core.Example{
		{
			Inputs:  map[string]interface{}{"input": "resume fixture input"},
			Outputs: map[string]interface{}{"output": "classifier best"},
		},
	})

	firstRun, err := NewGEPA(stoppedConfig)
	if err != nil {
		t.Fatalf("new stopped GEPA: %v", err)
	}
	_, err = firstRun.Compile(ctx, newCompatFeedbackFixtureProgram("classifier base"), dataset, compatResumeProgressMetric)
	if err != nil {
		t.Fatalf("compile stopped resume fixture: %v", err)
	}

	resumeConfig := buildConfig(stopRunDir)
	resumedRun, err := NewGEPA(resumeConfig)
	if err != nil {
		t.Fatalf("new resumed GEPA: %v", err)
	}
	resumedProgram, err := resumedRun.Compile(ctx, newCompatFeedbackFixtureProgram("classifier base"), dataset, compatResumeProgressMetric)
	if err != nil {
		t.Fatalf("compile resumed resume fixture: %v", err)
	}

	freshConfig := buildConfig(t.TempDir())
	freshRun, err := NewGEPA(freshConfig)
	if err != nil {
		t.Fatalf("new fresh GEPA: %v", err)
	}
	freshProgram, err := freshRun.Compile(ctx, newCompatFeedbackFixtureProgram("classifier base"), dataset, compatResumeProgressMetric)
	if err != nil {
		t.Fatalf("compile fresh resume fixture: %v", err)
	}

	return compatResumeResult{
		CheckpointWritten:              compatFileExists(filepath.Join(stopRunDir, "gepa_state.json")),
		StoppedMetricCalls:             firstRun.state.MetricCallCount(),
		ResumedMetricCalls:             resumedRun.state.MetricCallCount(),
		FreshMetricCalls:               freshRun.state.MetricCallCount(),
		ResumedCandidateCount:          compatPopulationCandidateCount(resumedRun.state),
		FreshCandidateCount:            compatPopulationCandidateCount(freshRun.state),
		ResumedFinalProgramInstruction: resumedProgram.Modules["classifier"].GetSignature().Instruction,
		FreshFinalProgramInstruction:   freshProgram.Modules["classifier"].GetSignature().Instruction,
	}
}

func compatPopulationCandidateCount(state *GEPAState) int {
	if state == nil {
		return 0
	}
	seen := make(map[string]struct{})
	for _, population := range state.PopulationHistory {
		if population == nil {
			continue
		}
		for _, candidate := range population.Candidates {
			if candidate == nil || strings.TrimSpace(candidate.ID) == "" {
				continue
			}
			seen[candidate.ID] = struct{}{}
		}
	}
	return len(seen)
}

func compatFileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func buildCompatFixtureReport(t *testing.T) compatFixtureReport {
	t.Helper()

	ctx := context.Background()
	report := compatFixtureReport{
		Runner: "go_dspy_go_test",
		Fixtures: compatFixtureCollection{
			ComponentSelection: compatComponentSelectionReport{
				Scenarios: make(map[string]compatFixtureScenarioResult),
			},
		},
	}

	for _, selector := range []string{"round_robin", "all"} {
		report.Fixtures.ComponentSelection.Scenarios[selector] = runCompatSelectorScenario(t, ctx, selector)
	}
	report.Fixtures.ValidationFrontier = runCompatValidationFrontierScenario(t, ctx)
	report.Fixtures.AncestorMerge = runCompatMergeScenario(t, ctx)
	report.Fixtures.FeedbackGuided = runCompatFeedbackScenario(t, ctx)
	report.Fixtures.FormatFailure = runCompatFormatFailureScenario(t, ctx)
	report.Fixtures.MinibatchAcceptance = runCompatMinibatchAcceptanceScenario(t, ctx)
	report.Fixtures.ResumeParity = runCompatResumeScenario(t, ctx)
	return report
}

func TestGEPAFixtureReport_WritesWhenEnvSet(t *testing.T) {
	outputPath := strings.TrimSpace(os.Getenv(gepaFixtureOutputEnv))
	if outputPath == "" {
		t.Skipf("%s is not set", gepaFixtureOutputEnv)
	}

	rendered, err := json.MarshalIndent(buildCompatFixtureReport(t), "", "  ")
	if err != nil {
		t.Fatalf("marshal fixture report: %v", err)
	}
	if err := os.WriteFile(outputPath, append(rendered, '\n'), 0o644); err != nil {
		t.Fatalf("write fixture report: %v", err)
	}
	fmt.Println(string(rendered))
}
