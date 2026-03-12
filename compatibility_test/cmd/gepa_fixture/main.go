package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

type fixtureModule struct {
	signature core.Signature
}

func newFixtureModule(instruction string) *fixtureModule {
	return &fixtureModule{
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

func (m *fixtureModule) Process(context.Context, map[string]any, ...core.Option) (map[string]any, error) {
	return map[string]any{"output": m.signature.Instruction}, nil
}

func (m *fixtureModule) GetSignature() core.Signature { return m.signature }

func (m *fixtureModule) SetSignature(signature core.Signature) { m.signature = signature }

func (m *fixtureModule) SetLLM(core.LLM) {}

func (m *fixtureModule) Clone() core.Module { return &fixtureModule{signature: m.signature} }

func (m *fixtureModule) GetDisplayName() string { return "FixtureModule" }

func (m *fixtureModule) GetModuleType() string { return "fixture" }

type fixtureLLM struct{}

func (f *fixtureLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	switch {
	case strings.Contains(prompt, `Original: "classifier base"`):
		return &core.LLMResponse{Content: "classifier improved"}, nil
	case strings.Contains(prompt, `Original: "generator base"`):
		return &core.LLMResponse{Content: "generator improved"}, nil
	default:
		return &core.LLMResponse{Content: "fixture fallback response"}, nil
	}
}

func (f *fixtureLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *fixtureLLM) GenerateWithFunctions(context.Context, string, []map[string]interface{}, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (f *fixtureLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (f *fixtureLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (f *fixtureLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *fixtureLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}

func (f *fixtureLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (f *fixtureLLM) ProviderName() string { return "fixture" }

func (f *fixtureLLM) ModelID() string { return "fixture-gepa" }

func (f *fixtureLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type fixtureScenarioResult struct {
	Selector                        string            `json:"selector"`
	FirstCandidateUpdatedComponents []string          `json:"first_candidate_updated_components"`
	FirstCandidateInstructions      map[string]string `json:"first_candidate_instructions"`
	FinalProgramUpdatedComponents   []string          `json:"final_program_updated_components"`
	FinalProgramInstructions        map[string]string `json:"final_program_instructions"`
	CandidateCount                  int               `json:"candidate_count"`
}

type fixtureReport struct {
	Runner    string                           `json:"runner"`
	Fixture   string                           `json:"fixture"`
	Scenarios map[string]fixtureScenarioResult `json:"scenarios"`
}

func newFixtureProgram(classifierInstruction, generatorInstruction string) core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"classifier": newFixtureModule(classifierInstruction),
		"generator":  newFixtureModule(generatorInstruction),
	}, func(modules map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
		return func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{
				"output": modules["classifier"].GetSignature().Instruction + "|" + modules["generator"].GetSignature().Instruction,
			}, nil
		}
	})
}

func instructionProgressMetric(_ map[string]interface{}, actual map[string]interface{}) float64 {
	output, _ := actual["output"].(string)
	score := 0.0
	for _, part := range strings.Split(output, "|") {
		if strings.Contains(part, "improved") {
			score += 0.5
		}
	}
	return score
}

func programInstructions(program core.Program) map[string]string {
	instructions := make(map[string]string, len(program.Modules))
	for moduleName, module := range program.Modules {
		instructions[moduleName] = module.GetSignature().Instruction
	}
	return instructions
}

func updatedComponents(original, current map[string]string) []string {
	updated := make([]string, 0)
	for moduleName, instruction := range current {
		if original[moduleName] != instruction {
			updated = append(updated, moduleName)
		}
	}
	sort.Strings(updated)
	return updated
}

func runSelectorScenario(ctx context.Context, selector string) (fixtureScenarioResult, error) {
	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	llm := &fixtureLLM{}
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	config := optimizers.DefaultGEPAConfig()
	config.PopulationSize = 1
	config.MaxGenerations = 2
	config.MutationRate = 1.0
	config.ReflectionFreq = 0
	config.ComponentSelection = selector
	config.SelectionStrategy = "tournament"
	config.TournamentSize = 1
	config.ConcurrencyLevel = 1
	config.EvaluationBatchSize = 1

	gepa, err := optimizers.NewGEPA(config)
	if err != nil {
		return fixtureScenarioResult{}, err
	}

	originalInstructions := map[string]string{
		"classifier": "classifier base",
		"generator":  "generator base",
	}
	program := newFixtureProgram(originalInstructions["classifier"], originalInstructions["generator"])
	dataset := datasets.NewSimpleDataset([]core.Example{{Outputs: map[string]interface{}{"output": "fixture"}}})

	optimizedProgram, err := gepa.Compile(ctx, program, dataset, instructionProgressMetric)
	if err != nil {
		return fixtureScenarioResult{}, err
	}

	population := gepa.CurrentPopulation()
	if population == nil || len(population.Candidates) == 0 {
		return fixtureScenarioResult{}, fmt.Errorf("missing GEPA population after compile")
	}

	// The Go side inspects the first candidate from the final population.
	// The Python fixture looks at candidates[1] because DSPy detailed_results
	// includes the seed candidate at index 0, while dspy-go exposes only the
	// current population after the first update step.
	firstCandidate := population.Candidates[0]
	firstCandidateInstructions := make(map[string]string)
	for moduleName, instruction := range firstCandidate.ComponentTexts {
		firstCandidateInstructions[moduleName] = instruction
	}

	finalProgramInstructions := programInstructions(optimizedProgram)
	return fixtureScenarioResult{
		Selector:                        selector,
		FirstCandidateUpdatedComponents: updatedComponents(originalInstructions, firstCandidateInstructions),
		FirstCandidateInstructions:      firstCandidateInstructions,
		FinalProgramUpdatedComponents:   updatedComponents(originalInstructions, finalProgramInstructions),
		FinalProgramInstructions:        finalProgramInstructions,
		CandidateCount:                  len(population.Candidates),
	}, nil
}

func main() {
	outputPath := flag.String("output", "", "Optional path to write JSON results.")
	flag.Parse()

	ctx := context.Background()
	report := fixtureReport{
		Runner:    "go_dspy_go",
		Fixture:   "gepa_component_selection",
		Scenarios: make(map[string]fixtureScenarioResult),
	}

	for _, selector := range []string{"round_robin", "all"} {
		result, err := runSelectorScenario(ctx, selector)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fixture %s failed: %v\n", selector, err)
			os.Exit(1)
		}
		report.Scenarios[selector] = result
	}

	rendered, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal fixture report: %v\n", err)
		os.Exit(1)
	}

	if *outputPath != "" {
		if err := os.WriteFile(*outputPath, append(rendered, '\n'), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write fixture report: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Println(string(rendered))
}
