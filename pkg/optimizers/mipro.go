package optimizers

import (
	"context"
	"fmt"
	"log"
	"math/rand"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type MIPRO struct {
	Metric               func(example, prediction map[string]interface{}, trace *core.Trace) float64
	NumCandidates        int
	MaxBootstrappedDemos int
	MaxLabeledDemos      int
	NumTrials            int
	PromptModel          core.LLM
	TaskModel            core.LLM
	MiniBatchSize        int
	FullEvalSteps        int
	Verbose              bool
}

func NewMIPRO(metric func(example, prediction map[string]interface{}, trace *core.Trace) float64, opts ...MIPROOption) *MIPRO {
	m := &MIPRO{
		Metric:               metric,
		NumCandidates:        10,
		MaxBootstrappedDemos: 5,
		MaxLabeledDemos:      5,
		NumTrials:            100,
		MiniBatchSize:        32,
		FullEvalSteps:        10,
		Verbose:              false,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

type MIPROOption func(*MIPRO)

func WithNumCandidates(n int) MIPROOption {
	return func(m *MIPRO) { m.NumCandidates = n }
}

func WithMaxBootstrappedDemos(n int) MIPROOption {
	return func(m *MIPRO) { m.MaxBootstrappedDemos = n }
}

func WithMaxLabeledDemos(n int) MIPROOption {
	return func(m *MIPRO) { m.MaxLabeledDemos = n }
}

func WithNumTrials(n int) MIPROOption {
	return func(m *MIPRO) { m.NumTrials = n }
}

func WithPromptModel(model core.LLM) MIPROOption {
	return func(m *MIPRO) { m.PromptModel = model }
}

func WithTaskModel(model core.LLM) MIPROOption {
	return func(m *MIPRO) { m.TaskModel = model }
}

func WithMiniBatchSize(n int) MIPROOption {
	return func(m *MIPRO) { m.MiniBatchSize = n }
}

func WithFullEvalSteps(n int) MIPROOption {
	return func(m *MIPRO) { m.FullEvalSteps = n }
}

func WithVerbose(v bool) MIPROOption {
	return func(m *MIPRO) { m.Verbose = v }
}

type Trial struct {
	Params map[string]int
	Score  float64
}

func (m *MIPRO) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	instructionCandidates, err := m.generateInstructionCandidates(ctx, program, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to generate instruction candidates: %w", err)
	}

	demoCandidates, err := m.generateDemoCandidates(program, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to generate demo candidates: %w", err)
	}

	var bestTrial Trial
	var compilationError error

	for i := 0; i < m.NumTrials; i++ {
		dataset.Reset()
		trial := m.generateTrial(program.GetModules(), len(instructionCandidates), len(demoCandidates))
		candidateProgram := m.constructProgram(program, trial, instructionCandidates, demoCandidates)

		score, err := m.evaluateProgram(ctx, candidateProgram, dataset, metric)
		if err != nil {
			log.Printf("Error evaluating program in trial %d: %v", i, err)
			compilationError = err
			continue

			// return core.Program{}, fmt.Errorf("failed to evaluate program in trial %d: %w", i, err)
		}

		trial.Score = score
		if score > bestTrial.Score || i == 0 {
			bestTrial = trial
		}

		if m.Verbose {
			m.logTrialResult(i, trial)
		}
	}

	if bestTrial.Params == nil {
		if compilationError != nil {
			return program, fmt.Errorf("compilation failed: %w", compilationError)
		}
		return program, fmt.Errorf("no successful trials")
	}
	return m.constructProgram(program, bestTrial, instructionCandidates, demoCandidates), nil
}

func (m *MIPRO) generateTrial(modules []core.Module, numInstructions, numDemos int) Trial {
	params := make(map[string]int)
	for i := range modules {
		params[fmt.Sprintf("instruction_%d", i)] = rand.Intn(numInstructions)
		params[fmt.Sprintf("demo_%d", i)] = rand.Intn(numDemos)
	}
	return Trial{Params: params}
}

func (m *MIPRO) constructProgram(baseProgram core.Program, trial Trial, instructionCandidates [][]string, demoCandidates [][][]core.Example) core.Program {
	program := baseProgram.Clone()
	modulesList := program.GetModules()

	for i, module := range modulesList {
		if predictor, ok := module.(*modules.Predict); ok {
			instructionIdx := trial.Params[fmt.Sprintf("instruction_%d", i)]
			demoIdx := trial.Params[fmt.Sprintf("demo_%d", i)]

			// Ensure we're using the correct index for instructionCandidates and demoCandidates
			if i < len(instructionCandidates) && instructionIdx < len(instructionCandidates[i]) {
				predictor.SetSignature(predictor.GetSignature().WithInstruction(instructionCandidates[i][instructionIdx]))
			}

			if i < len(demoCandidates) && demoIdx < len(demoCandidates[i]) {
				predictor.SetDemos(demoCandidates[i][demoIdx])
			}
		}
	}
	return program
}

func (m *MIPRO) evaluateProgram(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (float64, error) {
	totalScore := 0.0
	count := 0

	for i := 0; i < m.MiniBatchSize; i++ {
		example, ok := dataset.Next()
		traceManager := core.GetTraceManager(ctx)
		var trace *core.Trace
		if traceManager != nil {
			trace = traceManager.CurrentTrace
		}
		if !ok {
			if i == 0 {
				return 0, fmt.Errorf("no examples available from dataset")
			}
			break
		}

		prediction, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			return 0, fmt.Errorf("failed to evaluate program: error executing program: %w", err)
		}

		score := m.Metric(example.Inputs, prediction, trace)
		totalScore += score
		count++

	}

	if count == 0 {
		return 0, fmt.Errorf("failed to evaluate program: no examples evaluated")
	}
	averageScore := totalScore / float64(count)

	return averageScore, nil
}

func (m *MIPRO) logTrialResult(trialNum int, trial Trial) {
	fmt.Printf("Trial %d: Score %.4f\n", trialNum, trial.Score)
}

func (m *MIPRO) generateInstructionCandidates(ctx context.Context, program core.Program, dataset core.Dataset) ([][]string, error) {
	candidates := make([][]string, len(program.GetModules()))
	for i, module := range program.GetModules() {
		if predictor, ok := module.(*modules.Predict); ok {
			candidates[i] = make([]string, m.NumCandidates)
			for j := 0; j < m.NumCandidates; j++ {
				instruction, err := m.PromptModel.Generate(ctx, fmt.Sprintf("Generate an instruction for the following signature: %s", predictor.GetSignature()))
				if err != nil {
					return nil, err
				}
				candidates[i][j] = instruction.Content
			}
		}
	}
	return candidates, nil
}

func (m *MIPRO) generateDemoCandidates(program core.Program, dataset core.Dataset) ([][][]core.Example, error) {
	candidates := make([][][]core.Example, len(program.GetModules()))
	for i, module := range program.GetModules() {
		if _, ok := module.(*modules.Predict); ok {
			candidates[i] = make([][]core.Example, m.NumCandidates)
			for j := 0; j < m.NumCandidates; j++ {
				demos := make([]core.Example, m.MaxBootstrappedDemos+m.MaxLabeledDemos)
				for k := 0; k < m.MaxBootstrappedDemos; k++ {
					example, ok := dataset.Next()
					if !ok {
						dataset.Reset()
						example, _ = dataset.Next()
					}
					demos[k] = example
				}
				candidates[i][j] = demos
			}
		}
	}
	return candidates, nil
}
