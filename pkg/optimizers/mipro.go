package optimizers

import (
	"context"
	"fmt"
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

func NewMIPROv2(metric func(example, prediction map[string]interface{}, trace *core.Trace) float64, numCandidates, maxBootstrappedDemos, maxLabeledDemos, numTrials int, promptModel, taskModel core.LLM, miniBatchSize, fullEvalSteps int, verbose bool) *MIPRO {
	return &MIPRO{
		Metric:               metric,
		NumCandidates:        numCandidates,
		MaxBootstrappedDemos: maxBootstrappedDemos,
		MaxLabeledDemos:      maxLabeledDemos,
		NumTrials:            numTrials,
		PromptModel:          promptModel,
		TaskModel:            taskModel,
		MiniBatchSize:        miniBatchSize,
		FullEvalSteps:        fullEvalSteps,
		Verbose:              verbose,
	}
}

type Trial struct {
	Params map[string]int
	Score  float64
}

func (m *MIPRO) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	instructionCandidates, err := m.generateInstructionCandidates(ctx, program, dataset)
	if err != nil {
		return program, err
	}

	demoCandidates, err := m.generateDemoCandidates(ctx, program, dataset)
	if err != nil {
		return program, err
	}

	var bestTrial Trial
	for i := 0; i < m.NumTrials; i++ {
		trial := m.generateTrial(program.GetModules(), len(instructionCandidates), len(demoCandidates))
		candidateProgram := m.constructProgram(program, trial, instructionCandidates, demoCandidates)

		score, err := m.evaluateProgram(ctx, candidateProgram, dataset, metric)
		if err != nil {
			return program, err
		}

		trial.Score = score
		if score > bestTrial.Score || i == 0 {
			bestTrial = trial
		}

		if m.Verbose {
			m.logTrialResult(i, trial)
		}
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
	for i, module := range program.GetModules() {
		if predictor, ok := module.(*modules.Predict); ok {
			instructionIdx := trial.Params[fmt.Sprintf("instruction_%d", i)]
			demoIdx := trial.Params[fmt.Sprintf("demo_%d", i)]

			predictor.SetSignature(predictor.GetSignature().WithInstruction(instructionCandidates[i][instructionIdx]))
			predictor.SetDemos(demoCandidates[i][demoIdx])
		}
	}
	return program
}

func (m *MIPRO) evaluateProgram(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (float64, error) {
	totalScore := 0.0
	count := 0

	for i := 0; i < m.MiniBatchSize; i++ {
		example, ok := dataset.Next()
		if !ok {
			break
		}

		prediction, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			return 0, err
		}

		trace := core.GetTraceManager(ctx).CurrentTrace
		score := m.Metric(example.Inputs, prediction, trace)
		totalScore += score
		count++
	}

	if count == 0 {
		return 0, fmt.Errorf("no examples evaluated")
	}

	return totalScore / float64(count), nil
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
				candidates[i][j] = instruction
			}
		}
	}
	return candidates, nil
}

func (m *MIPRO) generateDemoCandidates(ctx context.Context, program core.Program, dataset core.Dataset) ([][][]core.Example, error) {
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
				// For simplicity, we're not implementing labeled demos here
				candidates[i][j] = demos
			}
		}
	}
	return candidates, nil
}
