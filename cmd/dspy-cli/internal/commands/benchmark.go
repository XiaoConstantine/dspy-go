package commands

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/benchmarks/tblite"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/spf13/cobra"
)

// NewBenchmarkCommand exposes benchmark runners.
func NewBenchmarkCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "benchmark",
		Short: "Run benchmark suites against dspy-go agents",
	}
	cmd.AddCommand(newTBLiteBenchmarkCommand(defaultTerminalTaskAgentFactory))
	return cmd
}

func newTBLiteBenchmarkCommand(factory func(*terminalTaskCommandConfig) (tblite.Agent, error)) *cobra.Command {
	agentCfg := &terminalTaskCommandConfig{}

	var split string
	var offset int
	var limit int
	var rootDir string
	var outputPath string
	var keepArtifacts bool
	var label string
	var shuffleSeed int64
	var useGEPA bool
	var populationSize int
	var generations int
	var reflectionFreq int
	var validationSplit float64
	var testSplit float64

	cmd := &cobra.Command{
		Use:   "tblite",
		Short: "Run a fixed-slice TBLite evaluation with the native tool-calling benchmark agent",
		RunE: func(cmd *cobra.Command, args []string) error {
			tasks, err := datasets.FetchTBLiteTasksFromHuggingFaceRange(split, offset, limit)
			if err != nil {
				return err
			}
			if len(tasks) == 0 {
				return fmt.Errorf("no tblite tasks returned for split=%s offset=%d limit=%d", split, offset, limit)
			}
			tasks = shuffledTBLiteTasks(tasks, shuffleSeed)

			if !useGEPA {
				agent, err := factory(agentCfg)
				if err != nil {
					return err
				}

				runner := tblite.NewRunner(agent, tblite.RunnerConfig{
					MaxTurns:          agentCfg.MaxTurns,
					UseTaskContainers: true,
				})

				report, err := tblite.EvaluateTasks(cmd.Context(), runner, tasks, tblite.EvalConfig{
					Label:         label,
					DatasetName:   "NousResearch/openthoughts-tblite",
					Split:         split,
					Offset:        offset,
					Limit:         limit,
					RootDir:       rootDir,
					KeepArtifacts: keepArtifacts,
				})
				if err != nil {
					return err
				}

				if outputPath != "" {
					if err := tblite.WriteReport(outputPath, report); err != nil {
						return err
					}
				}

				cmd.Printf("TBLite %s on %s/%s\n", report.LabelOrDefault(), agentCfg.ProviderOrDefault(), agentCfg.ModelOrDefault())
				cmd.Printf("Tasks: %d  Passed: %d  Pass rate: %.2f\n", report.Summary.TotalTasks, report.Summary.PassedTasks, report.Summary.PassRate)
				cmd.Printf("Avg tool calls: %.2f  Avg duration: %s\n", report.Summary.AverageToolCalls, report.Summary.AverageDuration.Round(100*time.Millisecond))
				cmd.Printf("Tokens: prompt=%d completion=%d total=%d\n", report.Summary.TotalPromptTokens, report.Summary.TotalCompletionTokens, report.Summary.TotalTokens)
				if outputPath != "" {
					cmd.Printf("Report: %s\n", outputPath)
				}
				return nil
			}

			return runTBLiteGEPABenchmark(cmd, agentCfg, tasks, split, offset, limit, rootDir, outputPath, keepArtifacts, label, populationSize, generations, reflectionFreq, validationSplit, testSplit)
		},
	}

	cmd.Flags().StringVar(&agentCfg.Provider, "provider", "google", "LLM provider used for the benchmark run")
	cmd.Flags().StringVar(&agentCfg.Model, "model", "", "Model ID to use (defaults by provider)")
	cmd.Flags().StringVar(&agentCfg.APIKey, "api-key", "", "Explicit API key (otherwise provider-specific environment variables are used)")
	cmd.Flags().IntVar(&agentCfg.MaxTurns, "max-turns", 20, "Maximum turns per task")
	cmd.Flags().IntVar(&agentCfg.MaxTokens, "max-tokens", 2048, "Maximum tokens per model response")
	cmd.Flags().Float64Var(&agentCfg.Temperature, "temperature", 0, "Sampling temperature for task execution")
	cmd.Flags().IntVar(&agentCfg.ToolOutputLimit, "tool-output-limit", 16384, "Maximum characters returned per tool observation")
	cmd.Flags().StringVar(&agentCfg.SystemPrompt, "system-prompt", "", "Optional benchmark agent system prompt override")

	cmd.Flags().StringVar(&split, "split", "train", "Dataset split")
	cmd.Flags().IntVar(&offset, "offset", 0, "Task offset in the dataset")
	cmd.Flags().IntVar(&limit, "limit", 5, "Number of tasks to evaluate")
	cmd.Flags().StringVar(&rootDir, "root-dir", "", "Directory for materialized task workspaces")
	cmd.Flags().StringVar(&outputPath, "output", "", "Optional path to write the JSON report")
	cmd.Flags().BoolVar(&keepArtifacts, "keep-artifacts", false, "Keep materialized task directories after the run")
	cmd.Flags().StringVar(&label, "label", "", "Optional report label (e.g. baseline or tuned)")
	cmd.Flags().Int64Var(&shuffleSeed, "shuffle-seed", 1, "Deterministic shuffle seed applied before splitting/evaluation")
	cmd.Flags().BoolVar(&useGEPA, "gepa", false, "Optimize the tool-calling system prompt with GEPA before evaluating on a held-out test split")
	cmd.Flags().IntVar(&populationSize, "population", 4, "GEPA population size for --gepa runs")
	cmd.Flags().IntVar(&generations, "generations", 2, "GEPA generations for --gepa runs")
	cmd.Flags().IntVar(&reflectionFreq, "reflection-freq", 1, "GEPA reflection frequency for --gepa runs")
	cmd.Flags().Float64Var(&validationSplit, "validation-split", 0.2, "Validation split used inside the optimization set for --gepa runs")
	cmd.Flags().Float64Var(&testSplit, "test-split", 0.2, "Held-out test split used for baseline vs tuned comparison")

	_ = cmd.MarkFlagRequired("root-dir")
	return cmd
}

func runTBLiteGEPABenchmark(
	cmd *cobra.Command,
	agentCfg *terminalTaskCommandConfig,
	tasks []datasets.TBLiteTask,
	split string,
	offset int,
	limit int,
	rootDir string,
	outputPath string,
	keepArtifacts bool,
	label string,
	populationSize int,
	generations int,
	reflectionFreq int,
	validationSplit float64,
	testSplit float64,
) error {
	trainTasks, validationTasks, testTasks, err := partitionTBLiteTasks(tasks, validationSplit, testSplit)
	if err != nil {
		return err
	}

	llm, err := newTerminalTaskLLM(agentCfg)
	if err != nil {
		return err
	}

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	baselineAgent, err := newToolCallingBenchmarkAgentWithLLM(llm, agentCfg)
	if err != nil {
		return err
	}
	baselineReport, err := tblite.EvaluateTasks(cmd.Context(), tblite.NewRunner(baselineAgent, tblite.RunnerConfig{
		MaxTurns:             agentCfg.MaxTurns,
		UseTaskContainers:    true,
		RespectAgentMaxTurns: true,
	}), testTasks, tblite.EvalConfig{
		Label:         "baseline",
		DatasetName:   "NousResearch/openthoughts-tblite",
		Split:         split,
		Offset:        offset,
		Limit:         len(testTasks),
		RootDir:       filepath.Join(rootDir, "baseline"),
		KeepArtifacts: keepArtifacts,
	})
	if err != nil {
		return err
	}

	optimizer := optimize.NewGEPAAgentOptimizer(
		baselineAgent,
		tblite.NewGEPAEvaluator(tblite.GEPAEvaluatorConfig{
			RootDir:           filepath.Join(rootDir, "gepa"),
			KeepArtifacts:     keepArtifacts,
			MaxTurns:          agentCfg.MaxTurns,
			UseTaskContainers: true,
		}),
		optimize.GEPAAdapterConfig{
			PopulationSize:  populationSize,
			MaxGenerations:  generations,
			ReflectionFreq:  reflectionFreq,
			ValidationSplit: validationSplit,
			EvalConcurrency: 1,
			PassThreshold:   1.0,
			ArtifactKeys:    []optimize.ArtifactKey{optimize.ArtifactSkillPack, optimize.ArtifactToolPolicy},
			PrimaryArtifact: optimize.ArtifactToolPolicy,
			IntMutationPlans: map[string]optimize.IntMutationConfig{
				"max_turns": tbliteMaxTurnMutationConfig(agentCfg.MaxTurns),
			},
		},
	)

	optimizeResult, err := optimizer.Optimize(cmd.Context(), optimize.GEPAOptimizeRequest{
		SeedArtifacts:      baselineAgent.GetArtifacts(),
		TrainingExamples:   tblite.ExamplesFromTasks(trainTasks),
		ValidationExamples: tblite.ExamplesFromTasks(validationTasks),
	})
	if err != nil {
		return err
	}

	bestArtifacts := optimizeResult.BestArtifacts.Clone()
	bestValidation := optimizeResult.BestValidationEvaluation
	if len(validationTasks) > 0 && optimizeResult.OptimizationState != nil {
		if selectedEvaluation, err := selectBestTBLiteValidationCandidate(
			cmd.Context(),
			optimizer,
			optimizeResult.OptimizationState,
			tblite.ExamplesFromTasks(validationTasks),
		); err != nil {
			return err
		} else if selectedEvaluation != nil {
			bestArtifacts = selectedEvaluation.Artifacts.Clone()
			bestValidation = selectedEvaluation
		}
	}

	tunedClone, err := baselineAgent.Clone()
	if err != nil {
		return err
	}
	tunedAgent, ok := tunedClone.(*tblite.ToolCallingAgent)
	if !ok {
		return fmt.Errorf("unexpected tuned agent type %T", tunedClone)
	}
	if err := tunedAgent.SetArtifacts(bestArtifacts); err != nil {
		return err
	}

	tunedReport, err := tblite.EvaluateTasks(cmd.Context(), tblite.NewRunner(tunedAgent, tblite.RunnerConfig{
		MaxTurns:             agentCfg.MaxTurns,
		UseTaskContainers:    true,
		RespectAgentMaxTurns: true,
	}), testTasks, tblite.EvalConfig{
		Label:         "tuned",
		DatasetName:   "NousResearch/openthoughts-tblite",
		Split:         split,
		Offset:        offset,
		Limit:         len(testTasks),
		RootDir:       filepath.Join(rootDir, "tuned"),
		KeepArtifacts: keepArtifacts,
	})
	if err != nil {
		return err
	}

	report := &tblite.ComparisonReport{
		Label:               label,
		DatasetName:         "NousResearch/openthoughts-tblite",
		Split:               split,
		Offset:              offset,
		Limit:               limit,
		StartedAt:           time.Now(),
		TrainingTaskCount:   len(trainTasks),
		ValidationTaskCount: len(validationTasks),
		TestTaskCount:       len(testTasks),
		BestArtifacts:       bestArtifacts.Clone(),
		Baseline:            baselineReport,
		Tuned:               tunedReport,
	}

	if outputPath != "" {
		if err := tblite.WriteComparisonReport(outputPath, report); err != nil {
			return err
		}
	}

	cmd.Printf("TBLite GEPA comparison on %s/%s\n", agentCfg.ProviderOrDefault(), agentCfg.ModelOrDefault())
	cmd.Printf("Train/val/test: %d/%d/%d\n", len(trainTasks), len(validationTasks), len(testTasks))
	cmd.Printf("Baseline pass rate: %.2f (%d/%d)\n", baselineReport.Summary.PassRate, baselineReport.Summary.PassedTasks, baselineReport.Summary.TotalTasks)
	cmd.Printf("Tuned pass rate: %.2f (%d/%d)\n", tunedReport.Summary.PassRate, tunedReport.Summary.PassedTasks, tunedReport.Summary.TotalTasks)
	cmd.Printf("Best artifact (%s): %q\n", optimize.ArtifactToolPolicy, bestArtifacts.Text[optimize.ArtifactToolPolicy])
	if bestValidation != nil {
		cmd.Printf("Best validation tblite_pass: %.2f\n", bestValidation.AverageScore)
	}
	if bestMaxTurns, ok := bestArtifacts.Int["max_turns"]; ok && bestMaxTurns > 0 {
		cmd.Printf("Best max_turns: %d\n", bestMaxTurns)
	}
	if outputPath != "" {
		cmd.Printf("Comparison report: %s\n", outputPath)
	}

	return nil
}

func tbliteMaxTurnMutationConfig(seed int) optimize.IntMutationConfig {
	if seed <= 0 {
		seed = 20
	}
	min := seed - 8
	if min < 8 {
		min = 8
	}
	max := seed + 16
	if max < min {
		max = min
	}
	return optimize.IntMutationConfig{
		Min:  min,
		Max:  max,
		Step: 4,
	}
}

func shuffledTBLiteTasks(tasks []datasets.TBLiteTask, seed int64) []datasets.TBLiteTask {
	shuffled := append([]datasets.TBLiteTask(nil), tasks...)
	if len(shuffled) <= 1 || seed == 0 {
		return shuffled
	}

	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled
}

func selectBestTBLiteValidationCandidate(
	ctx context.Context,
	optimizer *optimize.GEPAAgentOptimizer,
	state *optimizers.GEPAState,
	validationExamples []optimize.AgentExample,
) (*optimize.GEPACandidateEvaluation, error) {
	if optimizer == nil || state == nil || len(validationExamples) == 0 {
		return nil, nil
	}

	candidates := state.ParetoArchive
	if len(candidates) == 0 && state.BestCandidate != nil {
		candidates = []*optimizers.GEPACandidate{state.BestCandidate}
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no GEPA candidates available for TBLite validation selection")
	}

	var best *optimize.GEPACandidateEvaluation
	for _, candidate := range candidates {
		evaluation, err := optimizer.EvaluateCandidate(ctx, optimizers.CloneCandidate(candidate), validationExamples)
		if err != nil {
			return nil, err
		}
		if evaluation == nil || evaluation.Fitness == nil {
			continue
		}
		if best == nil || evaluation.AverageScore > best.AverageScore || (evaluation.AverageScore == best.AverageScore && evaluation.Fitness.WeightedScore > best.Fitness.WeightedScore) {
			best = evaluation
		}
	}

	if best == nil {
		return nil, fmt.Errorf("no successful GEPA validation evaluation found")
	}
	return best, nil
}

func partitionTBLiteTasks(tasks []datasets.TBLiteTask, validationSplit, testSplit float64) ([]datasets.TBLiteTask, []datasets.TBLiteTask, []datasets.TBLiteTask, error) {
	if len(tasks) < 3 {
		return nil, nil, nil, fmt.Errorf("gepa benchmark requires at least 3 tasks, got %d", len(tasks))
	}
	if validationSplit <= 0 {
		validationSplit = 0.2
	}
	if testSplit <= 0 {
		testSplit = 0.2
	}

	testCount := max(1, int(math.Floor(float64(len(tasks))*testSplit)))
	if testCount >= len(tasks) {
		testCount = 1
	}
	remaining := len(tasks) - testCount
	validationCount := max(1, int(math.Floor(float64(remaining)*validationSplit)))
	if validationCount >= remaining {
		validationCount = 1
	}
	trainCount := len(tasks) - validationCount - testCount
	if trainCount <= 0 {
		return nil, nil, nil, fmt.Errorf("not enough tasks after splitting into train/validation/test")
	}

	train := append([]datasets.TBLiteTask(nil), tasks[:trainCount]...)
	validation := append([]datasets.TBLiteTask(nil), tasks[trainCount:trainCount+validationCount]...)
	test := append([]datasets.TBLiteTask(nil), tasks[trainCount+validationCount:]...)
	return train, validation, test, nil
}
