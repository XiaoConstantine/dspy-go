package tblite

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

const exampleMetadataTBLiteTaskKey = "tblite_task"

var verifierCountPattern = regexp.MustCompile(`(?i)(\d+)\s+(passed|failed|errors?|skipped)`)

// ExampleFromTask converts a TBLite task into a GEPA-compatible agent example.
func ExampleFromTask(task datasets.TBLiteTask) optimize.AgentExample {
	task = task.Normalize()
	return optimize.AgentExample{
		ID: task.TaskName,
		Inputs: map[string]interface{}{
			"task_name":    task.TaskName,
			"instruction":  task.Instruction,
			"category":     task.Category,
			"difficulty":   task.Difficulty,
			"docker_image": task.DockerImage,
		},
		Outputs: map[string]interface{}{
			"passed": true,
		},
		Metadata: map[string]interface{}{
			exampleMetadataTBLiteTaskKey: task,
		},
	}
}

// ExamplesFromTasks converts a task slice into optimization examples.
func ExamplesFromTasks(tasks []datasets.TBLiteTask) []optimize.AgentExample {
	examples := make([]optimize.AgentExample, 0, len(tasks))
	for _, task := range tasks {
		examples = append(examples, ExampleFromTask(task))
	}
	return examples
}

// GEPAEvaluatorConfig controls real-task GEPA evaluation for TBLite.
type GEPAEvaluatorConfig struct {
	RootDir           string
	KeepArtifacts     bool
	MaxTurns          int
	UseTaskContainers bool
}

type gepaEvaluator struct {
	cfg GEPAEvaluatorConfig
}

var _ optimize.AgentEvaluator = (*gepaEvaluator)(nil)

// NewGEPAEvaluator creates an evaluator that scores candidates by real TBLite task success.
func NewGEPAEvaluator(cfg GEPAEvaluatorConfig) optimize.AgentEvaluator {
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 20
	}
	if cfg.RootDir == "" {
		cfg.RootDir = filepath.Join(os.TempDir(), "dspy-tblite-gepa")
	}
	return &gepaEvaluator{cfg: cfg}
}

func (e *gepaEvaluator) Evaluate(ctx context.Context, agent optimize.OptimizableAgent, ex optimize.AgentExample) (*optimize.EvalResult, error) {
	task, err := taskFromExample(ex)
	if err != nil {
		return nil, err
	}

	taskAgent, ok := agent.(Agent)
	if !ok {
		return nil, fmt.Errorf("optimize: agent %T does not implement tblite.Agent", agent)
	}

	evalRoot := filepath.Join(e.cfg.RootDir, fmt.Sprintf("%s-%d", task.TaskName, time.Now().UnixNano()))
	if err := os.MkdirAll(evalRoot, 0o755); err != nil {
		return nil, fmt.Errorf("create evaluator root: %w", err)
	}
	if !e.cfg.KeepArtifacts {
		defer os.RemoveAll(evalRoot)
	}

	runner := NewRunner(taskAgent, RunnerConfig{
		MaxTurns:             e.cfg.MaxTurns,
		UseTaskContainers:    e.cfg.UseTaskContainers,
		RespectAgentMaxTurns: true,
	})

	result, err := runner.EvaluateTask(ctx, task, evalRoot)
	if err != nil {
		return nil, err
	}

	score := 0.0
	if result.TestResult != nil && result.TestResult.Passed {
		score = 1.0
	} else if result.TestResult != nil {
		score = verifierPassFraction(result.TestResult.Stdout, result.TestResult.Stderr)
	}

	sideInfo := &optimize.SideInfo{
		Trace:       latestTrace(agent),
		Diagnostics: map[string]interface{}{},
		Scores: map[string]float64{
			"tblite_pass": score,
		},
	}
	if result.AgentResult != nil {
		sideInfo.LatencyMS = float64(result.AgentResult.Duration) / float64(time.Millisecond)
		sideInfo.Tokens = map[string]int64{
			"prompt_tokens":     result.AgentResult.TokenUsage.PromptTokens,
			"completion_tokens": result.AgentResult.TokenUsage.CompletionTokens,
			"total_tokens":      result.AgentResult.TokenUsage.TotalTokens,
		}
		sideInfo.Diagnostics["tool_calls"] = result.AgentResult.ToolCalls
		sideInfo.Diagnostics["final_answer"] = result.AgentResult.FinalAnswer
		sideInfo.Diagnostics["trace_path"] = result.AgentResult.TracePath
	}
	if result.TestResult != nil {
		sideInfo.Diagnostics["verifier_exit_code"] = result.TestResult.ExitCode
		sideInfo.Diagnostics["verifier_stdout"] = result.TestResult.Stdout
		sideInfo.Diagnostics["verifier_stderr"] = result.TestResult.Stderr
		if result.TestResult.Passed {
			sideInfo.PassedTests = []string{"tblite_verifier"}
		} else {
			sideInfo.FailedTests = []string{"tblite_verifier"}
		}
		sideInfo.Diagnostics["verifier_pass_fraction"] = score
	}

	return &optimize.EvalResult{
		Score:    score,
		SideInfo: sideInfo,
	}, nil
}

func taskFromExample(ex optimize.AgentExample) (datasets.TBLiteTask, error) {
	if ex.Metadata == nil {
		return datasets.TBLiteTask{}, fmt.Errorf("optimize: tblite example %q missing metadata", ex.ID)
	}
	rawTask, ok := ex.Metadata[exampleMetadataTBLiteTaskKey]
	if !ok {
		return datasets.TBLiteTask{}, fmt.Errorf("optimize: tblite example %q missing task metadata", ex.ID)
	}
	task, ok := rawTask.(datasets.TBLiteTask)
	if !ok {
		return datasets.TBLiteTask{}, fmt.Errorf("optimize: tblite example %q has invalid task metadata %T", ex.ID, rawTask)
	}
	return task.Normalize(), nil
}

type lastExecutionTraceProvider interface {
	LastExecutionTrace() *agents.ExecutionTrace
}

func latestTrace(agent optimize.OptimizableAgent) *agents.ExecutionTrace {
	provider, ok := agent.(lastExecutionTraceProvider)
	if !ok {
		return nil
	}
	return provider.LastExecutionTrace()
}

func verifierPassFraction(outputs ...string) float64 {
	passed := 0
	total := 0

	for _, output := range outputs {
		matches := verifierCountPattern.FindAllStringSubmatch(output, -1)
		for _, match := range matches {
			if len(match) != 3 {
				continue
			}
			count, err := strconv.Atoi(match[1])
			if err != nil {
				continue
			}
			total += count
			if strings.EqualFold(match[2], "passed") {
				passed += count
			}
		}
	}

	if total == 0 {
		return 0
	}
	return float64(passed) / float64(total)
}
