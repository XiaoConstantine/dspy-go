package optimize

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"sort"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ComparisonResult captures deterministic output-comparison details.
type ComparisonResult struct {
	Score       float64
	Scores      map[string]float64
	Diagnostics map[string]interface{}
	PassedTests []string
	FailedTests []string
}

// OutputComparator compares an agent execution result against an example's expectations.
type OutputComparator interface {
	Compare(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error)
}

// OutputComparatorFunc adapts a function to the OutputComparator interface.
type OutputComparatorFunc func(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error)

// Compare implements OutputComparator.
func (f OutputComparatorFunc) Compare(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error) {
	return f(ex, actual)
}

// DeterministicEvaluator runs an agent on concrete examples and attaches structured
// side information suitable for downstream optimization.
type DeterministicEvaluator struct {
	Comparator OutputComparator
}

// NewDeterministicEvaluator builds a deterministic evaluator with the provided comparator.
// When comparator is nil, it falls back to exact key/value output matching.
func NewDeterministicEvaluator(comparator OutputComparator) *DeterministicEvaluator {
	if comparator == nil {
		comparator = ExactMatchComparator{}
	}

	return &DeterministicEvaluator{
		Comparator: comparator,
	}
}

type lastExecutionTraceProvider interface {
	LastExecutionTrace() *agents.ExecutionTrace
}

// Evaluate executes the agent once and converts the outcome into score plus ASI.
// Agent execution failures are treated as candidate failures, not evaluator failures.
func (e *DeterministicEvaluator) Evaluate(ctx context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
	if agent == nil {
		return nil, fmt.Errorf("optimize: nil agent")
	}
	if e == nil {
		return nil, fmt.Errorf("optimize: nil evaluator")
	}

	comparator := e.Comparator
	if comparator == nil {
		comparator = ExactMatchComparator{}
	}

	startedAt := time.Now()
	output, execErr := agent.Execute(ctx, maps.Clone(ex.Inputs))
	trace := latestExecutionTrace(agent)
	latency := time.Since(startedAt)
	if trace != nil && trace.ProcessingTime > 0 {
		latency = trace.ProcessingTime
	}

	sideInfo := &SideInfo{
		Trace:       trace,
		Diagnostics: make(map[string]interface{}),
		Scores:      make(map[string]float64),
		LatencyMS:   float64(latency) / float64(time.Millisecond),
	}

	if trace != nil {
		sideInfo.Tokens = maps.Clone(trace.TokenUsage)
		sideInfo.Diagnostics["trace_status"] = string(trace.Status)
		if trace.TerminationCause != "" {
			sideInfo.Diagnostics["termination_cause"] = trace.TerminationCause
		}
		if len(trace.ToolUsageCount) > 0 {
			sideInfo.Diagnostics["tool_usage_count"] = maps.Clone(trace.ToolUsageCount)
		}
	}

	if execErr != nil {
		sideInfo.Diagnostics["execution_error"] = execErr.Error()
		sideInfo.Scores["execution_success"] = 0
		return &EvalResult{
			Score:    0,
			SideInfo: sideInfo,
		}, nil
	}

	sideInfo.Scores["execution_success"] = 1
	comparison, err := comparator.Compare(ex, output)
	if err != nil {
		sideInfo.Diagnostics["comparison_error"] = err.Error()
		sideInfo.Scores["comparison_success"] = 0
		return &EvalResult{
			Score:    0,
			SideInfo: sideInfo,
		}, nil
	}

	sideInfo.Scores["comparison_success"] = 1
	mergeFloatScores(sideInfo.Scores, comparison.Scores)
	mergeDiagnostics(sideInfo.Diagnostics, comparison.Diagnostics)
	sideInfo.PassedTests = append([]string(nil), comparison.PassedTests...)
	sideInfo.FailedTests = append([]string(nil), comparison.FailedTests...)

	return &EvalResult{
		Score:    comparison.Score,
		SideInfo: sideInfo,
	}, nil
}

func latestExecutionTrace(agent OptimizableAgent) *agents.ExecutionTrace {
	provider, ok := agent.(lastExecutionTraceProvider)
	if !ok {
		return nil
	}

	return provider.LastExecutionTrace()
}

func mergeFloatScores(dst map[string]float64, src map[string]float64) {
	for key, value := range src {
		dst[key] = value
	}
}

func mergeDiagnostics(dst map[string]interface{}, src map[string]interface{}) {
	for key, value := range src {
		dst[key] = value
	}
}

// ExactMatchComparator scores outputs by exact per-key equality against the example's Outputs map.
// If no expected outputs are provided, any successful execution receives a score of 1.0.
//
// Matching uses reflect.DeepEqual, so callers should expect Go's normal type
// sensitivity here. For example, int(1) and float64(1) do not match.
type ExactMatchComparator struct{}

// Compare implements OutputComparator.
func (ExactMatchComparator) Compare(ex AgentExample, actual map[string]interface{}) (*ComparisonResult, error) {
	if len(ex.Outputs) == 0 {
		return &ComparisonResult{
			Score:  1,
			Scores: map[string]float64{"output_match": 1},
		}, nil
	}

	keys := make([]string, 0, len(ex.Outputs))
	for key := range ex.Outputs {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	matched := 0
	scores := make(map[string]float64, len(keys)+1)
	passed := make([]string, 0, len(keys))
	failed := make([]string, 0, len(keys))
	mismatches := make(map[string]interface{})

	for _, key := range keys {
		expected := ex.Outputs[key]
		actualValue, ok := actual[key]
		testName := fmt.Sprintf("output:%s", key)

		if ok && reflect.DeepEqual(actualValue, expected) {
			matched++
			passed = append(passed, testName)
			scores[testName] = 1
			continue
		}

		failed = append(failed, testName)
		scores[testName] = 0
		mismatch := map[string]interface{}{
			"expected": expected,
		}
		if ok {
			mismatch["actual"] = actualValue
		} else {
			mismatch["missing"] = true
		}
		mismatches[key] = mismatch
	}

	score := float64(matched) / float64(len(keys))
	scores["output_match"] = score

	diagnostics := make(map[string]interface{})
	if len(mismatches) > 0 {
		diagnostics["mismatches"] = mismatches
	}

	return &ComparisonResult{
		Score:       score,
		Scores:      scores,
		Diagnostics: diagnostics,
		PassedTests: passed,
		FailedTests: failed,
	}, nil
}
