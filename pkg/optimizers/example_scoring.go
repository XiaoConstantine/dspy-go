package optimizers

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/sourcegraph/conc/pool"
)

// EvalPolicy is how an optimizer treats output errors while it evaluates a
// program, as failure_score and max_errors do in DSPy's Evaluate. An output
// error is a model reply that was refused, filtered, truncated, or could not
// be parsed (errors.IsOutputError); programs raise them when their predictors
// use a structured-output adapter.
//
// An example that ends in an output error scores FailureScore. Once one
// evaluation has seen more than MaxErrors of them, the optimizer stops with an
// error that reports the count. So a MaxErrors of zero, which is also the zero
// value of EvalPolicy, stops at the first output error; a negative MaxErrors
// sets no limit. Any other error is handled as each optimizer always has.
type EvalPolicy struct {
	FailureScore float64 `json:"failure_score"`
	MaxErrors    int     `json:"max_errors"`
}

// DefaultEvalPolicy returns the policy every optimizer constructor installs:
// an output error scores 0, and more than 10 of them stop the run.
func DefaultEvalPolicy() EvalPolicy {
	return EvalPolicy{
		FailureScore: 0.0,
		MaxErrors:    10,
	}
}

// outputErrorBudget applies an EvalPolicy to the output errors of one
// evaluation. It is safe for concurrent use.
type outputErrorBudget struct {
	policy EvalPolicy
	mu     sync.Mutex
	count  int
	first  error
}

func newOutputErrorBudget(policy EvalPolicy) *outputErrorBudget {
	return &outputErrorBudget{policy: policy}
}

// record reports whether err is an output error, which the caller then scores
// as policy.FailureScore, and counts it against policy.MaxErrors.
func (b *outputErrorBudget) record(err error) bool {
	if !dspyerrors.IsOutputError(err) {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.count++
	if b.first == nil {
		b.first = err
	}
	return true
}

// exceeded returns the error that stops the run once more than
// policy.MaxErrors output errors were recorded, and nil before that. A negative
// policy.MaxErrors never stops the run.
func (b *outputErrorBudget) exceeded() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.policy.MaxErrors < 0 || b.count <= b.policy.MaxErrors {
		return nil
	}
	return &maxErrorsError{count: b.count, limit: b.policy.MaxErrors, first: b.first}
}

// ErrMaxErrors matches, with errors.Is, the error an optimizer returns when it
// stops because an evaluation saw more output errors than EvalPolicy.MaxErrors.
var ErrMaxErrors = errors.New("evaluation exceeded max_errors")

func (e *maxErrorsError) Is(target error) bool { return target == ErrMaxErrors }

// maxErrorsError reports an evaluation stopped by EvalPolicy.MaxErrors. It
// wraps the first output error, so errors.Is matches its kind.
type maxErrorsError struct {
	count, limit int
	first        error
}

func (e *maxErrorsError) Error() string {
	return fmt.Sprintf("evaluation exceeded max_errors (%d output errors > limit %d): %v", e.count, e.limit, e.first)
}

func (e *maxErrorsError) Unwrap() error { return e.first }

// aggregateEvalScores averages per-example outcomes under policy. An output
// error scores policy.FailureScore; any other error stops the evaluation.
func aggregateEvalScores(ctx context.Context, results []exampleScore, policy EvalPolicy) (float64, error) {
	if len(results) == 0 {
		return 0, fmt.Errorf("no examples evaluated")
	}

	budget := newOutputErrorBudget(policy)
	var total float64
	for _, r := range results {
		switch {
		case r.Err == nil:
			total += r.Score
		case budget.record(r.Err):
			total += policy.FailureScore
		default:
			return 0, fmt.Errorf("failed to evaluate example: %w", r.Err)
		}
	}
	// Examples skipped after cancellation carry no error of their own.
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := budget.exceeded(); err != nil {
		return 0, err
	}
	return total / float64(len(results)), nil
}

// exampleScore is the outcome of scoring one example. Ran is false for an
// example that was never started.
type exampleScore struct {
	Score float64
	Err   error
	Ran   bool
}

// scoreExamples calls score for every example, at most workers at a time, and
// returns the outcomes in example order, so aggregates do not depend on
// completion order. It starts no further examples once ctx is done, or, with
// stopOnError, once a call has failed with anything but an output error, which
// an EvalPolicy scores instead. With one worker it is a plain loop.
// Above one worker, score, and the program and metric it calls, must be safe
// for concurrent use.
func scoreExamples(ctx context.Context, workers int, stopOnError bool, examples []core.Example, score func(core.Example) (float64, error)) []exampleScore {
	results := make([]exampleScore, len(examples))
	var failed atomic.Bool
	p := pool.New().WithMaxGoroutines(max(workers, 1))
	for i, example := range examples {
		p.Go(func() {
			if ctx.Err() != nil || (stopOnError && failed.Load()) {
				return
			}
			value, err := score(example)
			if err != nil && !dspyerrors.IsOutputError(err) {
				failed.Store(true)
			}
			results[i] = exampleScore{Score: value, Err: err, Ran: true}
		})
	}
	p.Wait()
	return results
}

// serialScorer runs one Score call at a time. The legacy Compile entry points
// use it where metric calls never overlapped, because a core.Metric was never
// required to be safe for concurrent use.
type serialScorer struct {
	mu     sync.Mutex
	scorer core.Scorer
}

func serialize(scorer core.Scorer) core.Scorer {
	return &serialScorer{scorer: scorer}
}

// serializeMetric is core.MetricScorer for optimizers whose metric calls never
// overlapped.
func serializeMetric(metric core.Metric) core.Scorer {
	if metric == nil {
		return nil
	}
	return serialize(core.MetricScorer(metric))
}

func (s *serialScorer) Score(ctx context.Context, example core.Example, prediction map[string]any) (float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scorer.Score(ctx, example, prediction)
}

// requireRebindableProgram rejects a program whose clones would still run the
// original modules, because an optimizer's changes to them would never execute.
func requireRebindableProgram(program core.Program) error {
	if len(program.Modules) > 0 && program.Forward != nil && !program.HasForwardFactory() {
		return errors.New("program was built with core.NewProgram: its forward function keeps calling the original modules, so optimized instructions and demonstrations would have no effect; build it with core.NewProgramWithForwardFactory")
	}
	return nil
}
