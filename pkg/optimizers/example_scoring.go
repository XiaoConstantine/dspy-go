package optimizers

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/sourcegraph/conc/pool"
)

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
// stopOnError, once a call has failed. With one worker it is a plain loop.
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
			if err != nil {
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
