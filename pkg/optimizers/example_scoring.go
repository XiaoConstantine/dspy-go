package optimizers

import (
	"context"
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
