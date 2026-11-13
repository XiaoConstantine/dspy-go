package modules

import (
	"context"
	"maps"
)

type predictionObserverKey struct{}

// WithPredictionObserver observes successful Predict calls, including parsed XML,
// structured JSON, and calls inside composite modules. The callback receives the
// actual predictor and shallow snapshots of its inputs and final outputs. Values
// inside those maps remain read-only. Parallel predictions may call it concurrently.
// The observer is scoped to ctx; it does not modify modules or global state.
func WithPredictionObserver(ctx context.Context, observer func(*Predict, map[string]any, map[string]any)) context.Context {
	return context.WithValue(ctx, predictionObserverKey{}, observer)
}

// withoutPredictionTrace lets a wrapper record its final parsed result instead
// of also recording its inner Predict call. Other nested predictors still trace.
func withoutPredictionTrace(ctx context.Context, predictor *Predict) context.Context {
	observer, _ := ctx.Value(predictionObserverKey{}).(func(*Predict, map[string]any, map[string]any))
	return WithPredictionObserver(ctx, func(p *Predict, inputs, outputs map[string]any) {
		if p != predictor && observer != nil {
			observer(p, inputs, outputs)
		}
	})
}

func predictionRecorder(ctx context.Context, predictor *Predict, inputs map[string]any) func(map[string]any, error) {
	observer, _ := ctx.Value(predictionObserverKey{}).(func(*Predict, map[string]any, map[string]any))
	if observer == nil {
		return nil
	}
	inputs = maps.Clone(inputs)
	return func(outputs map[string]any, err error) {
		if err == nil {
			observer(predictor, inputs, maps.Clone(outputs))
		}
	}
}
