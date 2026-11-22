package modules

import (
	"context"
	"maps"
	"sync"
)

type predictionObserverKey struct{}
type predictionMetaObserverKey struct{}

// PredictionTraceMeta carries adapter execution metadata for a Predict call.
type PredictionTraceMeta struct {
	AdapterName       string
	AdapterVersion    string
	FallbackUsed      bool
	SchemaEnforcement string
	CacheHit          bool
}

// WithPredictionTraceObserver observes successful Predict calls made through
// an adapter (WithAdapter), along with adapter metadata such as whether the
// JSON fallback was used. It is not called for a Predict without an adapter;
// WithPredictionObserver observes those too.
func WithPredictionTraceObserver(ctx context.Context, observer func(*Predict, map[string]any, map[string]any, PredictionTraceMeta)) context.Context {
	return context.WithValue(ctx, predictionMetaObserverKey{}, observer)
}

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

type bufferedPrediction struct {
	predictor       *Predict
	inputs, outputs map[string]any
}

// bufferPredictionTrace holds the Predict calls made under the returned
// context until publish passes them to ctx's observer, so a wrapper that runs
// several attempts, such as Refine, reports only the attempt it returns.
func bufferPredictionTrace(ctx context.Context) (context.Context, func()) {
	observer, _ := ctx.Value(predictionObserverKey{}).(func(*Predict, map[string]any, map[string]any))
	if observer == nil {
		return ctx, func() {}
	}
	var mu sync.Mutex
	var calls []bufferedPrediction
	ctx = WithPredictionObserver(ctx, func(p *Predict, inputs, outputs map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		calls = append(calls, bufferedPrediction{predictor: p, inputs: inputs, outputs: outputs})
	})
	return ctx, func() {
		mu.Lock()
		defer mu.Unlock()
		for _, call := range calls {
			observer(call.predictor, call.inputs, call.outputs)
		}
	}
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

func recordPredictionTraceMeta(ctx context.Context, predictor *Predict, inputs, outputs map[string]any, meta PredictionTraceMeta) {
	observer, _ := ctx.Value(predictionMetaObserverKey{}).(func(*Predict, map[string]any, map[string]any, PredictionTraceMeta))
	if observer == nil {
		return
	}
	observer(predictor, maps.Clone(inputs), maps.Clone(outputs), meta)
}
