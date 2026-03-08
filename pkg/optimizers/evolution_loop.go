package optimizers

import (
	"context"
	"fmt"
)

// EvolutionLoopConfig configures the shared GEPA-style generation loop.
type EvolutionLoopConfig struct {
	MaxGenerations int
	ReflectionFreq int
	PhaseName      string
}

// EvolutionLoopHooks injects domain-specific behavior into the shared loop.
type EvolutionLoopHooks struct {
	Initialize         func(ctx context.Context) error
	BeforeGeneration   func(ctx context.Context, generation int) error
	EvaluateGeneration func(ctx context.Context, generation int) error
	// AfterEvaluation runs immediately after EvaluateGeneration for the same generation.
	// Callers can use it to consume state populated during evaluation, such as cached
	// fitness maps needed by archive or frontier updates.
	AfterEvaluation func(ctx context.Context, generation int) error
	Reflect         func(ctx context.Context, generation int) error
	OnNonFatalError func(ctx context.Context, stage string, generation int, err error)
	HasConverged    func(ctx context.Context, generation int) bool
	// Evolve is treated as fatal if it returns an error. Hooks that want retry or
	// best-effort behavior should absorb those failures internally.
	Evolve         func(ctx context.Context, generation int) error
	ReportProgress func(phase string, current, total int)
	Finalize       func(ctx context.Context) error
}

// RunEvolutionLoop executes the shared GEPA-style generation loop once the caller
// provides initialization, evaluation, and evolution callbacks.
func RunEvolutionLoop(ctx context.Context, cfg EvolutionLoopConfig, hooks EvolutionLoopHooks) error {
	if cfg.MaxGenerations <= 0 {
		return fmt.Errorf("optimizers: max generations must be positive")
	}
	if hooks.EvaluateGeneration == nil {
		return fmt.Errorf("optimizers: evaluate generation hook is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	if hooks.Initialize != nil {
		if err := hooks.Initialize(ctx); err != nil {
			return err
		}
	}

	for generation := 0; generation < cfg.MaxGenerations; generation++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		if hooks.BeforeGeneration != nil {
			if err := hooks.BeforeGeneration(ctx, generation); err != nil {
				return err
			}
		}

		if err := hooks.EvaluateGeneration(ctx, generation); err != nil {
			return err
		}

		if hooks.AfterEvaluation != nil {
			if err := hooks.AfterEvaluation(ctx, generation); err != nil {
				return err
			}
		}

		if cfg.ReflectionFreq > 0 && generation > 0 && generation%cfg.ReflectionFreq == 0 && hooks.Reflect != nil {
			if err := hooks.Reflect(ctx, generation); err != nil {
				if hooks.OnNonFatalError != nil {
					hooks.OnNonFatalError(ctx, "reflection", generation, err)
				}
			}
		}

		if hooks.HasConverged != nil && hooks.HasConverged(ctx, generation) {
			break
		}

		if generation < cfg.MaxGenerations-1 && hooks.Evolve != nil {
			if err := hooks.Evolve(ctx, generation); err != nil {
				return err
			}
		}

		if hooks.ReportProgress != nil {
			hooks.ReportProgress(cfg.phaseName(), generation+1, cfg.MaxGenerations)
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	if hooks.Finalize != nil {
		if err := hooks.Finalize(ctx); err != nil {
			return err
		}
	}

	return nil
}

func (cfg EvolutionLoopConfig) phaseName() string {
	if cfg.PhaseName == "" {
		return "Evolution"
	}

	return cfg.PhaseName
}
