package core

import "context"

// Runtime holds explicit runtime-scoped dependencies for module execution.
// It is intended as a compatibility-friendly alternative to relying solely on
// package-level globals when resolving models during data-plane execution.
type Runtime struct {
	DefaultLLM LLM
	TeacherLLM LLM
}

type runtimeContextKey struct{}

var runtimeKey = &runtimeContextKey{}

// WithRuntime attaches a runtime to the context for downstream resolution.
func WithRuntime(ctx context.Context, runtime *Runtime) context.Context {
	if runtime == nil {
		return ctx
	}
	return context.WithValue(ctx, runtimeKey, runtime)
}

// RuntimeFromContext retrieves a runtime from context when present.
func RuntimeFromContext(ctx context.Context) *Runtime {
	runtime, _ := ctx.Value(runtimeKey).(*Runtime)
	return runtime
}

// ResolveDefaultLLM resolves the model to use for module execution in the
// following order:
// 1. module-local LLM from ModuleInfo
// 2. explicit runtime on the context
// 3. package-level default LLM for backward compatibility.
func ResolveDefaultLLM(ctx context.Context, info *ModuleInfo) LLM {
	if info != nil && info.LLM != nil {
		return info.LLM
	}
	if runtime := RuntimeFromContext(ctx); runtime != nil && runtime.DefaultLLM != nil {
		return runtime.DefaultLLM
	}
	return GetDefaultLLM()
}

// ResolveTeacherLLM resolves the teacher model from an explicit runtime first,
// then falls back to the package-level compatibility default.
func ResolveTeacherLLM(ctx context.Context) LLM {
	if runtime := RuntimeFromContext(ctx); runtime != nil && runtime.TeacherLLM != nil {
		return runtime.TeacherLLM
	}
	return GetTeacherLLM()
}
