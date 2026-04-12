package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResolveDefaultLLM_PreferenceOrder(t *testing.T) {
	global := &MockLLM{}
	runtimeLLM := &recordLLMCallCompatStub{modelID: "runtime"}
	moduleLLM := &recordLLMCallCompatStub{modelID: "module"}

	original := GetDefaultLLM()
	SetDefaultLLM(global)
	defer SetDefaultLLM(original)

	ctx := WithRuntime(context.Background(), &Runtime{DefaultLLM: runtimeLLM})

	resolved := ResolveDefaultLLM(ctx, nil)
	require.Same(t, runtimeLLM, resolved)

	info := NewModuleInfo("module", "test", Signature{}).WithLLM(moduleLLM)
	resolved = ResolveDefaultLLM(ctx, info)
	require.Same(t, moduleLLM, resolved)
}

func TestResolveTeacherLLM_UsesRuntimeBeforeGlobal(t *testing.T) {
	runtimeTeacher := &recordLLMCallCompatStub{modelID: "runtime-teacher"}
	original := GetTeacherLLM()
	SetTeacherLLM(nil)
	defer SetTeacherLLM(original)

	ctx := WithRuntime(context.Background(), &Runtime{TeacherLLM: runtimeTeacher})
	require.Same(t, runtimeTeacher, ResolveTeacherLLM(ctx))
}
