// core/trace_test.go

package core

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTraceManager(t *testing.T) {
    ctx := WithTraceManager(context.Background())
    tm := GetTraceManager(ctx)

    // Test starting a root trace
    rootTrace := tm.StartTrace("RootModule", "Root")
    assert.Equal(t, "RootModule", rootTrace.ModuleName)
    assert.Equal(t, "Root", rootTrace.ModuleType)

    // Test starting a child trace
    childTrace := tm.StartTrace("ChildModule", "Child")
    assert.Equal(t, "ChildModule", childTrace.ModuleName)
    assert.Equal(t, "Child", childTrace.ModuleType)

    // Test ending traces
    tm.EndTrace() // End child trace
    assert.Equal(t, rootTrace, tm.CurrentTrace)

    tm.EndTrace() // End root trace
    assert.Equal(t, rootTrace, tm.CurrentTrace) // Should still be root trace

    // Test setting inputs and outputs
    inputs := map[string]interface{}{"input": "test"}
    outputs := map[string]interface{}{"output": "result"}
    rootTrace.SetInputs(inputs)
    rootTrace.SetOutputs(outputs)

    assert.Equal(t, inputs, rootTrace.Inputs)
    assert.Equal(t, outputs, rootTrace.Outputs)
    assert.True(t, rootTrace.Duration > 0)

    // Test error setting
    err := errors.New("test error")
    rootTrace.SetError(err)
    assert.Equal(t, err, rootTrace.Error)
}
