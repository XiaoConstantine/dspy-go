package commands

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPartitionTBLiteTasks(t *testing.T) {
	tasks := []datasets.TBLiteTask{
		{TaskName: "t1"},
		{TaskName: "t2"},
		{TaskName: "t3"},
		{TaskName: "t4"},
		{TaskName: "t5"},
	}

	train, validation, test, err := partitionTBLiteTasks(tasks, 0.2, 0.2)
	require.NoError(t, err)
	assert.Len(t, train, 3)
	assert.Len(t, validation, 1)
	assert.Len(t, test, 1)
	assert.Equal(t, "t1", train[0].TaskName)
	assert.Equal(t, "t4", validation[0].TaskName)
	assert.Equal(t, "t5", test[0].TaskName)
}

func TestPartitionTBLiteTasks_SmallSliceKeepsTrainingRoom(t *testing.T) {
	tasks := []datasets.TBLiteTask{
		{TaskName: "t1"},
		{TaskName: "t2"},
		{TaskName: "t3"},
		{TaskName: "t4"},
	}

	train, validation, test, err := partitionTBLiteTasks(tasks, 0.34, 0.25)
	require.NoError(t, err)
	assert.Len(t, train, 2)
	assert.Len(t, validation, 1)
	assert.Len(t, test, 1)
	assert.Equal(t, "t3", validation[0].TaskName)
	assert.Equal(t, "t4", test[0].TaskName)
}

func TestTBLiteBenchmarkCommand_DefaultGEPAEvalConcurrency(t *testing.T) {
	cmd := newTBLiteBenchmarkCommand(defaultTerminalTaskAgentFactory)
	flag := cmd.Flags().Lookup("gepa-eval-concurrency")
	require.NotNil(t, flag)
	assert.Equal(t, "4", flag.DefValue)
}
