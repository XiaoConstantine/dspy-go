package commands

import (
	"context"
	"os"
	"path/filepath"
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

func TestTBLiteBenchmarkCommand_DefaultGEPASearchControls(t *testing.T) {
	cmd := newTBLiteBenchmarkCommand(defaultTerminalTaskAgentFactory)

	searchBatchFlag := cmd.Flags().Lookup("gepa-search-batch-size")
	require.NotNil(t, searchBatchFlag)
	assert.Equal(t, "4", searchBatchFlag.DefValue)

	stagnationFlag := cmd.Flags().Lookup("gepa-stagnation-limit-minutes")
	require.NotNil(t, stagnationFlag)
	assert.Equal(t, "60", stagnationFlag.DefValue)
}

func TestResolveTBLiteTaskSource_UsesInlineTasksFromFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tasks.json")
	content := `{
		"label":"focused-slice",
		"split":"train",
		"tasks":[
			{"task_name":"task-a","instruction":"a","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"},
			{"task_name":"task-b","instruction":"b","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"}
		]
	}`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	source, err := resolveTBLiteTaskSource(context.Background(), tbliteTaskSourceConfig{
		Split:     "train",
		Limit:     5,
		Label:     "",
		TasksFile: path,
	})
	require.NoError(t, err)
	require.NotNil(t, source)
	assert.Equal(t, "focused-slice", source.Label)
	assert.Equal(t, "train", source.Split)
	assert.Equal(t, 0, source.Offset)
	assert.Equal(t, 2, source.Limit)
	require.Len(t, source.Tasks, 2)
	assert.Equal(t, "task-a", source.Tasks[0].TaskName)
	assert.Equal(t, "task-b", source.Tasks[1].TaskName)
}

func TestResolveTBLiteTaskSource_RejectsOffsetOrLimitWithTasksFile(t *testing.T) {
	_, err := resolveTBLiteTaskSource(context.Background(), tbliteTaskSourceConfig{
		Split:          "train",
		Offset:         10,
		Limit:          5,
		TasksFile:      "tasks.json",
		OffsetExplicit: true,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "--tasks-file cannot be combined with --offset or --limit")
}

func TestResolveTBLiteTaskSource_RejectsSplitConflict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tasks.json")
	content := `{
		"split":"validation",
		"tasks":[
			{"task_name":"task-a","instruction":"a","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"}
		]
	}`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	_, err := resolveTBLiteTaskSource(context.Background(), tbliteTaskSourceConfig{
		Split:         "train",
		TasksFile:     path,
		SplitExplicit: true,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "conflicts with tasks file split")
}

func TestDedupeTBLiteTasksByName_PreservesFirstOccurrence(t *testing.T) {
	tasks := []datasets.TBLiteTask{
		{TaskName: "task-a", Instruction: "first"},
		{TaskName: "task-b", Instruction: "second"},
		{TaskName: "task-a", Instruction: "third"},
	}

	deduped := dedupeTBLiteTasksByName(tasks)
	require.Len(t, deduped, 2)
	assert.Equal(t, "first", deduped[0].Instruction)
	assert.Equal(t, "second", deduped[1].Instruction)
}
