package datasets

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTBLiteTaskUnmarshalJSON_AcceptsHuggingFaceShapes(t *testing.T) {
	var task TBLiteTask
	err := json.Unmarshal([]byte(`{
		"task_name":"acl-permissions-inheritance",
		"instruction":"do the thing",
		"docker_image":"nousresearch/tblite-acl:latest",
		"category":"system-administration",
		"difficulty":"medium",
		"tags":"[\"system\",\"acl\"]",
		"agent_timeout_sec":900.0,
		"test_timeout_sec":180.0,
		"environment_tar":"ZW52",
		"tests_tar":"dGVzdHM=",
		"test_sh":"#!/bin/bash"
	}`), &task)
	require.NoError(t, err)

	assert.Equal(t, "acl-permissions-inheritance", task.TaskName)
	assert.Equal(t, []string{"system", "acl"}, task.Tags)
	assert.Equal(t, 900, task.AgentTimeoutSec)
	assert.Equal(t, 180, task.TestTimeoutSec)
}

func TestLoadTBLiteTasksFromFile_NormalizesTasks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tblite.json")
	content := `[{
		"task_name":"broken-python",
		"instruction":"fix it",
		"docker_image":"nousresearch/tblite-broken-python:latest",
		"category":"coding",
		"difficulty":"easy",
		"tags":["python","debugging"],
		"agent_timeout_sec":600,
		"test_timeout_sec":120,
		"environment_tar":"ZW52",
		"tests_tar":"dGVzdHM=",
		"test_sh":"pytest -q"
	}]`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	tasks, err := LoadTBLiteTasksFromFile(path)
	require.NoError(t, err)
	require.Len(t, tasks, 1)
	assert.Equal(t, "broken-python", tasks[0].TaskName)
	assert.Equal(t, []string{"python", "debugging"}, tasks[0].Tags)
	assert.Equal(t, 600, tasks[0].AgentTimeoutSec)
	assert.Equal(t, 120, tasks[0].TestTimeoutSec)
}

func TestLoadTBLiteTaskSelectionFromFile_TaskNamesManifest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "selection.json")
	content := `{
		"label":"focused-coding",
		"split":"train",
		"task_names":["task-a", "task-b", "task-a"]
	}`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	selection, err := LoadTBLiteTaskSelectionFromFile(path)
	require.NoError(t, err)
	require.NotNil(t, selection)
	assert.Equal(t, "focused-coding", selection.Label)
	assert.Equal(t, "train", selection.Split)
	assert.Equal(t, []string{"task-a", "task-b"}, selection.TaskNames)
	assert.Empty(t, selection.Tasks)
}

func TestLoadTBLiteTaskSelectionFromFile_AcceptsTaskArray(t *testing.T) {
	path := filepath.Join(t.TempDir(), "selection.json")
	content := `[{
		"task_name":"task-a",
		"instruction":"do it",
		"docker_image":"img:latest",
		"category":"coding",
		"difficulty":"easy",
		"test_sh":"#!/bin/sh"
	}]`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	selection, err := LoadTBLiteTaskSelectionFromFile(path)
	require.NoError(t, err)
	require.NotNil(t, selection)
	require.Len(t, selection.Tasks, 1)
	assert.Equal(t, "task-a", selection.Tasks[0].TaskName)
	assert.Nil(t, selection.TaskNames)
}

func TestFetchTBLiteTasksByNamesContext_PreservesRequestedOrder(t *testing.T) {
	originalEndpoint := tbliteRowsEndpoint
	defer func() { tbliteRowsEndpoint = originalEndpoint }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"rows":[
				{"row":{"task_name":"task-a","instruction":"a","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"}},
				{"row":{"task_name":"task-b","instruction":"b","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"}},
				{"row":{"task_name":"task-c","instruction":"c","docker_image":"img:latest","category":"coding","difficulty":"easy","test_sh":"#!/bin/sh"}}
			]
		}`))
	}))
	defer server.Close()
	tbliteRowsEndpoint = server.URL

	tasks, err := FetchTBLiteTasksByNamesContext(context.Background(), "train", []string{"task-c", "task-a"})
	require.NoError(t, err)
	require.Len(t, tasks, 2)
	assert.Equal(t, "task-c", tasks[0].TaskName)
	assert.Equal(t, "task-a", tasks[1].TaskName)
}

func TestSliceTBLiteTasks_UsesDeterministicOffset(t *testing.T) {
	tasks := []TBLiteTask{
		{TaskName: "task-a"},
		{TaskName: "task-b"},
		{TaskName: "task-c"},
	}

	sliced := SliceTBLiteTasks(tasks, 1, 2)

	require.Len(t, sliced, 2)
	assert.Equal(t, "task-b", sliced[0].TaskName)
	assert.Equal(t, "task-c", sliced[1].TaskName)
}

func TestTBLiteTaskDecodeArchives(t *testing.T) {
	task := TBLiteTask{
		EnvironmentTar: base64.StdEncoding.EncodeToString([]byte("env-bytes")),
		TestsTar:       base64.StdEncoding.EncodeToString([]byte("test-bytes")),
	}

	envArchive, err := task.DecodeEnvironmentArchive()
	require.NoError(t, err)
	assert.Equal(t, []byte("env-bytes"), envArchive)

	testsArchive, err := task.DecodeTestsArchive()
	require.NoError(t, err)
	assert.Equal(t, []byte("test-bytes"), testsArchive)
}

func TestFetchTBLiteTasksFromHuggingFaceRangeContext_RespectsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := FetchTBLiteTasksFromHuggingFaceRangeContext(ctx, "train", 0, 1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fetch TBLite rows")
}
