package datasets

import (
	"encoding/base64"
	"encoding/json"
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
