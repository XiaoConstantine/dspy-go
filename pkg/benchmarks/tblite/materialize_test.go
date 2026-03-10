package tblite

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMaterializeTask_WritesTaskLayout(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:       "broken-python",
		Instruction:    "Fix the task",
		EnvironmentTar: mustEncodedTarGz(t, map[string]string{"workspace/main.py": "print('hello')\n"}),
		TestsTar:       mustEncodedTarGz(t, map[string]string{"tests/test_main.py": "def test_ok():\n    assert True\n"}),
		TestScript:     "#!/bin/bash\necho run\n",
	}

	materialized, err := MaterializeTask(task, t.TempDir())
	require.NoError(t, err)

	assert.FileExists(t, filepath.Join(materialized.EnvironmentDir, "workspace", "main.py"))
	assert.FileExists(t, filepath.Join(materialized.TestsDir, "tests", "test_main.py"))
	assert.FileExists(t, materialized.TestScriptPath)
	assert.FileExists(t, materialized.InstructionPath)

	instructionBytes, err := os.ReadFile(materialized.InstructionPath)
	require.NoError(t, err)
	assert.Equal(t, "Fix the task", string(instructionBytes))
}

func TestMaterializeTask_RejectsPathTraversal(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:       "bad-archive",
		EnvironmentTar: mustEncodedTarGz(t, map[string]string{"../escape.txt": "nope"}),
	}

	_, err := MaterializeTask(task, t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "path traversal")
}

func mustEncodedTarGz(t *testing.T, files map[string]string) string {
	t.Helper()

	var buffer bytes.Buffer
	gzipWriter := gzip.NewWriter(&buffer)
	tarWriter := tar.NewWriter(gzipWriter)

	for name, content := range files {
		payload := []byte(content)
		header := &tar.Header{
			Name: name,
			Mode: 0o644,
			Size: int64(len(payload)),
		}
		require.NoError(t, tarWriter.WriteHeader(header))
		_, err := tarWriter.Write(payload)
		require.NoError(t, err)
	}

	require.NoError(t, tarWriter.Close())
	require.NoError(t, gzipWriter.Close())
	return base64.StdEncoding.EncodeToString(buffer.Bytes())
}
