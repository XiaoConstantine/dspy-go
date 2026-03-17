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

func TestMaterializeTask_RejectsTaskNameTraversal(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName: "../../escape",
	}

	_, err := MaterializeTask(task, t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must not contain path separators")
}

func TestMaterializeTask_RejectsLinkEntries(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName: "bad-link",
		EnvironmentTar: mustEncodedTarGzEntries(t, []testTarEntry{
			{Name: "workspace/link", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"},
		}),
	}

	_, err := MaterializeTask(task, t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "link entry")
}

func TestExtractArchiveWithLimits_RejectsOversizedFile(t *testing.T) {
	payload := mustEncodedTarGzEntries(t, []testTarEntry{
		{Name: "workspace/main.py", Content: "12345", Typeflag: tar.TypeReg},
	})

	err := extractArchiveWithLimits(payload, t.TempDir(), 4, 16)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "oversized entry")
}

func TestExtractArchiveWithLimits_RejectsOversizedArchive(t *testing.T) {
	payload := mustEncodedTarGzEntries(t, []testTarEntry{
		{Name: "workspace/a.txt", Content: "123", Typeflag: tar.TypeReg},
		{Name: "workspace/b.txt", Content: "456", Typeflag: tar.TypeReg},
	})

	err := extractArchiveWithLimits(payload, t.TempDir(), 8, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "total limit")
}

func mustEncodedTarGz(t *testing.T, files map[string]string) string {
	t.Helper()

	entries := make([]testTarEntry, 0, len(files))
	for name, content := range files {
		entries = append(entries, testTarEntry{Name: name, Content: content, Typeflag: tar.TypeReg})
	}
	return mustEncodedTarGzEntries(t, entries)
}

type testTarEntry struct {
	Name     string
	Content  string
	Typeflag byte
	Linkname string
}

func mustEncodedTarGzEntries(t *testing.T, entries []testTarEntry) string {
	t.Helper()

	var buffer bytes.Buffer
	gzipWriter := gzip.NewWriter(&buffer)
	tarWriter := tar.NewWriter(gzipWriter)

	for _, entry := range entries {
		payload := []byte(entry.Content)
		typeflag := entry.Typeflag
		if typeflag == 0 {
			typeflag = tar.TypeReg
		}
		header := &tar.Header{
			Name:     entry.Name,
			Mode:     0o644,
			Size:     int64(len(payload)),
			Typeflag: typeflag,
			Linkname: entry.Linkname,
		}
		require.NoError(t, tarWriter.WriteHeader(header))
		if typeflag == tar.TypeReg || typeflag == tar.TypeRegA {
			_, err := tarWriter.Write(payload)
			require.NoError(t, err)
		}
	}

	require.NoError(t, tarWriter.Close())
	require.NoError(t, gzipWriter.Close())
	return base64.StdEncoding.EncodeToString(buffer.Bytes())
}
