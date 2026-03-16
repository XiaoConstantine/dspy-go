package tblite

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseDockerfileWorkdir_UsesLastResolvedWorkdir(t *testing.T) {
	dockerfile := `
FROM python:3.13
WORKDIR /app
WORKDIR services/api
`

	assert.Equal(t, "/app/services/api", parseDockerfileWorkdir(dockerfile))
}

func TestParseDockerfileWorkdir_EmptyWhenNoWorkdirPresent(t *testing.T) {
	dockerfile := `
FROM python:3.13
RUN echo hello
`

	assert.Equal(t, "", parseDockerfileWorkdir(dockerfile))
}

func TestDetectContainerEnvironmentRoot_DefaultsWhenDockerfileMissing(t *testing.T) {
	envDir := t.TempDir()
	root, err := detectContainerEnvironmentRoot(&MaterializedTask{EnvironmentDir: envDir})
	require.NoError(t, err)
	assert.Equal(t, containerEnvDir, root)
}

func TestDetectContainerEnvironmentRoot_UsesDockerfileWorkdir(t *testing.T) {
	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(envDir, "Dockerfile"), []byte("FROM python:3.13\nWORKDIR /app\n"), 0o644))

	root, err := detectContainerEnvironmentRoot(&MaterializedTask{EnvironmentDir: envDir})
	require.NoError(t, err)
	assert.Equal(t, "/app", root)
}

func TestDockerTaskRuntime_ContainerPathForHost_PrefersContainerEnvironmentRoot(t *testing.T) {
	taskRoot := t.TempDir()
	envRoot := filepath.Join(taskRoot, "environment")
	testsRoot := filepath.Join(taskRoot, "tests")
	require.NoError(t, os.MkdirAll(filepath.Join(envRoot, "api"), 0o755))
	require.NoError(t, os.MkdirAll(testsRoot, 0o755))

	runtime := &dockerTaskRuntime{
		taskRoot:         taskRoot,
		environmentRoot:  envRoot,
		containerEnvRoot: "/app",
	}

	containerPath, err := runtime.containerPathForHost(filepath.Join(envRoot, "api"))
	require.NoError(t, err)
	assert.Equal(t, "/app/api", containerPath)

	containerPath, err = runtime.containerPathForHost(testsRoot)
	require.NoError(t, err)
	assert.Equal(t, "/task/tests", containerPath)
}
