package tblite

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
)

const (
	containerTaskRoot = "/task"
	containerEnvDir   = "/task/environment"
	containerTestsDir = "/task/tests"
	defaultShellPath  = "/bin/bash"
)

type dockerTaskRuntime struct {
	containerID      string
	taskRoot         string
	environmentRoot  string
	containerEnvRoot string
}

func startDockerTaskRuntime(ctx context.Context, task *MaterializedTask, image string) (*dockerTaskRuntime, error) {
	if task == nil {
		return nil, fmt.Errorf("materialized task is required")
	}
	if strings.TrimSpace(image) == "" {
		return nil, fmt.Errorf("docker image is required")
	}

	taskRoot, err := filepath.Abs(task.RootDir)
	if err != nil {
		return nil, fmt.Errorf("resolve task root: %w", err)
	}
	environmentRoot, err := filepath.Abs(task.EnvironmentDir)
	if err != nil {
		return nil, fmt.Errorf("resolve environment root: %w", err)
	}
	containerEnvRoot, err := detectContainerEnvironmentRoot(task)
	if err != nil {
		return nil, err
	}

	args := []string{
		"run",
		"-d",
		"-v", taskRoot + ":" + containerTaskRoot,
		"-v", environmentRoot + ":" + containerEnvRoot,
		"-w", containerEnvRoot,
		image,
		defaultShellPath,
		"-lc",
		"mkdir -p /logs/verifier && ln -sfn " + containerTestsDir + " /tests && trap 'exit 0' TERM INT; while true; do sleep 1; done",
	}

	cmd := exec.CommandContext(ctx, "docker", args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("start docker task runtime: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	containerID := strings.TrimSpace(stdout.String())
	if containerID == "" {
		return nil, fmt.Errorf("docker run returned empty container id")
	}

	return &dockerTaskRuntime{
		containerID:      containerID,
		taskRoot:         taskRoot,
		environmentRoot:  environmentRoot,
		containerEnvRoot: containerEnvRoot,
	}, nil
}

func (r *dockerTaskRuntime) Close(ctx context.Context) error {
	if r == nil || r.containerID == "" {
		return nil
	}
	cmd := exec.CommandContext(ctx, "docker", "rm", "-f", r.containerID)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		message := strings.TrimSpace(stderr.String())
		if strings.Contains(strings.ToLower(message), "no such container") {
			return nil
		}
		return fmt.Errorf("remove docker container %s: %w: %s", r.containerID, err, strings.TrimSpace(stderr.String()))
	}
	return nil
}

func (r *dockerTaskRuntime) Exec(ctx context.Context, workingDir string, shellPath string, shellCommand string, extraEnv []string) ([]byte, error) {
	if r == nil || r.containerID == "" {
		return nil, fmt.Errorf("docker task runtime is not initialized")
	}
	if strings.TrimSpace(shellPath) == "" {
		shellPath = defaultShellPath
	}
	containerWorkdir, err := r.containerPathForHost(workingDir)
	if err != nil {
		return nil, err
	}

	args := []string{"exec", "-w", containerWorkdir}
	for _, env := range extraEnv {
		if strings.TrimSpace(env) == "" {
			continue
		}
		args = append(args, "-e", env)
	}
	args = append(args, r.containerID, shellPath, "-lc", shellCommand)

	cmd := exec.CommandContext(ctx, "docker", args...)
	return cmd.CombinedOutput()
}

func (r *dockerTaskRuntime) ReadFile(ctx context.Context, path string) (string, error) {
	if r == nil || r.containerID == "" {
		return "", fmt.Errorf("docker task runtime is not initialized")
	}
	cmd := exec.CommandContext(ctx, "docker", "exec", r.containerID, "cat", "--", path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func (r *dockerTaskRuntime) containerPathForHost(hostPath string) (string, error) {
	hostPath = strings.TrimSpace(hostPath)
	if hostPath == "" {
		return r.containerEnvironmentRoot(), nil
	}

	absPath, err := filepath.Abs(hostPath)
	if err != nil {
		return "", fmt.Errorf("resolve host path: %w", err)
	}

	environmentRoot := filepath.Clean(r.environmentRoot)
	if environmentRoot != "" {
		if absPath == environmentRoot {
			return r.containerEnvironmentRoot(), nil
		}
		if strings.HasPrefix(absPath, environmentRoot+string(filepath.Separator)) {
			rel, err := filepath.Rel(environmentRoot, absPath)
			if err != nil {
				return "", fmt.Errorf("compute environment relative path: %w", err)
			}
			return path.Join(r.containerEnvironmentRoot(), filepath.ToSlash(rel)), nil
		}
	}

	taskRoot := filepath.Clean(r.taskRoot)
	if absPath == taskRoot {
		return containerTaskRoot, nil
	}
	if !strings.HasPrefix(absPath, taskRoot+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside task root", hostPath)
	}

	rel, err := filepath.Rel(taskRoot, absPath)
	if err != nil {
		return "", fmt.Errorf("compute relative path: %w", err)
	}
	return filepath.ToSlash(filepath.Join(containerTaskRoot, rel)), nil
}

func (r *dockerTaskRuntime) containerEnvironmentRoot() string {
	if r == nil || strings.TrimSpace(r.containerEnvRoot) == "" {
		return containerEnvDir
	}
	return r.containerEnvRoot
}

func detectContainerEnvironmentRoot(task *MaterializedTask) (string, error) {
	if task == nil || strings.TrimSpace(task.EnvironmentDir) == "" {
		return containerEnvDir, nil
	}

	dockerfilePath := filepath.Join(task.EnvironmentDir, "Dockerfile")
	data, err := os.ReadFile(dockerfilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return containerEnvDir, nil
		}
		return "", fmt.Errorf("read task Dockerfile: %w", err)
	}

	workdir := parseDockerfileWorkdir(string(data))
	if strings.TrimSpace(workdir) == "" {
		return containerEnvDir, nil
	}
	return workdir, nil
}

func parseDockerfileWorkdir(contents string) string {
	current := ""
	for _, rawLine := range strings.Split(contents, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		upper := strings.ToUpper(line)
		if !strings.HasPrefix(upper, "WORKDIR ") {
			continue
		}

		value := strings.TrimSpace(line[len("WORKDIR "):])
		value = strings.Trim(value, `"'`)
		if value == "" {
			continue
		}
		if strings.HasPrefix(value, "/") {
			current = path.Clean(value)
			continue
		}
		if current == "" {
			continue
		}
		current = path.Clean(path.Join(current, value))
	}
	return current
}
