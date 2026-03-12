package tblite

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	toolspkg "github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

const (
	defaultToolOutputLimit = 16 * 1024
	defaultCommandTimeout  = 30 * time.Second
	maxListEntries         = 256
)

// ToolsetConfig controls the benchmark-local terminal/file tools exposed to the agent.
type ToolsetConfig struct {
	OutputLimit    int
	CommandTimeout time.Duration
	CommandRunner  CommandRunner
	ShellPath      string
}

// CommandRunner abstracts shell command execution for benchmark tools.
type CommandRunner interface {
	Run(ctx context.Context, workingDir string, command string, extraEnv []string) ([]byte, error)
}

// NewTerminalToolset creates a minimal tool bundle for terminal benchmark tasks.
func NewTerminalToolset(rootDir string, cfg ToolsetConfig) ([]core.Tool, error) {
	absRoot, err := filepath.Abs(rootDir)
	if err != nil {
		return nil, fmt.Errorf("resolve root dir: %w", err)
	}

	if cfg.OutputLimit <= 0 {
		cfg.OutputLimit = defaultToolOutputLimit
	}
	if cfg.CommandTimeout <= 0 {
		cfg.CommandTimeout = defaultCommandTimeout
	}
	if cfg.ShellPath == "" {
		cfg.ShellPath = defaultShellPath
	}
	if cfg.CommandRunner == nil {
		cfg.CommandRunner = hostCommandRunner{shellPath: cfg.ShellPath}
	}

	tools := []core.Tool{
		newListFilesTool(absRoot, cfg.OutputLimit),
		newReadFileTool(absRoot, cfg.OutputLimit),
		newWriteFileTool(absRoot, cfg.OutputLimit),
		newRunCommandTool(absRoot, cfg.OutputLimit, cfg.CommandTimeout, cfg.CommandRunner),
	}

	return tools, nil
}

func newListFilesTool(rootDir string, outputLimit int) core.Tool {
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"path": {
				Type:        "string",
				Description: "Relative directory path to list. Defaults to the task root.",
			},
			"recursive": {
				Type:        "boolean",
				Description: "Whether to recursively list descendant files.",
			},
		},
	}

	return toolspkg.NewFuncTool("list_files", "List files and directories within the benchmark workspace.", schema,
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			_ = ctx

			targetPath, err := resolveToolPath(rootDir, stringArg(args, "path", "."))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			recursive := boolArg(args, "recursive")
			entries, err := listEntries(targetPath, recursive)
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			return textToolResult(truncateString(strings.Join(entries, "\n"), outputLimit), false), nil
		})
}

func newReadFileTool(rootDir string, outputLimit int) core.Tool {
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"path": {
				Type:        "string",
				Description: "Relative file path to read.",
				Required:    true,
			},
		},
	}

	return toolspkg.NewFuncTool("read_file", "Read a UTF-8 text file from the benchmark workspace.", schema,
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			_ = ctx

			targetPath, err := resolveToolPath(rootDir, requiredStringArg(args, "path"))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			data, err := os.ReadFile(targetPath)
			if err != nil {
				return textToolResult(fmt.Sprintf("read file: %v", err), true), nil
			}

			return textToolResult(truncateString(string(data), outputLimit), false), nil
		})
}

func newWriteFileTool(rootDir string, outputLimit int) core.Tool {
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"path": {
				Type:        "string",
				Description: "Relative file path to write.",
				Required:    true,
			},
			"content": {
				Type:        "string",
				Description: "New file contents.",
				Required:    true,
			},
		},
	}

	return toolspkg.NewFuncTool("write_file", "Write a UTF-8 text file inside the benchmark workspace.", schema,
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			_ = ctx

			targetPath, err := resolveToolPath(rootDir, requiredStringArg(args, "path"))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			content := requiredStringArg(args, "content")
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return textToolResult(fmt.Sprintf("create parent directories: %v", err), true), nil
			}
			if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
				return textToolResult(fmt.Sprintf("write file: %v", err), true), nil
			}

			message := fmt.Sprintf("wrote %d bytes to %s", len(content), filepath.Base(targetPath))
			return textToolResult(truncateString(message, outputLimit), false), nil
		})
}

func newRunCommandTool(rootDir string, outputLimit int, commandTimeout time.Duration, runner CommandRunner) core.Tool {
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"command": {
				Type:        "string",
				Description: "Shell command to execute inside the benchmark workspace.",
				Required:    true,
			},
			"working_directory": {
				Type:        "string",
				Description: "Optional relative working directory for the command.",
			},
			"timeout_sec": {
				Type:        "number",
				Description: "Optional timeout override in seconds.",
			},
		},
	}

	return toolspkg.NewFuncTool("run_command", "Execute a shell command within the benchmark workspace.", schema,
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			command := requiredStringArg(args, "command")
			if strings.TrimSpace(command) == "" {
				return textToolResult("run_command requires a non-empty command", true), nil
			}

			workingDir, err := resolveToolPath(rootDir, stringArg(args, "working_directory", "."))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			timeout := durationFromSeconds(numberArg(args, "timeout_sec", 0), commandTimeout)
			runCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			output, err := runner.Run(runCtx, workingDir, command, nil)
			outputText := strings.TrimSpace(string(output))
			if outputText == "" {
				outputText = "(no output)"
			}

			if err == nil {
				return textToolResult(truncateString(outputText, outputLimit), false), nil
			}

			exitMessage := fmt.Sprintf("command failed: %v\n%s", err, outputText)
			return textToolResult(truncateString(exitMessage, outputLimit), true), nil
		})
}

type hostCommandRunner struct {
	shellPath string
}

func (r hostCommandRunner) Run(ctx context.Context, workingDir string, command string, extraEnv []string) ([]byte, error) {
	shellPath := r.shellPath
	if strings.TrimSpace(shellPath) == "" {
		shellPath = defaultShellPath
	}
	cmd := exec.CommandContext(ctx, shellPath, "-lc", command)
	cmd.Dir = workingDir
	cmd.Env = append(cmd.Env, extraEnv...)
	cmd.Env = append(cmd.Env, os.Environ()...)
	return cmd.CombinedOutput()
}

func resolveToolPath(rootDir, relativePath string) (string, error) {
	cleaned := filepath.Clean(strings.TrimSpace(relativePath))
	if cleaned == "." || cleaned == "" {
		return rootDir, nil
	}

	target := filepath.Join(rootDir, cleaned)
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", relativePath, err)
	}

	if absTarget != rootDir && !strings.HasPrefix(absTarget, rootDir+string(os.PathSeparator)) {
		return "", fmt.Errorf("path %q escapes benchmark workspace", relativePath)
	}

	return absTarget, nil
}

func listEntries(root string, recursive bool) ([]string, error) {
	entries := make([]string, 0, 32)

	if recursive {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if path == root {
				return nil
			}
			if len(entries) >= maxListEntries {
				return fs.SkipAll
			}
			entries = append(entries, path)
			return nil
		})
		if err != nil && err != fs.SkipAll {
			return nil, err
		}
	} else {
		dirEntries, err := os.ReadDir(root)
		if err != nil {
			return nil, err
		}
		for _, entry := range dirEntries {
			entries = append(entries, filepath.Join(root, entry.Name()))
		}
	}

	sort.Strings(entries)
	for i := range entries {
		entries[i] = normalizePath(entries[i], root)
	}

	if len(entries) == 0 {
		return []string{"(empty)"}, nil
	}

	return entries, nil
}

func normalizePath(path, root string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return path
	}
	return filepath.ToSlash(rel)
}

func textToolResult(text string, isError bool) *models.CallToolResult {
	return &models.CallToolResult{
		IsError: isError,
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: text,
			},
		},
	}
}

func truncateString(value string, limit int) string {
	if limit <= 0 || len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
}

func stringArg(args map[string]any, key, fallback string) string {
	raw, ok := args[key]
	if !ok || raw == nil {
		return fallback
	}
	value, ok := raw.(string)
	if !ok {
		return fallback
	}
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func requiredStringArg(args map[string]any, key string) string {
	value, _ := args[key].(string)
	return value
}

func boolArg(args map[string]any, key string) bool {
	raw, ok := args[key]
	if !ok {
		return false
	}
	value, ok := raw.(bool)
	return ok && value
}

func numberArg(args map[string]any, key string, fallback float64) float64 {
	raw, ok := args[key]
	if !ok || raw == nil {
		return fallback
	}
	switch value := raw.(type) {
	case float64:
		return value
	case int:
		return float64(value)
	case int64:
		return float64(value)
	default:
		return fallback
	}
}

func durationFromSeconds(seconds float64, fallback time.Duration) time.Duration {
	if seconds <= 0 {
		return fallback
	}
	return time.Duration(seconds * float64(time.Second))
}
