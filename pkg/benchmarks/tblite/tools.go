package tblite

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
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
	OutputLimit      int
	CommandTimeout   time.Duration
	CommandRunner    CommandRunner
	ShellPath        string
	TaskRoot         string
	TestsDir         string
	ContainerEnvRoot string
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
	resolver, err := newToolPathResolver(absRoot, cfg)
	if err != nil {
		return nil, err
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
		newListFilesTool(resolver, cfg.OutputLimit),
		newReadFileTool(resolver, cfg.OutputLimit),
		newWriteFileTool(resolver, cfg.OutputLimit),
		newRunCommandTool(resolver, cfg.OutputLimit, cfg.CommandTimeout, cfg.CommandRunner),
	}

	return tools, nil
}

func newListFilesTool(resolver toolPathResolver, outputLimit int) core.Tool {
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

			targetPath, err := resolver.resolveSecurePath(stringArg(args, "path", "."))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			recursive := boolArg(args, "recursive")
			entries, err := listEntries(targetPath, recursive)
			if err != nil {
				return textToolResult(resolver.sanitizeError(err.Error()), true), nil
			}

			return textToolResult(agentutil.TruncateString(strings.Join(entries, "\n"), outputLimit), false), nil
		})
}

func newReadFileTool(resolver toolPathResolver, outputLimit int) core.Tool {
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

			targetPath, err := resolver.resolveSecurePath(requiredStringArg(args, "path"))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			data, err := os.ReadFile(targetPath)
			if err != nil {
				return textToolResult(fmt.Sprintf("read file: %s", resolver.sanitizeError(err.Error())), true), nil
			}

			return textToolResult(agentutil.TruncateString(string(data), outputLimit), false), nil
		})
}

func newWriteFileTool(resolver toolPathResolver, outputLimit int) core.Tool {
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

			targetPath, err := resolver.resolveSecurePath(requiredStringArg(args, "path"))
			if err != nil {
				return textToolResult(err.Error(), true), nil
			}

			content := requiredStringArg(args, "content")
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return textToolResult(fmt.Sprintf("create parent directories: %s", resolver.sanitizeError(err.Error())), true), nil
			}
			if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
				return textToolResult(fmt.Sprintf("write file: %s", resolver.sanitizeError(err.Error())), true), nil
			}

			message := fmt.Sprintf("wrote %d bytes to %s", len(content), resolver.displayPath(targetPath))
			return textToolResult(agentutil.TruncateString(message, outputLimit), false), nil
		})
}

func newRunCommandTool(resolver toolPathResolver, outputLimit int, commandTimeout time.Duration, runner CommandRunner) core.Tool {
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

			workingDir, err := resolver.resolveSecurePath(stringArg(args, "working_directory", "."))
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
				return textToolResult(agentutil.TruncateString(outputText, outputLimit), false), nil
			}

			exitMessage := fmt.Sprintf("command failed: %v\n%s", err, outputText)
			return textToolResult(agentutil.TruncateString(exitMessage, outputLimit), true), nil
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
	cmd.Env = append(cmd.Env, os.Environ()...)
	cmd.Env = append(cmd.Env, extraEnv...)
	return cmd.CombinedOutput()
}

func resolveToolPath(rootDir, relativePath string) (string, error) {
	return newResolverForRoot(rootDir).resolveSecurePath(relativePath)
}

type toolPathResolver struct {
	rootDir          string
	taskRoot         string
	testsDir         string
	containerEnvRoot string
}

func newResolverForRoot(rootDir string) toolPathResolver {
	return toolPathResolver{rootDir: filepath.Clean(rootDir)}
}

func newToolPathResolver(rootDir string, cfg ToolsetConfig) (toolPathResolver, error) {
	resolver := toolPathResolver{rootDir: filepath.Clean(rootDir)}
	if strings.TrimSpace(cfg.TaskRoot) != "" {
		taskRoot, err := filepath.Abs(cfg.TaskRoot)
		if err != nil {
			return toolPathResolver{}, fmt.Errorf("resolve task root: %w", err)
		}
		resolver.taskRoot = filepath.Clean(taskRoot)
	}
	if strings.TrimSpace(cfg.TestsDir) != "" {
		testsDir, err := filepath.Abs(cfg.TestsDir)
		if err != nil {
			return toolPathResolver{}, fmt.Errorf("resolve tests dir: %w", err)
		}
		resolver.testsDir = filepath.Clean(testsDir)
	}
	if strings.TrimSpace(cfg.ContainerEnvRoot) != "" {
		resolver.containerEnvRoot = cleanContainerPath(cfg.ContainerEnvRoot)
	}
	return resolver, nil
}

func (r toolPathResolver) resolve(input string) (string, error) {
	raw := strings.TrimSpace(input)
	if raw == "" || raw == "." {
		return r.rootDir, nil
	}

	if mapped, ok, err := r.resolveAlias(raw); err != nil {
		return "", err
	} else if ok {
		return mapped, nil
	}

	cleaned := filepath.Clean(raw)
	target := filepath.Join(r.rootDir, cleaned)
	return r.ensureWithinAllowed(target, input)
}

func (r toolPathResolver) resolveAlias(input string) (string, bool, error) {
	if r.taskRoot == "" {
		return "", false, nil
	}

	if filepath.IsAbs(input) {
		return r.resolveAbsoluteAlias(input)
	}

	cleaned := filepath.Clean(input)
	switch cleaned {
	case "test.sh", "./test.sh":
		mapped, err := r.ensureWithinAllowed(filepath.Join(r.taskRoot, "test.sh"), input)
		return mapped, err == nil, err
	case "instruction.txt", "./instruction.txt":
		mapped, err := r.ensureWithinAllowed(filepath.Join(r.taskRoot, "instruction.txt"), input)
		return mapped, err == nil, err
	case "tests":
		if r.testsDir != "" {
			mapped, err := r.ensureWithinAllowed(r.testsDir, input)
			return mapped, err == nil, err
		}
	case "environment":
		mapped, err := r.ensureWithinAllowed(r.rootDir, input)
		return mapped, err == nil, err
	}

	if mapped, ok := r.resolveRelativeAlias(cleaned); ok {
		allowed, err := r.ensureWithinAllowed(mapped, input)
		return allowed, err == nil, err
	}
	return "", false, nil
}

func (r toolPathResolver) resolveAbsoluteAlias(input string) (string, bool, error) {
	cleaned := filepath.Clean(input)
	if mapped, ok := r.matchHostPath(cleaned); ok {
		return mapped, true, nil
	}

	if strings.HasPrefix(cleaned, containerTaskRoot) {
		rel := strings.TrimPrefix(cleaned, containerTaskRoot)
		rel = strings.TrimPrefix(rel, string(filepath.Separator))
		mapped, err := r.ensureWithinAllowed(filepath.Join(r.taskRoot, rel), input)
		return mapped, err == nil, err
	}
	if r.testsDir != "" && (cleaned == "/tests" || strings.HasPrefix(cleaned, "/tests/")) {
		rel := strings.TrimPrefix(cleaned, "/tests")
		rel = strings.TrimPrefix(rel, string(filepath.Separator))
		mapped, err := r.ensureWithinAllowed(filepath.Join(r.testsDir, rel), input)
		return mapped, err == nil, err
	}
	if r.containerEnvRoot != "" && (cleaned == r.containerEnvRoot || strings.HasPrefix(cleaned, r.containerEnvRoot+"/")) {
		rel := strings.TrimPrefix(cleaned, r.containerEnvRoot)
		rel = strings.TrimPrefix(rel, string(filepath.Separator))
		mapped, err := r.ensureWithinAllowed(filepath.Join(r.rootDir, rel), input)
		return mapped, err == nil, err
	}
	return "", false, nil
}

func (r toolPathResolver) resolveRelativeAlias(cleaned string) (string, bool) {
	if cleaned == "" || cleaned == "." {
		return r.rootDir, true
	}
	if strings.HasPrefix(cleaned, "tests"+string(filepath.Separator)) && r.testsDir != "" {
		return filepath.Join(r.testsDir, strings.TrimPrefix(cleaned, "tests"+string(filepath.Separator))), true
	}
	if strings.HasPrefix(cleaned, "environment"+string(filepath.Separator)) {
		return filepath.Join(r.rootDir, strings.TrimPrefix(cleaned, "environment"+string(filepath.Separator))), true
	}

	containerAlias := strings.TrimPrefix(r.containerEnvRoot, "/")
	if containerAlias != "" {
		if cleaned == containerAlias {
			return r.rootDir, true
		}
		if strings.HasPrefix(cleaned, containerAlias+string(filepath.Separator)) {
			return filepath.Join(r.rootDir, strings.TrimPrefix(cleaned, containerAlias+string(filepath.Separator))), true
		}
		aliasBase := filepath.Base(containerAlias)
		if aliasBase != containerAlias {
			if cleaned == aliasBase {
				return r.rootDir, true
			}
			if strings.HasPrefix(cleaned, aliasBase+string(filepath.Separator)) {
				return filepath.Join(r.rootDir, strings.TrimPrefix(cleaned, aliasBase+string(filepath.Separator))), true
			}
		}
	}
	return "", false
}

func (r toolPathResolver) matchHostPath(input string) (string, bool) {
	if r.taskRoot == "" {
		return "", false
	}
	input = filepath.Clean(input)
	switch {
	case input == r.rootDir || strings.HasPrefix(input, r.rootDir+string(filepath.Separator)):
		return input, true
	case r.testsDir != "" && (input == r.testsDir || strings.HasPrefix(input, r.testsDir+string(filepath.Separator))):
		return input, true
	case input == r.taskRoot || strings.HasPrefix(input, r.taskRoot+string(filepath.Separator)):
		return input, true
	default:
		return "", false
	}
}

func (r toolPathResolver) ensureWithinAllowed(target, original string) (string, error) {
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", original, err)
	}
	allowedRoots := []string{r.rootDir}
	if r.taskRoot != "" {
		allowedRoots = append(allowedRoots, r.taskRoot)
	}
	if r.testsDir != "" {
		allowedRoots = append(allowedRoots, r.testsDir)
	}
	for _, root := range allowedRoots {
		root = filepath.Clean(root)
		canonicalRoots := []string{root}
		if resolved, err := filepath.EvalSymlinks(root); err == nil {
			resolved = filepath.Clean(resolved)
			if resolved != root {
				canonicalRoots = append(canonicalRoots, resolved)
			}
		}
		for _, candidate := range canonicalRoots {
			if absTarget == candidate || strings.HasPrefix(absTarget, candidate+string(os.PathSeparator)) {
				return absTarget, nil
			}
		}
	}
	return "", fmt.Errorf("path %q escapes benchmark workspace", original)
}

func (r toolPathResolver) resolveSecurePath(input string) (string, error) {
	target, err := r.resolve(input)
	if err != nil {
		return "", err
	}
	resolvedTarget, err := resolvePathThroughExistingSymlinks(target)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", input, err)
	}
	return r.ensureWithinAllowed(resolvedTarget, input)
}

func resolvePathThroughExistingSymlinks(target string) (string, error) {
	current := filepath.Clean(target)
	missing := make([]string, 0, 4)

	for {
		_, err := os.Lstat(current)
		if err == nil {
			resolved, err := filepath.EvalSymlinks(current)
			if err != nil {
				return "", err
			}
			current = resolved
			break
		}
		if !os.IsNotExist(err) {
			return "", err
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		missing = append(missing, filepath.Base(current))
		current = parent
	}

	for i := len(missing) - 1; i >= 0; i-- {
		current = filepath.Join(current, missing[i])
	}
	return filepath.Clean(current), nil
}

func (r toolPathResolver) sanitizeError(message string) string {
	sanitized := message
	replacements := [][2]string{
		{filepath.ToSlash(filepath.Clean(r.rootDir)), r.displayPath(r.rootDir)},
	}
	if r.testsDir != "" {
		replacements = append(replacements, [2]string{filepath.ToSlash(filepath.Clean(r.testsDir)), "/tests"})
	}
	if r.taskRoot != "" {
		replacements = append(replacements, [2]string{filepath.ToSlash(filepath.Clean(r.taskRoot)), containerTaskRoot})
	}
	for _, replacement := range replacements {
		if replacement[0] == "" || replacement[1] == "" {
			continue
		}
		sanitized = strings.ReplaceAll(filepath.ToSlash(sanitized), replacement[0], replacement[1])
	}
	return sanitized
}

func (r toolPathResolver) displayPath(target string) string {
	target = filepath.Clean(target)
	if target == r.rootDir {
		return r.environmentDisplayRoot()
	}
	if r.testsDir != "" {
		if target == r.testsDir {
			return "/tests"
		}
		if strings.HasPrefix(target, r.testsDir+string(filepath.Separator)) {
			rel, _ := filepath.Rel(r.testsDir, target)
			return path.Join("/tests", filepath.ToSlash(rel))
		}
	}
	if r.taskRoot != "" {
		if target == filepath.Join(r.taskRoot, "test.sh") {
			return "/task/test.sh"
		}
		if target == filepath.Join(r.taskRoot, "instruction.txt") {
			return "/task/instruction.txt"
		}
		if strings.HasPrefix(target, r.rootDir+string(filepath.Separator)) {
			rel, _ := filepath.Rel(r.rootDir, target)
			return path.Join(r.environmentDisplayRoot(), filepath.ToSlash(rel))
		}
		if target == r.taskRoot {
			return containerTaskRoot
		}
		if strings.HasPrefix(target, r.taskRoot+string(filepath.Separator)) {
			rel, _ := filepath.Rel(r.taskRoot, target)
			return path.Join(containerTaskRoot, filepath.ToSlash(rel))
		}
	}
	return filepath.ToSlash(target)
}

func (r toolPathResolver) environmentDisplayRoot() string {
	if r.containerEnvRoot != "" {
		return r.containerEnvRoot
	}
	if r.taskRoot != "" {
		return containerEnvDir
	}
	return filepath.ToSlash(r.rootDir)
}

func cleanContainerPath(value string) string {
	cleaned := filepath.ToSlash(strings.TrimSpace(value))
	if cleaned == "" {
		return ""
	}
	if !strings.HasPrefix(cleaned, "/") {
		cleaned = "/" + cleaned
	}
	return path.Clean(cleaned)
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
