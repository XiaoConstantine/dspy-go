package bash

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools/internal/localfs"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

const (
	DefaultShellPath          = "/bin/bash"
	DefaultTimeout            = 30 * time.Second
	DefaultModelOutputLimit   = 1600
	DefaultDisplayOutputLimit = 6000
)

var allowedEnvironmentKeys = map[string]struct{}{
	"HOME":    {},
	"LANG":    {},
	"LC_ALL":  {},
	"LOGNAME": {},
	"PATH":    {},
	"SHELL":   {},
	"TEMP":    {},
	"TERM":    {},
	"TMP":     {},
	"TMPDIR":  {},
	"USER":    {},
}

type Config struct {
	Root               string
	ShellPath          string
	Timeout            time.Duration
	ModelOutputLimit   int
	DisplayOutputLimit int
	ExtraEnv           map[string]string
	PassthroughEnvKeys []string
}

type Tool struct {
	resolver           *localfs.Resolver
	root               string
	shellPath          string
	timeout            time.Duration
	modelOutputLimit   int
	displayOutputLimit int
	extraEnv           map[string]string
	passthroughEnvKeys []string
}

func NewTool(cfg Config) (core.Tool, error) {
	resolver, err := localfs.NewResolver(cfg.Root)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(cfg.ShellPath) == "" {
		cfg.ShellPath = DefaultShellPath
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if cfg.ModelOutputLimit <= 0 {
		cfg.ModelOutputLimit = DefaultModelOutputLimit
	}
	if cfg.DisplayOutputLimit <= 0 {
		cfg.DisplayOutputLimit = DefaultDisplayOutputLimit
	}
	return &Tool{
		resolver:           resolver,
		root:               resolver.Root(),
		shellPath:          cfg.ShellPath,
		timeout:            cfg.Timeout,
		modelOutputLimit:   cfg.ModelOutputLimit,
		displayOutputLimit: cfg.DisplayOutputLimit,
		extraEnv:           copyStringMap(cfg.ExtraEnv),
		passthroughEnvKeys: append([]string(nil), cfg.PassthroughEnvKeys...),
	}, nil
}

func (t *Tool) Name() string {
	return "bash"
}

func (t *Tool) Description() string {
	return "Execute a bash command inside the workspace."
}

func (t *Tool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         t.Name(),
		Description:  t.Description(),
		InputSchema:  t.InputSchema(),
		Capabilities: []string{"workspace", "shell", "bash"},
		Version:      "1.0.0",
	}
}

func (t *Tool) CanHandle(_ context.Context, intent string) bool {
	intent = strings.ToLower(intent)
	return strings.Contains(intent, "bash") ||
		strings.Contains(intent, "shell") ||
		strings.Contains(intent, "command") ||
		strings.Contains(intent, "terminal")
}

func (t *Tool) Execute(ctx context.Context, params map[string]any) (core.ToolResult, error) {
	command := stringValue(params["command"])
	if strings.TrimSpace(command) == "" {
		return t.errorResult("command is required", map[string]any{"error": "command is required"}), nil
	}

	workingDir, err := t.resolver.ResolveSecurePath(stringValue(params["working_directory"]))
	if err != nil {
		return t.errorResult(err.Error(), map[string]any{"error": err.Error()}), nil
	}

	timeout := durationFromSeconds(numberValue(params["timeout_sec"]), t.timeout)
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, t.shellPath, "-c", command)
	cmd.Dir = workingDir
	cmd.Env = buildEnvironment(t.extraEnv, t.passthroughEnvKeys)
	output, execErr := cmd.CombinedOutput()
	outputText := strings.TrimSpace(string(output))
	if outputText == "" {
		outputText = "(no output)"
	}

	details := map[string]any{
		"command":           command,
		"working_directory": t.resolver.DisplayPath(workingDir),
		"timeout_sec":       timeout.Seconds(),
		"output":            truncateRunes(outputText, t.displayOutputLimit),
	}

	if execErr == nil {
		return successToolResult(
			truncateRunes(outputText, t.modelOutputLimit),
			truncateRunes(outputText, t.displayOutputLimit),
			details,
		), nil
	}

	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		message := fmt.Sprintf("command timed out after %.0f seconds", timeout.Seconds())
		details["timed_out"] = true
		return t.errorResult(message+"\n"+truncateRunes(outputText, t.displayOutputLimit), details), nil
	}

	details["exit_code"] = exitCode(execErr)
	message := fmt.Sprintf("command failed: %v", execErr)
	if outputText != "(no output)" {
		message += "\n" + outputText
	}
	return t.errorResult(truncateRunes(message, t.displayOutputLimit), details), nil
}

func (t *Tool) Validate(params map[string]any) error {
	if strings.TrimSpace(stringValue(params["command"])) == "" {
		return fmt.Errorf("command is required")
	}
	return nil
}

func (t *Tool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"command": {
				Type:        "string",
				Description: "Bash command to execute inside the workspace.",
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
}

func (t *Tool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	cloned.extraEnv = copyStringMap(t.extraEnv)
	cloned.passthroughEnvKeys = append([]string(nil), t.passthroughEnvKeys...)
	return &cloned
}

func (t *Tool) Root() string {
	if t == nil {
		return ""
	}
	return t.root
}

func successToolResult(modelText, displayText string, details map[string]any) core.ToolResult {
	if strings.TrimSpace(displayText) == "" {
		displayText = modelText
	}
	return core.ToolResult{
		Data: displayText,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   modelText,
			core.ToolResultDisplayTextMeta: displayText,
			core.ToolResultIsErrorMeta:     false,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: details,
		},
	}
}

func (t *Tool) errorResult(message string, details map[string]any) core.ToolResult {
	modelText := truncateRunes(message, t.modelOutputLimit)
	displayText := truncateRunes(message, t.displayOutputLimit)
	return core.ToolResult{
		Data: displayText,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   modelText,
			core.ToolResultDisplayTextMeta: displayText,
			core.ToolResultIsErrorMeta:     true,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: details,
		},
	}
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		return ""
	}
}

func numberValue(value any) float64 {
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int32:
		return float64(typed)
	case int64:
		return float64(typed)
	case float32:
		return float64(typed)
	case float64:
		return typed
	default:
		return 0
	}
}

func durationFromSeconds(seconds float64, fallback time.Duration) time.Duration {
	if seconds <= 0 {
		return fallback
	}
	return time.Duration(seconds * float64(time.Second))
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(strings.TrimSpace(text))
	if len(runes) <= limit {
		return string(runes)
	}
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

func exitCode(err error) int {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
			return status.ExitStatus()
		}
	}
	return -1
}

func buildEnvironment(extraEnv map[string]string, passthroughKeys []string) []string {
	allowed := make(map[string]struct{}, len(allowedEnvironmentKeys)+len(passthroughKeys))
	for key := range allowedEnvironmentKeys {
		allowed[key] = struct{}{}
	}
	for _, key := range passthroughKeys {
		key = strings.TrimSpace(key)
		if key != "" {
			allowed[key] = struct{}{}
		}
	}

	env := make([]string, 0, len(allowed)+len(extraEnv))
	for _, entry := range os.Environ() {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		if _, ok := allowed[key]; ok {
			env = append(env, entry)
			continue
		}
		if strings.HasPrefix(key, "LC_") {
			env = append(env, entry)
		}
	}
	for key, value := range extraEnv {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		env = append(env, key+"="+value)
	}
	return env
}

func copyStringMap(source map[string]string) map[string]string {
	if len(source) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}
