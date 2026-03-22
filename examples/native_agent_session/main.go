package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/native"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

const (
	defaultModelID      = string(core.ModelGoogleGeminiFlash)
	defaultSessionID    = "native-agent-demo"
	defaultDBPath       = "./tmp/native-agent-session/session.db"
	defaultWorkspaceDir = "./tmp/native-agent-session/workspace"
	defaultTaskTimeout  = 2 * time.Minute
	modelOutputLimit    = 1600
	displayOutputLimit  = 6000
)

const exampleSystemPrompt = `You are a pragmatic native tool-calling assistant working inside a local workspace.

Use the available tools to inspect files, edit files, and create new files.
Prefer session recall when it already answers the question from previous runs.
Use list_files before making assumptions about the workspace layout.
Use edit_file for targeted changes when a file already exists.
Use write_file when creating a new file or replacing the full contents intentionally.
Call the finish tool once the task is complete.`

var errUsage = errors.New("usage")

func main() {
	if err := run(); err != nil {
		if errors.Is(err, errUsage) {
			printUsage(os.Stderr)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		apiKey       string
		model        string
		task         string
		sessionID    string
		sessionDB    string
		workspaceDir string
		branchID     string
		forkName     string
		activateFork bool
		showState    bool
		verbose      bool
		maxTurns     int
		temperature  float64
		timeout      time.Duration
	)

	flag.StringVar(&apiKey, "api-key", "", "API key for the selected provider (optional when provider env vars are set)")
	flag.StringVar(&model, "model", defaultModelID, "Tool-calling model ID")
	flag.StringVar(&task, "task", "", "Task for the native agent. If empty, remaining args are joined as the task")
	flag.StringVar(&sessionID, "session", defaultSessionID, "Logical session ID used for recall and branching")
	flag.StringVar(&sessionDB, "session-db", defaultDBPath, "SQLite path for the session event store")
	flag.StringVar(&workspaceDir, "workspace", defaultWorkspaceDir, "Workspace root exposed to the example file tools")
	flag.StringVar(&branchID, "branch", "", "Branch ID to resume explicitly before this run")
	flag.StringVar(&forkName, "fork-name", "", "Fork the active branch after the run using this branch name")
	flag.BoolVar(&activateFork, "activate-fork", true, "When forking after the run, make the new branch active")
	flag.BoolVar(&showState, "show-state", true, "Print session and branch state after the run")
	flag.BoolVar(&verbose, "verbose", false, "Print selected native-agent lifecycle events")
	flag.IntVar(&maxTurns, "max-turns", 12, "Maximum tool-calling turns for the run")
	flag.Float64Var(&temperature, "temperature", 0.2, "Sampling temperature")
	flag.DurationVar(&timeout, "timeout", defaultTaskTimeout, "End-to-end timeout for one run")
	flag.Parse()

	task = strings.TrimSpace(task)
	if task == "" {
		task = strings.TrimSpace(strings.Join(flag.Args(), " "))
	}
	if task == "" {
		return errUsage
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	llms.EnsureFactory()
	llm, err := llms.NewLLM(strings.TrimSpace(apiKey), core.ModelID(strings.TrimSpace(model)))
	if err != nil {
		return fmt.Errorf("create llm for model %q: %v\nHint: pass -api-key or configure credentials for the selected provider in the environment", model, err)
	}

	store, err := sessionevent.NewSQLiteStore(sessionDB)
	if err != nil {
		return fmt.Errorf("open session store %q: %w", sessionDB, err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "warning: close session store: %v\n", err)
		}
	}()

	resolver, err := newWorkspaceResolver(workspaceDir)
	if err != nil {
		return fmt.Errorf("resolve workspace %q: %w", workspaceDir, err)
	}
	if err := ensureWorkspaceSeed(resolver.root); err != nil {
		return fmt.Errorf("prepare workspace seed: %w", err)
	}

	agent, err := native.NewAgent(llm, native.Config{
		MaxTurns:              maxTurns,
		Temperature:           temperature,
		SystemPrompt:          exampleSystemPrompt,
		SessionID:             sessionID,
		SessionBranchID:       strings.TrimSpace(branchID),
		SessionRecallLimit:    4,
		SessionRecallMaxChars: 1800,
		SessionEventStore:     store,
		OnEvent:               makeEventPrinter(verbose),
	})
	if err != nil {
		return fmt.Errorf("create native agent: %w", err)
	}

	for _, tool := range newWorkspaceTools(resolver) {
		if err := agent.RegisterTool(tool); err != nil {
			return fmt.Errorf("register tool %q: %w", tool.Name(), err)
		}
	}

	fmt.Printf("Running native agent\n")
	fmt.Printf("  model: %s\n", llm.ModelID())
	fmt.Printf("  provider: %s\n", llm.ProviderName())
	fmt.Printf("  session: %s\n", sessionID)
	fmt.Printf("  session_db: %s\n", filepath.Clean(sessionDB))
	fmt.Printf("  workspace: %s\n\n", resolver.root)

	result, err := agent.Execute(ctx, map[string]any{
		"task": task,
	})
	if err != nil {
		return fmt.Errorf("execute task: %w", err)
	}

	fmt.Printf("Task: %s\n\n", task)
	fmt.Printf("Completed: %t\n", boolValue(result["completed"]))
	if answer := stringValue(result["final_answer"]); answer != "" {
		fmt.Printf("Final answer:\n%s\n", answer)
	}
	if errText := stringValue(result["error"]); errText != "" {
		fmt.Printf("Error: %s\n", errText)
	}
	fmt.Printf("Turns: %d\n", intValue(result["turns"], 0))
	fmt.Printf("Tool calls: %d\n", intValue(result["tool_calls"], 0))

	if trace := agent.LastNativeTrace(); trace != nil {
		fmt.Printf("Trace steps recorded: %d\n", len(trace.Steps))
	}

	if strings.TrimSpace(forkName) != "" {
		branch, err := agent.ForkActiveSession(ctx, sessionID, forkName, activateFork)
		if err != nil {
			return fmt.Errorf("fork active session: %w", err)
		}
		fmt.Printf("\nForked branch:\n")
		fmt.Printf("  id: %s\n", branch.ID)
		fmt.Printf("  name: %s\n", branch.Name)
		fmt.Printf("  activate_fork: %t\n", activateFork)
	}

	if showState {
		if err := printSessionState(ctx, agent, sessionID); err != nil {
			return fmt.Errorf("load session state: %w", err)
		}
	}
	return nil
}

func printUsage(w io.Writer) {
	fmt.Fprintf(w, "Usage:\n")
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -task \"Read project_brief.md, create rollout_notes.md with a short rollout plan, then finish.\"\n")
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -task \"Edit rollout_notes.md to add a rollback note, then finish.\"\n")
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -task \"What file did you create in the previous run? Answer from session recall, then finish.\"\n")
}

func printSessionState(ctx context.Context, agent *native.Agent, sessionID string) error {
	state, err := agent.GetSessionState(ctx, sessionID)
	if err != nil {
		return err
	}

	fmt.Printf("\nSession state\n")
	fmt.Printf("  session_id: %s\n", state.Session.ID)
	if state.ActiveBranch != nil {
		fmt.Printf("  active_branch_id: %s\n", state.ActiveBranch.ID)
		fmt.Printf("  active_branch_name: %s\n", state.ActiveBranch.Name)
	} else {
		fmt.Printf("  active_branch_id: <none>\n")
	}
	if state.HeadEntry != nil {
		fmt.Printf("  head_entry_id: %s\n", state.HeadEntry.ID)
		fmt.Printf("  head_entry_kind: %s\n", state.HeadEntry.Kind)
	}
	fmt.Printf("  branches:\n")
	for _, branch := range state.Branches {
		prefix := " "
		if state.ActiveBranch != nil && branch.ID == state.ActiveBranch.ID {
			prefix = "*"
		}
		fmt.Printf("  %s %s  name=%s", prefix, branch.ID, branch.Name)
		if branch.OriginEntryID != "" {
			fmt.Printf("  origin=%s", branch.OriginEntryID)
		}
		if branch.HeadEntryID != "" {
			fmt.Printf("  head=%s", branch.HeadEntryID)
		}
		fmt.Printf("\n")
	}
	return nil
}

func makeEventPrinter(verbose bool) func(agents.AgentEvent) {
	if !verbose {
		return nil
	}
	return func(event agents.AgentEvent) {
		switch event.Type {
		case agents.EventSessionLoaded:
			fmt.Printf("[event] session_loaded source=%s branch=%s entries=%d summaries=%d\n",
				stringValue(event.Data["source"]),
				stringValue(event.Data["branch_id"]),
				intValue(event.Data["entry_count"], 0),
				intValue(event.Data["summary_count"], 0),
			)
		case agents.EventToolCallStarted:
			fmt.Printf("[event] tool_call_started tool=%s\n", stringValue(event.Data["tool_name"]))
		case agents.EventToolCallFinished:
			fmt.Printf("[event] tool_call_finished tool=%s error=%t\n",
				stringValue(event.Data["tool_name"]),
				boolValue(event.Data["is_error"]),
			)
		case agents.EventRunFinished:
			fmt.Printf("[event] run_finished completed=%t turns=%d tool_calls=%d\n",
				boolValue(event.Data["completed"]),
				intValue(event.Data["turns"], 0),
				intValue(event.Data["tool_calls"], 0),
			)
		}
	}
}

type workspaceResolver struct {
	root string
}

func newWorkspaceResolver(root string) (workspaceResolver, error) {
	if strings.TrimSpace(root) == "" {
		root = defaultWorkspaceDir
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return workspaceResolver{}, err
	}
	if err := os.MkdirAll(absRoot, 0o755); err != nil {
		return workspaceResolver{}, err
	}
	return workspaceResolver{root: filepath.Clean(absRoot)}, nil
}

func (r workspaceResolver) resolveSecurePath(input string) (string, error) {
	raw := strings.TrimSpace(input)
	if raw == "" || raw == "." {
		return r.root, nil
	}

	// This is demo-grade confinement for the example: it resolves existing
	// symlinks and rejects paths outside the workspace root, but it does not
	// eliminate TOCTOU races between path resolution and later file operations.
	target := filepath.Join(r.root, filepath.Clean(raw))
	resolved, err := resolvePathThroughExistingSymlinks(target)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", input, err)
	}
	return r.ensureWithinRoot(resolved, input)
}

func (r workspaceResolver) ensureWithinRoot(target, original string) (string, error) {
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(r.root, absTarget)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes workspace root", original)
	}
	return filepath.Clean(absTarget), nil
}

func (r workspaceResolver) displayPath(target string) string {
	rel, err := filepath.Rel(r.root, target)
	if err != nil || rel == "." {
		return "."
	}
	return filepath.ToSlash(rel)
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

func ensureWorkspaceSeed(root string) error {
	seedPath := filepath.Join(root, "project_brief.md")
	if _, err := os.Stat(seedPath); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	content := strings.TrimSpace(`
# Project Brief

Project: Aurora deployment refresh

Goals:
- reduce deploy time under 10 minutes
- add a rollback checklist
- keep customer-facing downtime under 60 seconds

Known risks:
- database migration sequencing
- stale worker processes after rollout
`) + "\n"

	return os.WriteFile(seedPath, []byte(content), 0o644)
}

func newWorkspaceTools(resolver workspaceResolver) []core.Tool {
	return []core.Tool{
		newWorkspaceTool(
			"list_files",
			"List files and directories within the workspace.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative path to list. Defaults to the workspace root.",
					},
					"recursive": {
						Type:        "boolean",
						Description: "Whether to recursively list descendants.",
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := resolver.resolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				recursive := boolValue(params["recursive"])
				entries, err := listWorkspaceEntries(targetPath, resolver.root, recursive)
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				text := strings.Join(entries, "\n")
				if text == "" {
					text = "(empty)"
				}
				return successToolResult(
					truncateRunes(text, modelOutputLimit),
					truncateRunes(text, displayOutputLimit),
					map[string]any{"path": resolver.displayPath(targetPath), "entry_count": len(entries)},
				), nil
			},
		),
		newWorkspaceTool(
			"read_file",
			"Read a UTF-8 text file inside the workspace.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative file path to read.",
						Required:    true,
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := resolver.resolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				data, err := os.ReadFile(targetPath)
				if err != nil {
					return errorToolResult(fmt.Sprintf("read file: %v", err)), nil
				}
				content := string(data)
				return successToolResult(
					truncateRunes(content, modelOutputLimit),
					truncateRunes(content, displayOutputLimit),
					map[string]any{"path": resolver.displayPath(targetPath), "bytes": len(data)},
				), nil
			},
		),
		newWorkspaceTool(
			"write_file",
			"Write a UTF-8 text file inside the workspace, creating parent directories when needed.",
			models.InputSchema{
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
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := resolver.resolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				content := stringValue(params["content"])
				if content == "" {
					return errorToolResult("content is required"), nil
				}
				if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
					return errorToolResult(fmt.Sprintf("create parent directories: %v", err)), nil
				}
				if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
					return errorToolResult(fmt.Sprintf("write file: %v", err)), nil
				}
				message := fmt.Sprintf("wrote %d bytes to %s", len(content), resolver.displayPath(targetPath))
				return successToolResult(message, message, map[string]any{"path": resolver.displayPath(targetPath), "bytes": len(content)}), nil
			},
		),
		newWorkspaceTool(
			"edit_file",
			"Edit an existing UTF-8 text file by replacing exact text.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative file path to edit.",
						Required:    true,
					},
					"old_text": {
						Type:        "string",
						Description: "Exact existing text to replace.",
						Required:    true,
					},
					"new_text": {
						Type:        "string",
						Description: "Replacement text.",
						Required:    true,
					},
					"replace_all": {
						Type:        "boolean",
						Description: "Replace every occurrence instead of only the first match.",
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := resolver.resolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				oldText := stringValue(params["old_text"])
				if oldText == "" {
					return errorToolResult("old_text is required"), nil
				}
				newText := stringValue(params["new_text"])
				data, err := os.ReadFile(targetPath)
				if err != nil {
					return errorToolResult(fmt.Sprintf("read file: %v", err)), nil
				}
				content := string(data)
				if !strings.Contains(content, oldText) {
					return errorToolResult("old_text was not found in the target file"), nil
				}

				replaceAll := boolValue(params["replace_all"])
				updated := content
				replacements := 1
				if replaceAll {
					replacements = strings.Count(content, oldText)
					updated = strings.ReplaceAll(content, oldText, newText)
				} else {
					updated = strings.Replace(content, oldText, newText, 1)
				}

				if err := os.WriteFile(targetPath, []byte(updated), 0o644); err != nil {
					return errorToolResult(fmt.Sprintf("write file: %v", err)), nil
				}

				message := fmt.Sprintf("edited %s with %d replacement(s)", resolver.displayPath(targetPath), replacements)
				return successToolResult(message, message, map[string]any{
					"path":          resolver.displayPath(targetPath),
					"replacements":  replacements,
					"replace_all":   replaceAll,
					"old_text_size": len(oldText),
					"new_text_size": len(newText),
				}), nil
			},
		),
	}
}

type workspaceTool struct {
	name        string
	description string
	schema      models.InputSchema
	run         func(context.Context, map[string]any) (core.ToolResult, error)
}

func newWorkspaceTool(name, description string, schema models.InputSchema, run func(context.Context, map[string]any) (core.ToolResult, error)) *workspaceTool {
	return &workspaceTool{
		name:        name,
		description: description,
		schema:      schema,
		run:         run,
	}
}

func (t *workspaceTool) Name() string {
	return t.name
}

func (t *workspaceTool) Description() string {
	return t.description
}

func (t *workspaceTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         t.name,
		Description:  t.description,
		InputSchema:  t.schema,
		Capabilities: []string{"workspace", "files"},
		Version:      "1.0.0",
	}
}

func (t *workspaceTool) CanHandle(_ context.Context, intent string) bool {
	intent = strings.ToLower(intent)
	return strings.Contains(intent, "file") || strings.Contains(intent, "workspace") || strings.Contains(intent, "edit") || strings.Contains(intent, "read") || strings.Contains(intent, "write")
}

func (t *workspaceTool) Execute(ctx context.Context, params map[string]any) (core.ToolResult, error) {
	return t.run(ctx, params)
}

func (t *workspaceTool) Validate(params map[string]any) error {
	for name, property := range t.schema.Properties {
		if property.Required && strings.TrimSpace(stringValue(params[name])) == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	return nil
}

func (t *workspaceTool) InputSchema() models.InputSchema {
	return t.schema
}

func (t *workspaceTool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	return &cloned
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

func errorToolResult(message string) core.ToolResult {
	return core.ToolResult{
		Data: message,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   message,
			core.ToolResultDisplayTextMeta: message,
			core.ToolResultIsErrorMeta:     true,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: map[string]any{"error": message},
		},
	}
}

func listWorkspaceEntries(targetPath, workspaceRoot string, recursive bool) ([]string, error) {
	if recursive {
		entries := make([]string, 0, 16)
		err := filepath.WalkDir(targetPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, relErr := filepath.Rel(workspaceRoot, path)
			if relErr != nil {
				return relErr
			}
			if rel == "." {
				entries = append(entries, ".")
				return nil
			}
			label := filepath.ToSlash(rel)
			if d.IsDir() {
				label += "/"
			}
			entries = append(entries, label)
			return nil
		})
		return entries, err
	}

	dirEntries, err := os.ReadDir(targetPath)
	if err != nil {
		return nil, err
	}
	entries := make([]string, 0, len(dirEntries))
	for _, entry := range dirEntries {
		childPath := filepath.Join(targetPath, entry.Name())
		rel, relErr := filepath.Rel(workspaceRoot, childPath)
		if relErr != nil {
			return nil, relErr
		}
		label := filepath.ToSlash(rel)
		if entry.IsDir() {
			label += "/"
		}
		entries = append(entries, label)
	}
	return entries, nil
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

func boolValue(value any) bool {
	typed, ok := value.(bool)
	return ok && typed
}

func intValue(value any, fallback int) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float32:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return fallback
	}
}
