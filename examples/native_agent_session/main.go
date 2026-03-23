package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/native"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	filetools "github.com/XiaoConstantine/dspy-go/pkg/tools/files"
)

const (
	defaultModelID      = string(core.ModelGoogleGeminiFlash)
	defaultSessionID    = "native-agent-demo"
	defaultDBPath       = "./tmp/native-agent-session/session.db"
	defaultWorkspaceDir = "./tmp/native-agent-session/workspace"
	defaultTaskTimeout  = 2 * time.Minute
)

const exampleSystemPrompt = `You are a pragmatic native tool-calling assistant working inside a local workspace.

Use the available tools to inspect files, edit files, and create new files.
Prefer session recall when it already answers the question from previous runs.
Use ls before making assumptions about the workspace layout.
Use edit for targeted changes when a file already exists.
Use write when creating a new file or replacing the full contents intentionally.
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
		reset        bool
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
	flag.BoolVar(&reset, "reset", false, "Remove the session database and workspace before running for a cold-start demo")
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
	task = buildExampleTask(task)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	llms.EnsureFactory()
	llm, err := llms.NewLLM(strings.TrimSpace(apiKey), core.ModelID(strings.TrimSpace(model)))
	if err != nil {
		return fmt.Errorf("create llm for model %q: %v\nHint: pass -api-key or configure credentials for the selected provider in the environment", model, err)
	}

	if reset {
		if err := resetExampleState(sessionDB, workspaceDir); err != nil {
			return fmt.Errorf("reset example state: %w", err)
		}
	} else if reusingExistingState(sessionDB, workspaceDir) {
		fmt.Printf("Reusing existing session/workspace state. Use -reset for a cold-start run.\n\n")
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

	toolset, err := filetools.NewToolset(filetools.Config{
		Root: workspaceDir,
	})
	if err != nil {
		return fmt.Errorf("resolve workspace %q: %w", workspaceDir, err)
	}
	if err := ensureWorkspaceSeed(toolset.Root()); err != nil {
		return fmt.Errorf("prepare workspace seed: %w", err)
	}

	agent, err := native.NewAgent(llm, native.Config{
		MaxTurns:                      maxTurns,
		MaxConsecutiveNoCallResponses: 6,
		Temperature:                   temperature,
		SystemPrompt:                  exampleSystemPrompt,
		SessionID:                     sessionID,
		SessionBranchID:               strings.TrimSpace(branchID),
		SessionRecallLimit:            4,
		SessionRecallMaxChars:         1800,
		SessionEventStore:             store,
		OnEvent:                       makeEventPrinter(verbose),
	})
	if err != nil {
		return fmt.Errorf("create native agent: %w", err)
	}

	for _, tool := range toolset.Tools() {
		if err := agent.RegisterTool(tool); err != nil {
			return fmt.Errorf("register tool %q: %w", tool.Name(), err)
		}
	}

	fmt.Printf("Running native agent\n")
	fmt.Printf("  model: %s\n", llm.ModelID())
	fmt.Printf("  provider: %s\n", llm.ProviderName())
	fmt.Printf("  session: %s\n", sessionID)
	fmt.Printf("  session_db: %s\n", filepath.Clean(sessionDB))
	fmt.Printf("  workspace: %s\n\n", toolset.Root())

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
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -reset -task \"Read project_brief.md, create rollout_notes.md with a short rollout plan, then finish.\"\n")
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -task \"Edit rollout_notes.md to add a rollback note, then finish.\"\n")
	fmt.Fprintf(w, "  go run ./examples/native_agent_session -task \"What file did you create in the previous run? Answer from session recall, then finish.\"\n")
}

func buildExampleTask(task string) string {
	task = strings.TrimSpace(task)
	return fmt.Sprintf(`Complete the task using tools.

Rules:
- Start with a tool call. Do not begin with a plain-text answer.
- Use ls before assuming the workspace layout.
- Use read to inspect existing files before editing them.
- Use write when creating a new file.
- Use edit for targeted replacements in an existing file.
- If session recall already answers the question, call Finish with that answer instead of narrating first.

Task:
%s`, task)
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

func reusingExistingState(sessionDB, workspaceDir string) bool {
	if pathExists(sessionDB) {
		return true
	}
	if pathExists(workspaceDir) {
		return true
	}
	return false
}

func resetExampleState(sessionDB, workspaceDir string) error {
	if err := os.RemoveAll(filepath.Clean(workspaceDir)); err != nil {
		return err
	}
	for _, suffix := range []string{"", "-wal", "-shm"} {
		if err := os.Remove(filepath.Clean(sessionDB) + suffix); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func pathExists(path string) bool {
	_, err := os.Stat(filepath.Clean(path))
	return err == nil
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
