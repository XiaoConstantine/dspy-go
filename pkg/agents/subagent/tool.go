package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

const (
	defaultModelTextLimit   = 1600
	defaultDisplayTextLimit = 6000
)

type BuildAgentFunc func(ctx context.Context, args map[string]any) (agents.Agent, error)
type BuildInputFunc func(args map[string]any, parent ParentContext) (map[string]any, error)
type BuildResultFunc func(output map[string]any, run ResultContext) (core.ToolResult, error)

type ToolConfig struct {
	Name        string
	Description string
	InputSchema models.InputSchema

	BuildAgent BuildAgentFunc
	Agent      agents.Agent

	StaticParentContext ParentContext

	BuildInput  BuildInputFunc
	BuildResult BuildResultFunc

	SessionPolicy SessionPolicy
	Timeout       time.Duration
	Metadata      map[string]string
}

type Tool struct {
	name                string
	description         string
	inputSchema         models.InputSchema
	buildAgent          BuildAgentFunc
	staticParentContext ParentContext
	buildInput          BuildInputFunc
	buildResult         BuildResultFunc
	sessionPolicy       SessionPolicy
	timeout             time.Duration
	metadata            map[string]string
}

type traceProvider interface {
	LastExecutionTrace() *agents.ExecutionTrace
}

// Info captures static metadata about a subagent-wrapped tool.
type Info struct {
	Name          string
	SessionPolicy string
}

type infoProvider interface {
	subagentInfo() Info
}

// InfoFromTool returns subagent metadata when tool is a subagent-wrapped tool.
func InfoFromTool(tool core.Tool) (Info, bool) {
	if tool == nil {
		return Info{}, false
	}
	provider, ok := tool.(infoProvider)
	if !ok {
		return Info{}, false
	}
	return provider.subagentInfo(), true
}

func AsTool(cfg ToolConfig) (core.Tool, error) {
	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		return nil, fmt.Errorf("subagent tool name is required")
	}

	description := strings.TrimSpace(cfg.Description)
	if description == "" {
		return nil, fmt.Errorf("subagent tool description is required")
	}

	buildAgent, err := makeBuildAgentFunc(cfg)
	if err != nil {
		return nil, err
	}

	buildInput := cfg.BuildInput
	if buildInput == nil {
		buildInput = defaultBuildInput
	}

	buildResult := cfg.BuildResult
	if buildResult == nil {
		buildResult = defaultBuildResult(name, cfg.SessionPolicy)
	}

	schema := cfg.InputSchema
	if strings.TrimSpace(schema.Type) == "" {
		schema.Type = "object"
	}

	return &Tool{
		name:                name,
		description:         description,
		inputSchema:         schema,
		buildAgent:          buildAgent,
		staticParentContext: cloneParentContext(cfg.StaticParentContext),
		buildInput:          buildInput,
		buildResult:         buildResult,
		sessionPolicy:       cfg.SessionPolicy,
		timeout:             cfg.Timeout,
		metadata:            core.ShallowCopyMap(cfg.Metadata),
	}, nil
}

func (t *Tool) Name() string {
	return t.name
}

func (t *Tool) Description() string {
	return t.description
}

func (t *Tool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         t.name,
		Description:  t.description,
		InputSchema:  t.InputSchema(),
		Capabilities: []string{"subagent", "delegation", "isolated"},
		Version:      "1.0.0",
	}
}

func (t *Tool) CanHandle(_ context.Context, intent string) bool {
	intent = strings.ToLower(strings.TrimSpace(intent))
	if intent == "" {
		return false
	}
	return containsIntentToken(intent, t.name) || containsIntentToken(intent, t.description)
}

func (t *Tool) Validate(_ map[string]any) error {
	return nil
}

func (t *Tool) InputSchema() models.InputSchema {
	return t.inputSchema
}

func (t *Tool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	cloned.staticParentContext = cloneParentContext(t.staticParentContext)
	cloned.metadata = core.ShallowCopyMap(t.metadata)
	return &cloned
}

func (t *Tool) subagentInfo() Info {
	if t == nil {
		return Info{}
	}
	return Info{
		Name:          t.name,
		SessionPolicy: t.sessionPolicy.String(),
	}
}

func (t *Tool) Execute(ctx context.Context, params map[string]any) (core.ToolResult, error) {
	params = core.ShallowCopyMap(params)
	parent := t.parentContext(ctx, params)

	runCtx := ctx
	var cancel context.CancelFunc
	if t.timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, t.timeout)
		defer cancel()
	}

	child, err := t.buildAgent(runCtx, core.ShallowCopyMap(params))
	if err != nil {
		return t.errorResult(fmt.Sprintf("subagent %q could not start: %v", t.name, err), nil, SubagentTraceRef{}), nil
	}
	if child == nil {
		return t.errorResult(fmt.Sprintf("subagent %q returned a nil agent", t.name), nil, SubagentTraceRef{}), nil
	}

	childInput, err := t.buildInput(core.ShallowCopyMap(params), parent)
	if err != nil {
		return t.errorResult(fmt.Sprintf("subagent %q input build failed: %v", t.name, err), nil, SubagentTraceRef{}), nil
	}
	childInput = applySessionPolicy(t.sessionPolicy, t.name, childInput, parent)

	startedAt := time.Now()
	output, execErr := child.Execute(runCtx, childInput)
	duration := time.Since(startedAt)

	trace := childTrace(child)
	run := ResultContext{
		Output:    core.ShallowCopyMap(output),
		Duration:  duration,
		Completed: execErr == nil && completedValue(output),
		Trace:     trace,
	}

	if execErr != nil {
		return t.errorResult(fmt.Sprintf("subagent %q failed: %v", t.name, execErr), output, run.TraceRef(t.name)), nil
	}

	result, err := t.buildResult(core.ShallowCopyMap(output), run)
	if err != nil {
		return t.errorResult(fmt.Sprintf("subagent %q result build failed: %v", t.name, err), output, run.TraceRef(t.name)), nil
	}
	return result, nil
}

func (r ResultContext) TraceRef(name string) SubagentTraceRef {
	return SubagentTraceRef{
		Name:      name,
		Completed: r.Completed,
		Duration:  r.Duration,
		Trace:     r.Trace.Clone(),
	}
}

func (t *Tool) parentContext(ctx context.Context, params map[string]any) ParentContext {
	parent := mergeParentContext(t.staticParentContext, ParentContextFromContext(ctx))
	if parent.TaskID == "" {
		parent.TaskID = stringArg(params, "task_id")
	}
	if parent.ParentAgentID == "" {
		parent.ParentAgentID = stringArg(params, "parent_agent_id")
	}
	if parent.ParentAgentType == "" {
		parent.ParentAgentType = stringArg(params, "parent_agent_type")
	}
	if parent.SessionID == "" {
		parent.SessionID = stringArg(params, "session_id")
	}
	if parent.BranchID == "" {
		parent.BranchID = stringArg(params, "session_branch_id")
	}
	return parent
}

func makeBuildAgentFunc(cfg ToolConfig) (BuildAgentFunc, error) {
	if cfg.BuildAgent != nil && cfg.Agent != nil {
		return nil, fmt.Errorf("subagent tool %q cannot set both BuildAgent and Agent", strings.TrimSpace(cfg.Name))
	}
	if cfg.BuildAgent != nil {
		return cfg.BuildAgent, nil
	}
	if cfg.Agent == nil {
		return nil, fmt.Errorf("subagent tool %q requires BuildAgent or a cloneable Agent", strings.TrimSpace(cfg.Name))
	}
	cloneable, ok := cfg.Agent.(optimize.OptimizableAgent)
	if !ok {
		return nil, fmt.Errorf("subagent tool %q requires BuildAgent or an optimize.OptimizableAgent", strings.TrimSpace(cfg.Name))
	}
	return func(context.Context, map[string]any) (agents.Agent, error) {
		return cloneable.Clone()
	}, nil
}

func defaultBuildInput(args map[string]any, _ ParentContext) (map[string]any, error) {
	return core.ShallowCopyMap(args), nil
}

func defaultBuildResult(name string, policy SessionPolicy) BuildResultFunc {
	return func(output map[string]any, run ResultContext) (core.ToolResult, error) {
		modelText, displayText := summarizeOutput(output)
		traceRef := run.TraceRef(name)
		details := map[string]any{
			"subagent":       true,
			"subagent_name":  name,
			"session_policy": policy.String(),
			"completed":      run.Completed,
			"duration_ms":    run.Duration.Milliseconds(),
			"output":         core.ShallowCopyMap(output),
			"trace":          traceRef,
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
		}, nil
	}
}

func (t *Tool) errorResult(message string, output map[string]any, traceRef SubagentTraceRef) core.ToolResult {
	modelText := truncateRunes(strings.TrimSpace(message), defaultModelTextLimit)
	displayText := truncateRunes(strings.TrimSpace(message), defaultDisplayTextLimit)
	details := map[string]any{
		"subagent":       true,
		"subagent_name":  t.name,
		"session_policy": t.sessionPolicy.String(),
		"completed":      false,
	}
	if output != nil {
		details["output"] = core.ShallowCopyMap(output)
	}
	if traceRef.Name != "" {
		details["trace"] = traceRef
	}
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

func applySessionPolicy(policy SessionPolicy, name string, childInput map[string]any, parent ParentContext) map[string]any {
	if childInput == nil {
		childInput = make(map[string]any)
	}
	switch policy {
	case SessionPolicyDerived:
		if strings.TrimSpace(stringArg(childInput, "session_id")) == "" && strings.TrimSpace(parent.SessionID) != "" {
			childInput["session_id"] = deriveSessionID(parent.SessionID, name)
		}
	case SessionPolicyExplicit:
		// Explicit sessions are caller-managed. Preserve provided session keys as-is.
	case SessionPolicyEphemeral:
		delete(childInput, "session_id")
		delete(childInput, "session_branch_id")
		delete(childInput, "session_fork_from_entry_id")
	}
	return childInput
}

func deriveSessionID(parentSessionID, name string) string {
	parentSessionID = strings.TrimSpace(parentSessionID)
	name = sanitizeSegment(name)
	if parentSessionID == "" {
		return name
	}
	if name == "" {
		return parentSessionID
	}
	return parentSessionID + "/" + name
}

func sanitizeSegment(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	lastDash := false
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		default:
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func childTrace(agent agents.Agent) *agents.ExecutionTrace {
	provider, ok := agent.(traceProvider)
	if !ok {
		return nil
	}
	return provider.LastExecutionTrace()
}

func summarizeOutput(output map[string]any) (string, string) {
	if text := firstString(output, "final_answer", "answer", "result", "summary", "text"); text != "" {
		return truncateRunes(text, defaultModelTextLimit), truncateRunes(text, defaultDisplayTextLimit)
	}
	if len(output) == 0 {
		return "(no output)", "(no output)"
	}
	formatted, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		fallback := truncateRunes(fmt.Sprintf("%v", output), defaultDisplayTextLimit)
		return truncateRunes(fallback, defaultModelTextLimit), fallback
	}
	display := string(formatted)
	return truncateRunes(display, defaultModelTextLimit), truncateRunes(display, defaultDisplayTextLimit)
}

func completedValue(output map[string]any) bool {
	if output == nil {
		return true
	}
	raw, ok := output["completed"]
	if !ok || raw == nil {
		return true
	}
	flag, ok := raw.(bool)
	if !ok {
		return true
	}
	return flag
}

func firstString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if text := stringArg(values, key); text != "" {
			return text
		}
	}
	return ""
}

func stringArg(values map[string]any, key string) string {
	if values == nil {
		return ""
	}
	raw, ok := values[key]
	if !ok || raw == nil {
		return ""
	}
	text, ok := raw.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func truncateRunes(value string, limit int) string {
	if limit <= 0 {
		return value
	}

	count := 0
	cut := -1
	for idx := range value {
		if count == limit {
			cut = idx
			break
		}
		count++
	}
	if cut < 0 {
		return value
	}
	return value[:cut] + "... (truncated)"
}

func containsIntentToken(intent string, source string) bool {
	for _, token := range tokenize(source) {
		if len(token) < 4 || isStopToken(token) {
			continue
		}
		if strings.Contains(intent, token) {
			return true
		}
	}
	return false
}

func tokenize(value string) []string {
	return strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func isStopToken(token string) bool {
	switch token {
	case "into", "with", "from", "that", "this", "your", "their", "then", "than":
		return true
	case "tool", "tools", "agent", "agents", "worker", "workers":
		return true
	case "task", "tasks", "run", "runs", "using", "used":
		return true
	case "perform", "return":
		return true
	default:
		return false
	}
}
