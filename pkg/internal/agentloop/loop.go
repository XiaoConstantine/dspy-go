package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/schema"
)

// SubmitToolName is the reserved tool name used by ReAct and multi-output
// agents to submit validated final signature outputs.
const SubmitToolName = "submit"

// TurnStep records one executed tool call for tracing in modules.ReAct and pkg/agents.
type TurnStep struct {
	Turn        int
	Thought     string
	Tool        string
	Arguments   map[string]any
	Observation string
	Duration    time.Duration
	Success     bool
	Error       string
}

// Config configures one sequential native tool-calling run with schema-validated submit.
type Config struct {
	Signature        core.Signature
	Tools            []core.Tool
	MaxTurns         int
	ForceFinalSubmit bool
	Options          []core.GenerateOption
	ReasoningEffort  core.ReasoningEffort
	Execute          func(ctx context.Context, tool core.Tool, args map[string]any) (core.ToolResult, error)
}

// Result is the terminal outcome of a Run call.
type Result struct {
	Outputs          map[string]any
	Messages         []core.ChatMessage
	Steps            []TurnStep
	Turns            int
	ToolCalls        int
	TerminationCause string
	Usage            *core.TokenInfo
}

// Run executes a native tool-calling loop over core.ChatLLM with a reserved
// schema-validated `submit` tool. Every model call counts against MaxTurns,
// including the forced final submit call when ForceFinalSubmit is enabled.
func Run(
	ctx context.Context,
	model core.LLM,
	initialMessages []core.ChatMessage,
	cfg Config,
) (*Result, error) {
	chatLLM, ok := core.ResolveChatLLM(model)
	if !ok {
		return nil, dspyerrors.New(dspyerrors.UnsupportedOperation, "model does not implement core.ChatLLM")
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 1
	}

	compiledSubmit, err := schema.CompileSignatureSchema(cfg.Signature)
	if err != nil {
		return nil, fmt.Errorf("compile submit schema: %w", err)
	}

	toolByName := make(map[string]core.Tool, len(cfg.Tools))
	chatTools := make([]core.ChatTool, 0, len(cfg.Tools)+1)
	for _, tool := range cfg.Tools {
		if tool == nil {
			continue
		}
		name := strings.TrimSpace(tool.Name())
		if name == "" {
			return nil, dspyerrors.New(dspyerrors.InvalidInput, "tool name must not be empty")
		}
		if strings.EqualFold(name, SubmitToolName) {
			return nil, dspyerrors.New(dspyerrors.InvalidInput, "`submit` is reserved as the final-output tool")
		}
		if _, dup := toolByName[name]; dup {
			return nil, dspyerrors.New(dspyerrors.InvalidInput, fmt.Sprintf("duplicate tool %q", name))
		}
		toolByName[name] = tool
		chatTools = append(chatTools, chatToolFromCoreTool(tool))
	}
	sort.Slice(chatTools, func(i, j int) bool { return chatTools[i].Name < chatTools[j].Name })

	submitTool := core.ChatTool{
		Name:        SubmitToolName,
		Description: "Submit the final validated outputs for the task.",
		InputSchema: compiledSubmit.Raw,
		Strictness:  core.ToolStrictPrefer,
	}
	chatTools = append(chatTools, submitTool)

	execFn := cfg.Execute
	if execFn == nil {
		execFn = func(ctx context.Context, tool core.Tool, args map[string]any) (core.ToolResult, error) {
			return tool.Execute(ctx, args)
		}
	}

	transcript := append([]core.ChatMessage(nil), initialMessages...)
	res := &Result{}

	normalTurns := cfg.MaxTurns
	if cfg.ForceFinalSubmit && cfg.MaxTurns > 1 {
		normalTurns = cfg.MaxTurns - 1
	}

	var forcedNeeded bool
	var breakReason string

	for turn := 1; turn <= normalTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res.Turns++

		req := core.ChatRequest{
			Messages:        transcript,
			Tools:           chatTools,
			ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceAuto},
			ReasoningEffort: cfg.ReasoningEffort,
			Options:         cfg.Options,
		}
		// If MaxTurns == 1 and ForceFinalSubmit is true, force submit on the only turn.
		if cfg.ForceFinalSubmit && cfg.MaxTurns == 1 {
			req.ToolChoice = core.ToolChoice{Mode: core.ToolChoiceNamed, Name: SubmitToolName}
			req.ReasoningEffort = core.ReasoningEffortNone
		}

		core.RecordLLMCall(ctx, model)
		resp, err := chatLLM.Chat(ctx, req)
		if err != nil {
			return res, err
		}
		addUsage(res, resp.Usage)

		if statusErr := checkChatStatus(resp); statusErr != nil {
			return res, statusErr
		}

		assistantMsg := core.ChatMessage{
			Role:            "assistant",
			Content:         resp.ContentBlocks,
			ToolCalls:       resp.ToolCalls,
			RawProviderData: resp.ProviderData,
		}
		if len(assistantMsg.Content) == 0 && resp.Content != "" {
			assistantMsg.Content = []core.ContentBlock{core.NewTextBlock(resp.Content)}
		}
		transcript = append(transcript, assistantMsg)

		if len(resp.ToolCalls) == 0 {
			if cfg.ForceFinalSubmit && res.Turns < cfg.MaxTurns {
				forcedNeeded = true
				breakReason = "empty_tool_calls"
				break
			}
			return res, &dspyerrors.ParseError{
				Adapter: "react",
				Message: "model returned no tool calls and did not call submit",
				Raw:     resp.Content,
			}
		}

		toolResults := make([]core.ChatToolResult, 0, len(resp.ToolCalls))
		var finalOutputs map[string]any

		for _, call := range resp.ToolCalls {
			res.ToolCalls++
			if call.Name == SubmitToolName {
				validated, valErr := ValidateSubmitArguments(cfg.Signature, compiledSubmit, call.Arguments)
				if valErr != nil {
					errText := fmt.Sprintf("invalid submit arguments: %v", valErr)
					toolResults = append(toolResults, core.ChatToolResult{
						ToolCallID: call.ID,
						Name:       call.Name,
						Content:    []core.ContentBlock{core.NewTextBlock(errText)},
						IsError:    true,
					})
					res.Steps = append(res.Steps, TurnStep{
						Turn:        turn,
						Thought:     resp.Content,
						Tool:        call.Name,
						Arguments:   maps.Clone(call.Arguments),
						Observation: errText,
						Success:     false,
						Error:       errText,
					})
					continue
				}
				finalOutputs = validated
				toolResults = append(toolResults, core.ChatToolResult{
					ToolCallID: call.ID,
					Name:       call.Name,
					Content:    []core.ContentBlock{core.NewTextBlock("submitted")},
				})
				res.Steps = append(res.Steps, TurnStep{
					Turn:        turn,
					Thought:     resp.Content,
					Tool:        call.Name,
					Arguments:   maps.Clone(validated),
					Observation: "submitted",
					Success:     true,
				})
				continue
			}

			tool, exists := toolByName[call.Name]
			if !exists {
				errText := fmt.Sprintf("unknown tool %q", call.Name)
				toolResults = append(toolResults, core.ChatToolResult{
					ToolCallID: call.ID,
					Name:       call.Name,
					Content:    []core.ContentBlock{core.NewTextBlock(errText)},
					IsError:    true,
				})
				res.Steps = append(res.Steps, TurnStep{
					Turn:        turn,
					Thought:     resp.Content,
					Tool:        call.Name,
					Arguments:   maps.Clone(call.Arguments),
					Observation: errText,
					Success:     false,
					Error:       errText,
				})
				continue
			}

			args := maps.Clone(call.Arguments)
			if args == nil {
				args = map[string]any{}
			}
			if vErr := tool.Validate(args); vErr != nil {
				errText := fmt.Sprintf("invalid arguments for tool %q: %v", call.Name, vErr)
				toolResults = append(toolResults, core.ChatToolResult{
					ToolCallID: call.ID,
					Name:       call.Name,
					Content:    []core.ContentBlock{core.NewTextBlock(errText)},
					IsError:    true,
				})
				res.Steps = append(res.Steps, TurnStep{
					Turn:        turn,
					Thought:     resp.Content,
					Tool:        call.Name,
					Arguments:   args,
					Observation: errText,
					Success:     false,
					Error:       errText,
				})
				continue
			}

			start := time.Now()
			toolRes, execErr := execFn(ctx, tool, args)
			dur := time.Since(start)
			if execErr != nil {
				errText := fmt.Sprintf("error executing tool %q: %v", call.Name, execErr)
				toolResults = append(toolResults, core.ChatToolResult{
					ToolCallID: call.ID,
					Name:       call.Name,
					Content:    []core.ContentBlock{core.NewTextBlock(errText)},
					IsError:    true,
				})
				res.Steps = append(res.Steps, TurnStep{
					Turn:        turn,
					Thought:     resp.Content,
					Tool:        call.Name,
					Arguments:   args,
					Observation: errText,
					Duration:    dur,
					Success:     false,
					Error:       execErr.Error(),
				})
				continue
			}

			obsText := formatToolResultText(toolRes)
			toolResults = append(toolResults, core.ChatToolResult{
				ToolCallID: call.ID,
				Name:       call.Name,
				Content:    []core.ContentBlock{core.NewTextBlock(obsText)},
			})
			res.Steps = append(res.Steps, TurnStep{
				Turn:        turn,
				Thought:     resp.Content,
				Tool:        call.Name,
				Arguments:   args,
				Observation: obsText,
				Duration:    dur,
				Success:     true,
			})
		}

		transcript = append(transcript, core.ChatMessage{
			Role:        "tool",
			ToolResults: toolResults,
		})

		if finalOutputs != nil {
			res.Outputs = finalOutputs
			res.Messages = transcript
			if cfg.ForceFinalSubmit && cfg.MaxTurns == 1 {
				res.TerminationCause = "forced_submit"
			} else {
				res.TerminationCause = "submit"
			}
			return res, nil
		}
	}

	if !forcedNeeded && cfg.ForceFinalSubmit && res.Turns < cfg.MaxTurns {
		forcedNeeded = true
		breakReason = "max_turns"
	}

	if !forcedNeeded {
		res.Messages = transcript
		res.TerminationCause = "max_turns"
		return res, &dspyerrors.ParseError{
			Adapter: "react",
			Message: fmt.Sprintf("ReAct reached max turns (%d) without calling submit", cfg.MaxTurns),
		}
	}

	// Forced final submit turn (counted against MaxTurns budget).
	if err := ctx.Err(); err != nil {
		return res, err
	}
	res.Turns++
	transcript = append(transcript, core.ChatMessage{
		Role:    "user",
		Content: []core.ContentBlock{core.NewTextBlock("No turns remain. Call `submit` now with your final outputs.")},
	})

	forcedReq := core.ChatRequest{
		Messages:        transcript,
		Tools:           chatTools,
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceNamed, Name: SubmitToolName},
		ReasoningEffort: core.ReasoningEffortNone,
		Options:         cfg.Options,
	}
	core.RecordLLMCall(ctx, model)
	resp, err := chatLLM.Chat(ctx, forcedReq)
	if err != nil {
		return res, err
	}
	addUsage(res, resp.Usage)
	if statusErr := checkChatStatus(resp); statusErr != nil {
		return res, statusErr
	}

	for _, call := range resp.ToolCalls {
		if call.Name != SubmitToolName {
			continue
		}
		res.ToolCalls++
		validated, valErr := ValidateSubmitArguments(cfg.Signature, compiledSubmit, call.Arguments)
		if valErr != nil {
			return res, &dspyerrors.ParseError{
				Adapter: "react",
				Message: fmt.Sprintf("forced submit after %s failed validation", breakReason),
				Raw:     resp.Content,
				Err:     valErr,
			}
		}
		res.Outputs = validated
		res.Messages = transcript
		res.TerminationCause = "forced_submit"
		res.Steps = append(res.Steps, TurnStep{
			Turn:        res.Turns,
			Thought:     resp.Content,
			Tool:        SubmitToolName,
			Arguments:   maps.Clone(validated),
			Observation: "submitted",
			Success:     true,
		})
		return res, nil
	}

	return res, &dspyerrors.ParseError{
		Adapter: "react",
		Message: fmt.Sprintf("forced submit after %s produced no submit tool call", breakReason),
		Raw:     resp.Content,
	}
}

// ValidateSubmitArguments validates submit tool arguments locally against the
// compiled signature schema and coerces every field to its declared Go type.
func ValidateSubmitArguments(
	sig core.Signature,
	compiled schema.CompiledSchema,
	args map[string]any,
) (map[string]any, error) {
	if args == nil {
		args = map[string]any{}
	}
	// Fill nil for omitted optional fields so strict schema validation passes.
	normalized := make(map[string]any, len(args)+len(sig.Outputs))
	for k, v := range args {
		normalized[k] = v
	}
	for _, f := range sig.Outputs {
		if !f.IsRequired() {
			if _, ok := normalized[f.Name]; !ok {
				normalized[f.Name] = nil
			}
		}
	}

	raw, err := json.Marshal(normalized)
	if err != nil {
		return nil, fmt.Errorf("marshal submit args: %w", err)
	}
	if err := schema.ValidateJSON(compiled.Raw, raw); err != nil {
		return nil, err
	}

	out := make(map[string]any, len(sig.Outputs))
	for _, f := range sig.Outputs {
		coerced, err := schema.CoerceFieldValue(f, normalized[f.Name])
		if err != nil {
			return nil, fmt.Errorf("field %q: %w", f.Name, err)
		}
		out[f.Name] = coerced
	}
	return out, nil
}

func checkChatStatus(resp *core.ChatResponse) error {
	if resp == nil {
		return &dspyerrors.ParseError{Adapter: "react", Message: "nil response from model"}
	}
	if resp.Refusal != nil || resp.FinishReason == core.FinishReasonRefusal {
		reason := ""
		if resp.Refusal != nil {
			reason = *resp.Refusal
		}
		return &dspyerrors.RefusalError{Reason: reason}
	}
	if resp.FinishReason == core.FinishReasonContentFilter {
		return &dspyerrors.ContentFilterError{Raw: resp.Content}
	}
	if resp.FinishReason == core.FinishReasonLength {
		return &dspyerrors.TruncatedError{Raw: resp.Content}
	}
	return nil
}

func chatToolFromCoreTool(tool core.Tool) core.ChatTool {
	schema := tool.InputSchema()
	props := make(map[string]any, len(schema.Properties))
	required := make([]string, 0, len(schema.Properties))
	for name, prop := range schema.Properties {
		def := map[string]any{
			"type":        prop.Type,
			"description": prop.Description,
		}
		if prop.Minimum != nil {
			def["minimum"] = *prop.Minimum
		}
		if prop.Maximum != nil {
			def["maximum"] = *prop.Maximum
		}
		props[name] = def
		if prop.Required {
			required = append(required, name)
		}
	}
	sort.Strings(required)
	schemaType := strings.TrimSpace(schema.Type)
	if schemaType == "" {
		schemaType = "object"
	}
	raw, _ := json.Marshal(map[string]any{
		"type":       schemaType,
		"properties": props,
		"required":   required,
	})
	return core.ChatTool{
		Name:        strings.TrimSpace(tool.Name()),
		Description: tool.Description(),
		InputSchema: raw,
	}
}

func formatToolResultText(result core.ToolResult) string {
	if result.Data == nil {
		return ""
	}
	switch v := result.Data.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		if encoded, err := json.Marshal(v); err == nil {
			return string(encoded)
		}
		return fmt.Sprintf("%v", v)
	}
}

func addUsage(res *Result, usage *core.TokenInfo) {
	if usage == nil {
		return
	}
	if res.Usage == nil {
		res.Usage = &core.TokenInfo{}
	}
	res.Usage.PromptTokens += usage.PromptTokens
	res.Usage.CompletionTokens += usage.CompletionTokens
	res.Usage.TotalTokens += usage.TotalTokens
}
