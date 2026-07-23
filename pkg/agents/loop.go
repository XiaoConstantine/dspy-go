package agents

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// StopReason explains why an agent loop terminated.
type StopReason string

const (
	StopReasonFinish   StopReason = "finish"
	StopReasonText     StopReason = "text"
	StopReasonMaxTurns StopReason = "max_turns"
	StopReasonCanceled StopReason = "canceled"
	StopReasonError    StopReason = "error"
)

// CompletionMode controls which model response can complete a run.
type CompletionMode string

const (
	CompletionModeFinish       CompletionMode = "finish"
	CompletionModeText         CompletionMode = "text"
	CompletionModeFinishOrText CompletionMode = "finish_or_text"
)

// CompletionConfig defines concrete terminal behavior for RunLoop.
type CompletionConfig struct {
	Mode              CompletionMode
	FinishName        string
	FinishDescription string
	FinishAnswerField string
}

// FinishCompletion requires an explicit Finish tool call.
func FinishCompletion(description string) CompletionConfig {
	return CompletionConfig{
		Mode:              CompletionModeFinish,
		FinishName:        "Finish",
		FinishDescription: description,
		FinishAnswerField: "answer",
	}
}

// TextCompletion completes when the model returns an assistant turn without
// tool calls.
func TextCompletion() CompletionConfig {
	return CompletionConfig{Mode: CompletionModeText}
}

// FinishOrTextCompletion accepts either explicit Finish or an ordinary text
// response without tool calls.
func FinishOrTextCompletion(description string) CompletionConfig {
	config := FinishCompletion(description)
	config.Mode = CompletionModeFinishOrText
	return config
}

// ToolExecutor executes one validated tool call. The default calls Tool.Execute
// directly; wrappers can provide interception without changing the loop.
type ToolExecutor func(context.Context, core.Tool, map[string]any) (core.ToolResult, error)

// LoopConfig configures one pure sequential RunLoop invocation.
type LoopConfig struct {
	RunID      string
	Task       string
	MaxTurns   int
	Completion CompletionConfig
	Options    []core.GenerateOption
	Events     EventSink
	Execute    ToolExecutor
}

// LoopResult is the provider-neutral terminal state of a RunLoop invocation.
type LoopResult struct {
	Messages    []Message
	StopReason  StopReason
	Turns       int
	ToolCalls   int
	FinalAnswer string
	Usage       *core.TokenInfo
}

// RunLoop executes model call -> assistant message -> sequential tool calls ->
// tool-result messages until the configured completion condition is reached.
func RunLoop(
	ctx context.Context,
	model Model,
	tools []core.Tool,
	messages []Message,
	config LoopConfig,
) (LoopResult, error) {
	prepared, err := prepareLoop(model, tools, config)
	if err != nil {
		return LoopResult{}, err
	}

	transcript := CloneMessages(messages)
	result := LoopResult{Messages: transcript}
	emitter := NewEventEmitter(config.Events)
	emitter.Emit(ctx, RunStartedEvent{
		RunID:    config.RunID,
		Task:     config.Task,
		MaxTurns: config.MaxTurns,
		Model:    model.ModelID(),
		Provider: model.ProviderName(),
		Messages: transcript,
	})

	finishRun := func(event RunFinishedEvent, runErr error) (LoopResult, error) {
		result.Messages = CloneMessages(transcript)
		result.StopReason = event.StopReason
		event.RunID = config.RunID
		event.Turns = result.Turns
		event.ToolCalls = result.ToolCalls
		event.FinalAnswer = result.FinalAnswer
		emitter.Emit(ctx, event)
		return result, runErr
	}

	for turn := 1; turn <= config.MaxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return finishRun(canceledRunEvent(err), err)
		}

		result.Turns = turn
		emitter.Emit(ctx, TurnStartedEvent{RunID: config.RunID, Turn: turn, MaxTurns: config.MaxTurns})
		response, modelErr := model.Complete(ctx, ModelRequest{
			Messages: CloneMessages(transcript),
			Tools:    cloneModelTools(prepared.modelTools),
			Options:  append([]core.GenerateOption(nil), config.Options...),
		})
		if modelErr != nil {
			status := OperationStatusFailed
			runEvent := RunFinishedEvent{Status: RunStatusFailed, StopReason: StopReasonError, Err: modelErr}
			if isCancellation(ctx, modelErr) {
				status = OperationStatusCanceled
				runEvent = canceledRunEvent(modelErr)
			}
			emitter.Emit(ctx, TurnFinishedEvent{RunID: config.RunID, Turn: turn, Status: status, Err: modelErr})
			return finishRun(runEvent, modelErr)
		}
		addTokenInfo(&result, response.Usage)

		assistant := response.Message.Clone()
		if len(response.Diagnostics) > 0 {
			if assistant.Metadata == nil {
				assistant.Metadata = map[string]any{}
			}
			assistant.Metadata[ModelDiagnosticsMetadataKey] = cloneAnyMap(response.Diagnostics)
		}
		if assistant.Role != RoleAssistant {
			roleErr := fmt.Errorf("model response role is %q, want %q", assistant.Role, RoleAssistant)
			emitter.Emit(ctx, TurnFinishedEvent{
				RunID: config.RunID, Turn: turn, Status: OperationStatusFailed,
				Assistant: &assistant, Usage: response.Usage, Err: roleErr,
			})
			return finishRun(RunFinishedEvent{Status: RunStatusFailed, StopReason: StopReasonError, Err: roleErr}, roleErr)
		}
		transcript = append(transcript, assistant)
		emitter.Emit(ctx, MessageAddedEvent{RunID: config.RunID, Turn: turn, Message: assistant})

		if len(assistant.ToolCalls) == 0 {
			if cancellationErr := ctx.Err(); cancellationErr != nil {
				emitter.Emit(ctx, TurnFinishedEvent{
					RunID: config.RunID, Turn: turn, Status: OperationStatusCanceled,
					Assistant: &assistant, Usage: response.Usage, Err: cancellationErr,
				})
				return finishRun(canceledRunEvent(cancellationErr), cancellationErr)
			}
			emitter.Emit(ctx, TurnFinishedEvent{
				RunID: config.RunID, Turn: turn, Status: OperationStatusCompleted,
				Assistant: &assistant, Usage: response.Usage,
			})
			if prepared.allowText {
				result.FinalAnswer = assistant.TextContent()
				return finishRun(RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonText}, nil)
			}
			continue
		}

		for toolIndex, call := range assistant.ToolCalls {
			emitter.Emit(ctx, ToolCallProposedEvent{
				RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: cloneToolCall(call),
			})
		}
		for toolIndex, call := range assistant.ToolCalls {
			call = cloneToolCall(call)
			if cancellationErr := ctx.Err(); cancellationErr != nil {
				closeCanceledCalls(ctx, emitter, config.RunID, turn, assistant.ToolCalls[toolIndex:], toolIndex, &transcript, cancellationErr)
				emitter.Emit(ctx, TurnFinishedEvent{
					RunID: config.RunID, Turn: turn, Status: OperationStatusCanceled,
					Assistant: &assistant, ToolCallCount: len(assistant.ToolCalls), Usage: response.Usage, Err: cancellationErr,
				})
				return finishRun(canceledRunEvent(cancellationErr), cancellationErr)
			}

			if prepared.isFinish(call.Name) {
				answer, finishErr := finishAnswer(call, prepared.answerField)
				if len(assistant.ToolCalls) != 1 {
					finishErr = fmt.Errorf("Finish must be the only tool call in an assistant turn")
				}
				if finishErr != nil {
					message := errorToolResultMessage(call, finishErr.Error())
					transcript = append(transcript, message)
					emitter.Emit(ctx, ToolCallFinishedEvent{
						RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
						Outcome: ToolCallOutcomeRejected, Result: &message,
						Status: OperationStatusFailed, Err: finishErr,
					})
					emitter.Emit(ctx, MessageAddedEvent{RunID: config.RunID, Turn: turn, Message: message})
					continue
				}
				emitter.Emit(ctx, ToolCallFinishedEvent{
					RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
					Outcome: ToolCallOutcomeFinish, Status: OperationStatusCompleted,
				})
				emitter.Emit(ctx, TurnFinishedEvent{
					RunID: config.RunID, Turn: turn, Status: OperationStatusCompleted,
					Assistant: &assistant, ToolCallCount: len(assistant.ToolCalls), Usage: response.Usage,
				})
				result.FinalAnswer = answer
				return finishRun(RunFinishedEvent{Status: RunStatusCompleted, StopReason: StopReasonFinish}, nil)
			}

			tool, exists := prepared.tools[call.Name]
			if !exists {
				toolErr := fmt.Errorf("unknown tool %q", call.Name)
				message := errorToolResultMessage(call, toolErr.Error())
				transcript = append(transcript, message)
				emitter.Emit(ctx, ToolCallFinishedEvent{
					RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
					Outcome: ToolCallOutcomeRejected, Result: &message,
					Status: OperationStatusFailed, Err: toolErr,
				})
				emitter.Emit(ctx, MessageAddedEvent{RunID: config.RunID, Turn: turn, Message: message})
				continue
			}
			arguments := cloneAnyMap(call.Arguments)
			if arguments == nil {
				arguments = map[string]any{}
			}
			if validationErr := tool.Validate(arguments); validationErr != nil {
				toolErr := fmt.Errorf("invalid arguments for tool %q: %w", call.Name, validationErr)
				message := errorToolResultMessage(call, toolErr.Error())
				transcript = append(transcript, message)
				emitter.Emit(ctx, ToolCallFinishedEvent{
					RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
					Outcome: ToolCallOutcomeRejected, Result: &message,
					Status: OperationStatusFailed, Err: toolErr,
				})
				emitter.Emit(ctx, MessageAddedEvent{RunID: config.RunID, Turn: turn, Message: message})
				continue
			}
			if cancellationErr := ctx.Err(); cancellationErr != nil {
				closeCanceledCalls(ctx, emitter, config.RunID, turn, assistant.ToolCalls[toolIndex:], toolIndex, &transcript, cancellationErr)
				emitter.Emit(ctx, TurnFinishedEvent{
					RunID: config.RunID, Turn: turn, Status: OperationStatusCanceled,
					Assistant: &assistant, ToolCallCount: len(assistant.ToolCalls), Usage: response.Usage, Err: cancellationErr,
				})
				return finishRun(canceledRunEvent(cancellationErr), cancellationErr)
			}

			emitter.Emit(ctx, ToolExecutionStartedEvent{
				RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
			})
			result.ToolCalls++
			var toolResult core.ToolResult
			var executeErr error
			if cancellationErr := ctx.Err(); cancellationErr != nil {
				executeErr = cancellationErr
			} else {
				toolResult, executeErr = prepared.execute(ctx, tool, arguments)
			}
			outcome := ToolCallOutcomeExecuted
			status := OperationStatusCompleted
			eventErr := executeErr
			if executeErr != nil {
				if isCancellation(ctx, executeErr) {
					status = OperationStatusCanceled
					toolResult = errorToolResult(fmt.Sprintf("tool %q execution canceled: %v", call.Name, executeErr))
				} else if errors.Is(executeErr, core.ErrToolBlocked) {
					outcome = ToolCallOutcomeBlocked
					status = OperationStatusBlocked
					toolResult = blockedToolResult(call.Name, executeErr)
				} else {
					status = OperationStatusFailed
					toolResult = errorToolResult(fmt.Sprintf("tool %q execution failed: %v", call.Name, executeErr))
				}
			} else if ctx.Err() != nil {
				status = OperationStatusCanceled
				eventErr = ctx.Err()
				toolResult = errorToolResult(fmt.Sprintf("tool %q execution canceled: %v", call.Name, ctx.Err()))
			} else if NormalizeToolResult(toolResult).IsError {
				status = OperationStatusFailed
				eventErr = fmt.Errorf("tool %q returned an error result", call.Name)
			}
			message := NewToolResultMessage(call.ID, call.Name, toolResult)
			transcript = append(transcript, message)
			emitter.Emit(ctx, ToolCallFinishedEvent{
				RunID: config.RunID, Turn: turn, ToolIndex: toolIndex, Call: call,
				Outcome: outcome, Result: &message, Status: status, Err: eventErr,
			})
			emitter.Emit(ctx, MessageAddedEvent{RunID: config.RunID, Turn: turn, Message: message})

			if isCancellation(ctx, executeErr) {
				cancellationErr := executeErr
				if cancellationErr == nil {
					cancellationErr = ctx.Err()
				}
				closeCanceledCalls(ctx, emitter, config.RunID, turn, assistant.ToolCalls[toolIndex+1:], toolIndex+1, &transcript, cancellationErr)
				emitter.Emit(ctx, TurnFinishedEvent{
					RunID: config.RunID, Turn: turn, Status: OperationStatusCanceled,
					Assistant: &assistant, ToolCallCount: len(assistant.ToolCalls), Usage: response.Usage, Err: cancellationErr,
				})
				return finishRun(canceledRunEvent(cancellationErr), cancellationErr)
			}
		}

		emitter.Emit(ctx, TurnFinishedEvent{
			RunID: config.RunID, Turn: turn, Status: OperationStatusCompleted,
			Assistant: &assistant, ToolCallCount: len(assistant.ToolCalls), Usage: response.Usage,
		})
	}

	if err := ctx.Err(); err != nil {
		return finishRun(canceledRunEvent(err), err)
	}
	diagnostic := fmt.Sprintf("max turns reached without completion after %d turns", config.MaxTurns)
	return finishRun(RunFinishedEvent{
		Status: RunStatusStopped, StopReason: StopReasonMaxTurns, Diagnostic: diagnostic,
	}, nil)
}

type preparedLoop struct {
	tools       map[string]core.Tool
	modelTools  []ModelTool
	execute     ToolExecutor
	allowText   bool
	finishName  string
	answerField string
}

func prepareLoop(model Model, tools []core.Tool, config LoopConfig) (preparedLoop, error) {
	if model == nil {
		return preparedLoop{}, fmt.Errorf("model is required")
	}
	if config.MaxTurns <= 0 {
		return preparedLoop{}, fmt.Errorf("max turns must be greater than zero")
	}
	prepared := preparedLoop{tools: make(map[string]core.Tool, len(tools))}
	for _, tool := range tools {
		definition, err := ModelToolFromTool(tool)
		if err != nil {
			return preparedLoop{}, err
		}
		definition.Name = strings.TrimSpace(definition.Name)
		if definition.Name == "" {
			return preparedLoop{}, fmt.Errorf("tool name is required")
		}
		if _, exists := prepared.tools[definition.Name]; exists {
			return preparedLoop{}, fmt.Errorf("duplicate tool %q", definition.Name)
		}
		prepared.tools[definition.Name] = tool
		prepared.modelTools = append(prepared.modelTools, definition)
	}

	switch config.Completion.Mode {
	case CompletionModeFinish, CompletionModeFinishOrText:
		prepared.finishName = strings.TrimSpace(config.Completion.FinishName)
		prepared.answerField = strings.TrimSpace(config.Completion.FinishAnswerField)
		if prepared.finishName == "" || prepared.answerField == "" {
			return preparedLoop{}, fmt.Errorf("Finish name and answer field are required")
		}
		for name := range prepared.tools {
			if strings.EqualFold(name, prepared.finishName) {
				return preparedLoop{}, fmt.Errorf("Finish tool %q conflicts with executable tool %q", prepared.finishName, name)
			}
		}
		prepared.modelTools = append(prepared.modelTools, finishModelTool(config.Completion))
		prepared.allowText = config.Completion.Mode == CompletionModeFinishOrText
	case CompletionModeText:
		prepared.allowText = true
	default:
		return preparedLoop{}, fmt.Errorf("unsupported completion mode %q", config.Completion.Mode)
	}
	prepared.execute = config.Execute
	if prepared.execute == nil {
		prepared.execute = func(ctx context.Context, tool core.Tool, arguments map[string]any) (core.ToolResult, error) {
			return tool.Execute(ctx, arguments)
		}
	}
	return prepared, nil
}

func (p preparedLoop) isFinish(name string) bool {
	return p.finishName != "" && strings.EqualFold(strings.TrimSpace(name), p.finishName)
}

func finishModelTool(config CompletionConfig) ModelTool {
	description := strings.TrimSpace(config.FinishDescription)
	if description == "" {
		description = "Call this tool when the task is complete."
	}
	return ModelTool{
		Name:        strings.TrimSpace(config.FinishName),
		Description: description,
		InputSchema: JSONSchema{
			"type": "object",
			"properties": map[string]any{
				strings.TrimSpace(config.FinishAnswerField): map[string]any{
					"type": "string", "description": "The final answer or result of the task",
				},
			},
			"required": []string{strings.TrimSpace(config.FinishAnswerField)},
		},
	}
}

func finishAnswer(call core.ToolCall, field string) (string, error) {
	value, exists := call.Arguments[field]
	if !exists {
		return "", fmt.Errorf("Finish call is missing required %q argument", field)
	}
	answer, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("Finish argument %q has type %T, want string", field, value)
	}
	answer = strings.TrimSpace(answer)
	if answer == "" {
		return "", fmt.Errorf("Finish argument %q must not be empty", field)
	}
	return answer, nil
}

func cloneModelTools(tools []ModelTool) []ModelTool {
	if tools == nil {
		return nil
	}
	cloned := make([]ModelTool, len(tools))
	for i, tool := range tools {
		cloned[i] = tool
		cloned[i].InputSchema = cloneJSONSchema(tool.InputSchema)
	}
	return cloned
}

func errorToolResultMessage(call core.ToolCall, text string) Message {
	return NewToolResultMessage(call.ID, call.Name, errorToolResult(text))
}

func errorToolResult(text string) core.ToolResult {
	return core.ToolResult{
		Data: text,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   text,
			core.ToolResultDisplayTextMeta: text,
			core.ToolResultIsErrorMeta:     true,
			core.ToolResultSyntheticMeta:   true,
		},
	}
}

func blockedToolResult(toolName string, err error) core.ToolResult {
	observation := BlockedToolObservation(toolName, strings.TrimPrefix(err.Error(), core.ErrToolBlocked.Error()+":"))
	return core.ToolResult{
		Data: observation.ModelText,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   observation.ModelText,
			core.ToolResultDisplayTextMeta: observation.DisplayText,
			core.ToolResultIsErrorMeta:     true,
			core.ToolResultSyntheticMeta:   true,
		},
		Annotations: map[string]any{core.ToolResultDetailsAnnotation: observation.Details},
	}
}

func addTokenInfo(result *LoopResult, usage *core.TokenInfo) {
	if usage == nil {
		return
	}
	if result.Usage == nil {
		result.Usage = &core.TokenInfo{}
	}
	result.Usage.PromptTokens += usage.PromptTokens
	result.Usage.CompletionTokens += usage.CompletionTokens
	result.Usage.TotalTokens += usage.TotalTokens
}

func canceledRunEvent(err error) RunFinishedEvent {
	return RunFinishedEvent{Status: RunStatusCanceled, StopReason: StopReasonCanceled, Err: err}
}

func isCancellation(ctx context.Context, err error) bool {
	if err != nil {
		return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
	}
	return ctx.Err() != nil
}

// ModelDiagnosticsMetadataKey stores ownership-safe provider diagnostics on the
// assistant message that produced them.
const ModelDiagnosticsMetadataKey = "agents.model_diagnostics"

func closeCanceledCalls(
	ctx context.Context,
	emitter *EventEmitter,
	runID string,
	turn int,
	calls []core.ToolCall,
	startIndex int,
	transcript *[]Message,
	err error,
) {
	for offset, call := range calls {
		call = cloneToolCall(call)
		message := errorToolResultMessage(call, fmt.Sprintf("tool call canceled: %v", err))
		*transcript = append(*transcript, message)
		emitter.Emit(ctx, ToolCallFinishedEvent{
			RunID: runID, Turn: turn, ToolIndex: startIndex + offset, Call: call,
			Outcome: ToolCallOutcomeRejected, Result: &message,
			Status: OperationStatusCanceled, Err: err,
		})
		emitter.Emit(ctx, MessageAddedEvent{RunID: runID, Turn: turn, Message: message})
	}
}
