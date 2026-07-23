package agents

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunLoop_ExecutesToolsSequentiallyAndFinishes(t *testing.T) {
	model := &loopTestModel{responses: []ModelResponse{
		{
			Message: Message{Role: RoleAssistant, Content: []core.ContentBlock{core.NewTextBlock("working")}, ToolCalls: []core.ToolCall{
				{ID: "call-1", Name: "first", Arguments: map[string]any{"value": "a"}},
				{ID: "call-2", Name: "second", Arguments: map[string]any{"value": "b"}},
			}},
			Usage:       &core.TokenInfo{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
			Diagnostics: map[string]any{"finish_reason": "tool_use", "nested": map[string]any{"value": "original"}},
		},
		{
			Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{
				{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}},
			}},
			Usage: &core.TokenInfo{PromptTokens: 4, CompletionTokens: 1, TotalTokens: 5},
		},
	}}
	var order []string
	first := &loopTestTool{name: "first", execute: func(_ context.Context, arguments map[string]any) (core.ToolResult, error) {
		order = append(order, "first:"+arguments["value"].(string))
		return core.ToolResult{Data: "first result"}, nil
	}}
	second := &loopTestTool{name: "second", execute: func(_ context.Context, arguments map[string]any) (core.ToolResult, error) {
		order = append(order, "second:"+arguments["value"].(string))
		return core.ToolResult{Data: "second result"}, nil
	}}
	var events []ExecutionEvent

	result, err := RunLoop(context.Background(), model, []core.Tool{first, second},
		[]Message{NewTextMessage(RoleUser, "do work")}, LoopConfig{
			RunID: "run-1", Task: "do work", MaxTurns: 3,
			Completion: FinishCompletion("finish when done"),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
				events = append(events, event)
			}),
		})
	require.NoError(t, err)
	assert.Equal(t, StopReasonFinish, result.StopReason)
	assert.Equal(t, "done", result.FinalAnswer)
	assert.Equal(t, 2, result.Turns)
	assert.Equal(t, 2, result.ToolCalls)
	assert.Equal(t, &core.TokenInfo{PromptTokens: 6, CompletionTokens: 2, TotalTokens: 8}, result.Usage)
	assert.Equal(t, []string{"first:a", "second:b"}, order)
	require.Len(t, model.requests, 2)
	require.Len(t, model.requests[1].Messages, 4)
	assert.Equal(t, RoleAssistant, model.requests[1].Messages[1].Role)
	assert.Equal(t, "call-1", model.requests[1].Messages[2].ToolResult.ToolCallID)
	assert.Equal(t, "call-2", model.requests[1].Messages[3].ToolResult.ToolCallID)
	assert.Equal(t, "first result", model.requests[1].Messages[2].ToolResult.Content[0].Text)
	assert.Equal(t, "tool_use", result.Messages[1].Metadata[ModelDiagnosticsMetadataKey].(map[string]any)["finish_reason"])
	model.responses[0].Diagnostics["nested"].(map[string]any)["value"] = "changed"
	assert.Equal(t, "original", result.Messages[1].Metadata[ModelDiagnosticsMetadataKey].(map[string]any)["nested"].(map[string]any)["value"])
	require.NoError(t, ValidateEventLifecycle(events))

	var payloadTypes []string
	for _, event := range events {
		payloadTypes = append(payloadTypes, fmt.Sprintf("%T", event.Payload))
	}
	assert.Equal(t, []string{
		"agents.RunStartedEvent", "agents.TurnStartedEvent", "agents.MessageAddedEvent",
		"agents.ToolCallProposedEvent", "agents.ToolCallProposedEvent",
		"agents.ToolExecutionStartedEvent", "agents.ToolCallFinishedEvent", "agents.MessageAddedEvent",
		"agents.ToolExecutionStartedEvent", "agents.ToolCallFinishedEvent", "agents.MessageAddedEvent",
		"agents.TurnFinishedEvent", "agents.TurnStartedEvent", "agents.MessageAddedEvent",
		"agents.ToolCallProposedEvent", "agents.ToolCallFinishedEvent", "agents.TurnFinishedEvent", "agents.RunFinishedEvent",
	}, payloadTypes)
}

func TestRunLoop_MalformedFinishBecomesToolErrorAndCanRecover(t *testing.T) {
	model := &loopTestModel{responses: []ModelResponse{
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{
			ID: "bad-finish", Name: "Finish", Arguments: map[string]any{"answer": 42},
		}}}},
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{
			ID: "good-finish", Name: "Finish", Arguments: map[string]any{"answer": "recovered"},
		}}}},
	}}
	var events []ExecutionEvent
	result, err := RunLoop(context.Background(), model, nil, nil, LoopConfig{
		MaxTurns: 3, Completion: FinishCompletion(""),
		Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
	})
	require.NoError(t, err)
	assert.Equal(t, "recovered", result.FinalAnswer)
	require.Len(t, model.requests, 2)
	require.Len(t, model.requests[1].Messages, 2)
	badResult := model.requests[1].Messages[1]
	require.NotNil(t, badResult.ToolResult)
	assert.True(t, badResult.ToolResult.IsError)
	assert.Contains(t, badResult.ToolResult.Content[0].Text, `has type int`)
	require.NoError(t, ValidateEventLifecycle(events))
	assert.Equal(t, ToolCallOutcomeRejected, events[4].Payload.(ToolCallFinishedEvent).Outcome)
}

func TestRunLoop_RejectsFinishCombinedWithOtherCalls(t *testing.T) {
	tool := &loopTestTool{name: "work"}
	model := &loopTestModel{responses: []ModelResponse{
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{
			{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "too early"}},
			{ID: "work-1", Name: "work"},
		}}},
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-2", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}},
	}}
	var events []ExecutionEvent
	result, err := RunLoop(context.Background(), model, []core.Tool{tool}, nil, LoopConfig{
		MaxTurns: 2, Completion: FinishCompletion(""),
		Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
	})
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalAnswer)
	assert.Equal(t, 1, tool.executeCalls)
	assert.Contains(t, model.requests[1].Messages[1].ToolResult.Content[0].Text, "only tool call")
	require.NoError(t, ValidateEventLifecycle(events))
}

func TestRunLoop_SupportsTextAndMaxTurnCompletion(t *testing.T) {
	t.Run("ordinary text", func(t *testing.T) {
		model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "final text")}}}
		result, err := RunLoop(context.Background(), model, nil, nil, LoopConfig{
			MaxTurns: 2, Completion: TextCompletion(),
		})
		require.NoError(t, err)
		assert.Equal(t, StopReasonText, result.StopReason)
		assert.Equal(t, "final text", result.FinalAnswer)
		assert.Equal(t, 1, result.Turns)
	})

	t.Run("Finish or text advertises Finish and accepts text", func(t *testing.T) {
		model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "hybrid final")}}}
		result, err := RunLoop(context.Background(), model, nil, nil, LoopConfig{
			MaxTurns: 1, Completion: FinishOrTextCompletion("finish explicitly"),
		})
		require.NoError(t, err)
		assert.Equal(t, StopReasonText, result.StopReason)
		require.Len(t, model.requests, 1)
		require.Len(t, model.requests[0].Tools, 1)
		assert.Equal(t, "Finish", model.requests[0].Tools[0].Name)
	})

	t.Run("Finish mode does not accept plain text", func(t *testing.T) {
		model := &loopTestModel{responses: []ModelResponse{
			{Message: NewTextMessage(RoleAssistant, "not finished")},
			{Message: NewTextMessage(RoleAssistant, "still not finished")},
		}}
		var events []ExecutionEvent
		result, err := RunLoop(context.Background(), model, nil, nil, LoopConfig{
			MaxTurns: 2, Completion: FinishCompletion(""),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
		})
		require.NoError(t, err)
		assert.Equal(t, StopReasonMaxTurns, result.StopReason)
		assert.Empty(t, result.FinalAnswer)
		require.NoError(t, ValidateEventLifecycle(events))
		terminal := events[len(events)-1].Payload.(RunFinishedEvent)
		assert.Equal(t, RunStatusStopped, terminal.Status)
		assert.Contains(t, terminal.Diagnostic, "max turns")
	})
}

func TestRunLoop_RejectsUnknownAndInvalidToolsWithoutExecution(t *testing.T) {
	validationErr := errors.New("value is required")
	tool := &loopTestTool{name: "known", validationErr: validationErr}
	model := &loopTestModel{responses: []ModelResponse{
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{
			{ID: "unknown-1", Name: "unknown"},
			{ID: "invalid-1", Name: "known"},
		}}},
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{
			ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"},
		}}}},
	}}
	var events []ExecutionEvent
	result, err := RunLoop(context.Background(), model, []core.Tool{tool}, nil, LoopConfig{
		MaxTurns: 2, Completion: FinishCompletion(""),
		Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
	})
	require.NoError(t, err)
	assert.Equal(t, "done", result.FinalAnswer)
	assert.Equal(t, 0, tool.executeCalls)
	require.Len(t, model.requests[1].Messages, 3)
	assert.Contains(t, model.requests[1].Messages[1].ToolResult.Content[0].Text, "unknown tool")
	assert.Contains(t, model.requests[1].Messages[2].ToolResult.Content[0].Text, "invalid arguments")
	require.NoError(t, ValidateEventLifecycle(events))
	for _, event := range events {
		_, started := event.Payload.(ToolExecutionStartedEvent)
		assert.False(t, started, "unknown and invalid tools must not start execution")
	}
}

func TestRunLoop_ConvertsToolExecutionFailuresAndUsesExecutorSeam(t *testing.T) {
	tests := []struct {
		name        string
		toolResult  core.ToolResult
		executeErr  error
		wantOutcome ToolCallOutcome
		wantStatus  OperationStatus
		wantText    string
	}{
		{name: "blocked", executeErr: &core.ToolBlockedError{Reason: "denied"}, wantOutcome: ToolCallOutcomeBlocked, wantStatus: OperationStatusBlocked, wantText: "blocked"},
		{name: "execution error", executeErr: errors.New("boom"), wantOutcome: ToolCallOutcomeExecuted, wantStatus: OperationStatusFailed, wantText: "execution failed"},
		{name: "error result", toolResult: core.ToolResult{Data: "bad result", Metadata: map[string]any{core.ToolResultIsErrorMeta: true}}, wantOutcome: ToolCallOutcomeExecuted, wantStatus: OperationStatusFailed, wantText: "bad result"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := &loopTestTool{name: "work"}
			model := &loopTestModel{responses: []ModelResponse{
				{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{ID: "work-1", Name: "work"}}}},
				{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}},
			}}
			var events []ExecutionEvent
			executorCalls := 0
			result, err := RunLoop(context.Background(), model, []core.Tool{tool}, nil, LoopConfig{
				MaxTurns: 2, Completion: FinishCompletion(""),
				Execute: func(context.Context, core.Tool, map[string]any) (core.ToolResult, error) {
					executorCalls++
					return tt.toolResult, tt.executeErr
				},
				Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
			})
			require.NoError(t, err)
			assert.Equal(t, "done", result.FinalAnswer)
			assert.Equal(t, 1, executorCalls)
			assert.Equal(t, 0, tool.executeCalls)
			assert.Contains(t, model.requests[1].Messages[1].ToolResult.Content[0].Text, tt.wantText)
			require.NoError(t, ValidateEventLifecycle(events))
			var terminal ToolCallFinishedEvent
			for _, event := range events {
				if payload, ok := event.Payload.(ToolCallFinishedEvent); ok && payload.Call.Name == "work" {
					terminal = payload
				}
			}
			assert.Equal(t, tt.wantOutcome, terminal.Outcome)
			assert.Equal(t, tt.wantStatus, terminal.Status)
		})
	}
}

func TestRunLoop_PropagatesProviderAndCancellationErrors(t *testing.T) {
	t.Run("provider failure is not reclassified by concurrent cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		providerErr := errors.New("provider failed")
		model := &loopTestModel{complete: func(ModelRequest) (ModelResponse, error) {
			cancel()
			return ModelResponse{}, providerErr
		}}
		result, err := RunLoop(ctx, model, nil, nil, LoopConfig{MaxTurns: 1, Completion: FinishCompletion("")})
		assert.ErrorIs(t, err, providerErr)
		assert.NotErrorIs(t, err, context.Canceled)
		assert.Equal(t, StopReasonError, result.StopReason)
	})

	t.Run("successful text response cannot hide cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		model := &loopTestModel{complete: func(ModelRequest) (ModelResponse, error) {
			cancel()
			return ModelResponse{Message: NewTextMessage(RoleAssistant, "text")}, nil
		}}
		result, err := RunLoop(ctx, model, nil, nil, LoopConfig{MaxTurns: 1, Completion: TextCompletion()})
		assert.ErrorIs(t, err, context.Canceled)
		assert.Equal(t, StopReasonCanceled, result.StopReason)
	})

	t.Run("provider failure", func(t *testing.T) {
		providerErr := errors.New("provider failed")
		model := &loopTestModel{errs: []error{providerErr}}
		var events []ExecutionEvent
		result, err := RunLoop(context.Background(), model, nil, nil, LoopConfig{
			MaxTurns: 1, Completion: FinishCompletion(""),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
		})
		assert.ErrorIs(t, err, providerErr)
		assert.Equal(t, StopReasonError, result.StopReason)
		require.NoError(t, ValidateEventLifecycle(events))
		terminal := events[len(events)-1].Payload.(RunFinishedEvent)
		assert.Equal(t, RunStatusFailed, terminal.Status)
	})

	t.Run("cancellation after rejected call skips later dispatch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		tool := &loopTestTool{name: "work"}
		model := &loopTestModel{responses: []ModelResponse{
			{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{
				{ID: "unknown-1", Name: "unknown"}, {ID: "work-1", Name: "work"},
			}}},
		}}
		var events []ExecutionEvent
		result, err := RunLoop(ctx, model, []core.Tool{tool}, nil, LoopConfig{
			MaxTurns: 1, Completion: FinishCompletion(""),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) {
				events = append(events, event)
				if payload, ok := event.Payload.(ToolCallFinishedEvent); ok && payload.Call.Name == "unknown" {
					cancel()
				}
			}),
		})
		assert.ErrorIs(t, err, context.Canceled)
		assert.Equal(t, StopReasonCanceled, result.StopReason)
		assert.Equal(t, 0, tool.executeCalls)
		require.NoError(t, ValidateEventLifecycle(events))
	})

	t.Run("cancellation during validation prevents dispatch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		tool := &loopTestTool{name: "work", validate: func(map[string]any) error {
			cancel()
			return nil
		}}
		model := &loopTestModel{responses: []ModelResponse{{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{ID: "work-1", Name: "work"}}}}}}
		var events []ExecutionEvent
		result, err := RunLoop(ctx, model, []core.Tool{tool}, nil, LoopConfig{
			MaxTurns: 1, Completion: FinishCompletion(""),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
		})
		assert.ErrorIs(t, err, context.Canceled)
		assert.Equal(t, StopReasonCanceled, result.StopReason)
		assert.Equal(t, 0, tool.executeCalls)
		require.NoError(t, ValidateEventLifecycle(events))
	})

	t.Run("canceled tool execution", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		tool := &loopTestTool{name: "wait", execute: func(context.Context, map[string]any) (core.ToolResult, error) {
			cancel()
			return core.ToolResult{}, context.Canceled
		}}
		later := &loopTestTool{name: "later"}
		model := &loopTestModel{responses: []ModelResponse{
			{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{
				{ID: "wait-1", Name: "wait"}, {ID: "later-1", Name: "later"},
			}}},
		}}
		var events []ExecutionEvent
		result, err := RunLoop(ctx, model, []core.Tool{tool, later}, nil, LoopConfig{
			MaxTurns: 2, Completion: FinishCompletion(""),
			Events: EventSinkFunc(func(_ context.Context, event ExecutionEvent) { events = append(events, event) }),
		})
		assert.ErrorIs(t, err, context.Canceled)
		assert.Equal(t, StopReasonCanceled, result.StopReason)
		assert.Equal(t, 0, later.executeCalls)
		require.NoError(t, ValidateEventLifecycle(events))
		terminal := events[len(events)-1].Payload.(RunFinishedEvent)
		assert.Equal(t, RunStatusCanceled, terminal.Status)
	})
}

func TestRunLoop_ValidatesConfigurationAndSnapshotsInputs(t *testing.T) {
	model := &loopTestModel{}
	_, err := RunLoop(context.Background(), nil, nil, nil, LoopConfig{MaxTurns: 1, Completion: TextCompletion()})
	require.EqualError(t, err, "model is required")
	_, err = RunLoop(context.Background(), model, nil, nil, LoopConfig{Completion: TextCompletion()})
	require.EqualError(t, err, "max turns must be greater than zero")
	_, err = RunLoop(context.Background(), model, nil, nil, LoopConfig{MaxTurns: 1})
	require.ErrorContains(t, err, "unsupported completion mode")

	conflict := &loopTestTool{name: "finish"}
	_, err = RunLoop(context.Background(), model, []core.Tool{conflict}, nil, LoopConfig{MaxTurns: 1, Completion: FinishCompletion("")})
	require.ErrorContains(t, err, "conflicts with executable tool")

	message := NewTextMessage(RoleUser, "original")
	model.responses = []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}
	result, err := RunLoop(context.Background(), model, []core.Tool{&loopTestTool{name: "unused"}}, []Message{message}, LoopConfig{
		MaxTurns: 1, Completion: TextCompletion(),
	})
	require.NoError(t, err)
	message.Content[0].Text = "changed"
	model.requests[0].Messages[0].Content[0].Text = "provider mutation"
	assert.Equal(t, "original", result.Messages[0].Content[0].Text)
}

type loopTestModel struct {
	mu        sync.Mutex
	responses []ModelResponse
	errs      []error
	requests  []ModelRequest
	complete  func(ModelRequest) (ModelResponse, error)
}

func (m *loopTestModel) Complete(_ context.Context, request ModelRequest) (ModelResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, ModelRequest{
		Messages: CloneMessages(request.Messages), Tools: cloneModelTools(request.Tools), Options: append([]core.GenerateOption(nil), request.Options...),
	})
	index := len(m.requests) - 1
	if m.complete != nil {
		return m.complete(request)
	}
	if index < len(m.errs) && m.errs[index] != nil {
		return ModelResponse{}, m.errs[index]
	}
	if index >= len(m.responses) {
		return ModelResponse{}, errors.New("unexpected model call")
	}
	return m.responses[index], nil
}

func (*loopTestModel) ModelID() string      { return "loop-model" }
func (*loopTestModel) ProviderName() string { return "loop-provider" }

type loopTestTool struct {
	name          string
	validationErr error
	validate      func(map[string]any) error
	execute       func(context.Context, map[string]any) (core.ToolResult, error)
	executeCalls  int
}

func (t *loopTestTool) Name() string        { return t.name }
func (t *loopTestTool) Description() string { return t.name + " tool" }
func (t *loopTestTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.Description(), InputSchema: t.InputSchema()}
}
func (*loopTestTool) CanHandle(context.Context, string) bool { return true }
func (t *loopTestTool) Execute(ctx context.Context, arguments map[string]any) (core.ToolResult, error) {
	t.executeCalls++
	if t.execute != nil {
		return t.execute(ctx, arguments)
	}
	return core.ToolResult{Data: "ok"}, nil
}
func (t *loopTestTool) Validate(arguments map[string]any) error {
	if t.validate != nil {
		return t.validate(arguments)
	}
	return t.validationErr
}
func (*loopTestTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{
		"value": {Type: "string"},
	}}
}
