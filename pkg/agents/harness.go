package agents

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

var (
	// ErrHarnessRunning indicates that Prompt or Continue was called while a run
	// was already active.
	ErrHarnessRunning = errors.New("agent harness already has an active run")
	// ErrHarnessIdle indicates that Cancel was called without an active run.
	ErrHarnessIdle = errors.New("agent harness has no active run")
	// ErrNothingToContinue indicates that Continue was called before Prompt.
	ErrNothingToContinue = errors.New("agent harness has no prompt to continue")
)

// Prompt is one typed user input to a Harness. Text is appended before any
// additional multimodal content blocks.
type Prompt struct {
	RunID    string
	Text     string
	Content  []core.ContentBlock
	Fields   map[string]any
	Metadata map[string]any
}

func (p Prompt) message(renderer promptFieldRenderer) (Message, error) {
	if len(p.Fields) > 0 {
		if p.Text != "" || len(p.Content) > 0 {
			return Message{}, fmt.Errorf("prompt fields cannot be combined with text or content")
		}
		if renderer == nil {
			return Message{}, fmt.Errorf("prompt fields require signature-backed instructions")
		}
		message, err := renderer.PromptFieldMessage(cloneAnyMap(p.Fields))
		if err != nil {
			return Message{}, err
		}
		message.Metadata = cloneAnyMap(p.Metadata)
		return message, nil
	}
	content := make([]core.ContentBlock, 0, 1+len(p.Content))
	if p.Text != "" {
		content = append(content, core.NewTextBlock(p.Text))
	}
	content = append(content, cloneContentBlocks(p.Content)...)
	if len(content) == 0 {
		return Message{}, fmt.Errorf("prompt content is required")
	}
	return Message{Role: RoleUser, Content: content, Metadata: cloneAnyMap(p.Metadata)}, nil
}

// HarnessOption configures a Harness during construction.
type HarnessOption func(*harnessConfig) error

type harnessConfig struct {
	instructions InstructionSource
	loop         LoopConfig
}

// WithSystemPrompt configures static system instructions.
func WithSystemPrompt(text string) HarnessOption {
	return func(config *harnessConfig) error {
		if config.instructions != nil {
			return fmt.Errorf("instructions are already configured")
		}
		if strings.TrimSpace(text) == "" {
			return fmt.Errorf("system prompt must not be empty")
		}
		config.instructions = staticInstructions(text)
		return nil
	}
}

// WithInstructions configures instructions from a DSPy signature, compatible
// optimized module, or single-module program.
func WithInstructions(provider InstructionProvider) HarnessOption {
	return func(config *harnessConfig) error {
		if config.instructions != nil {
			return fmt.Errorf("instructions are already configured")
		}
		source, err := instructionSourceFromProvider(provider)
		if err != nil {
			return err
		}
		config.instructions = source
		return nil
	}
}

// WithInstructionSource installs the lower-level custom instruction seam.
func WithInstructionSource(source InstructionSource) HarnessOption {
	return func(config *harnessConfig) error {
		if config.instructions != nil {
			return fmt.Errorf("instructions are already configured")
		}
		if isNilInstructionValue(source) {
			return fmt.Errorf("instruction source is required")
		}
		config.instructions = source
		return nil
	}
}

// WithHarnessMaxTurns sets the maximum model turns for each run.
func WithHarnessMaxTurns(maxTurns int) HarnessOption {
	return func(config *harnessConfig) error {
		if maxTurns <= 0 {
			return fmt.Errorf("max turns must be greater than zero")
		}
		config.loop.MaxTurns = maxTurns
		return nil
	}
}

// WithHarnessCompletion sets the completion behavior for each run.
func WithHarnessCompletion(completion CompletionConfig) HarnessOption {
	return func(config *harnessConfig) error {
		config.loop.Completion = completion
		return nil
	}
}

// WithHarnessModelOptions snapshots provider generation options for each run.
func WithHarnessModelOptions(options ...core.GenerateOption) HarnessOption {
	return func(config *harnessConfig) error {
		config.loop.Options = append([]core.GenerateOption(nil), options...)
		return nil
	}
}

// WithHarnessEventSink sets the typed event consumer used by each run.
func WithHarnessEventSink(sink EventSink) HarnessOption {
	return func(config *harnessConfig) error {
		config.loop.Events = sink
		return nil
	}
}

// WithHarnessToolExecutor sets the optional tool execution/interceptor seam.
func WithHarnessToolExecutor(executor ToolExecutor) HarnessOption {
	return func(config *harnessConfig) error {
		config.loop.Execute = executor
		return nil
	}
}

// Harness owns a reusable transcript and serializes stateful RunLoop calls.
type Harness struct {
	mu           sync.Mutex
	model        Model
	tools        []core.Tool
	config       LoopConfig
	promptFields promptFieldRenderer
	baseMessages []Message
	messages     []Message
	hasPrompt    bool
	active       bool
	cancel       context.CancelFunc
}

// NewHarness creates a reusable stateful agent harness. Instruction artifacts
// are snapshotted without invoking core.Module.Process or core.Program.Execute.
func NewHarness(model Model, tools []core.Tool, options ...HarnessOption) (*Harness, error) {
	config := harnessConfig{loop: LoopConfig{
		MaxTurns:   40,
		Completion: FinishOrTextCompletion("Call this tool when the task is complete."),
	}}
	for _, option := range options {
		if option == nil {
			continue
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}
	if _, err := prepareLoop(model, tools, config.loop); err != nil {
		return nil, err
	}

	var initial []Message
	if config.instructions != nil {
		messages, err := config.instructions.InstructionMessages()
		if err != nil {
			return nil, fmt.Errorf("resolve instructions: %w", err)
		}
		initial = CloneMessages(messages)
	}
	renderer, _ := config.instructions.(promptFieldRenderer)
	return &Harness{
		model: model, tools: append([]core.Tool(nil), tools...), config: cloneLoopConfig(config.loop),
		promptFields: renderer, baseMessages: CloneMessages(initial), messages: initial,
	}, nil
}

// Prompt appends one user message and runs until a completion condition.
func (h *Harness) Prompt(ctx context.Context, prompt Prompt) (LoopResult, error) {
	if h == nil {
		return LoopResult{}, fmt.Errorf("agent harness is nil")
	}
	return h.run(ctx, &prompt, prompt.RunID, false)
}

// Continue resumes the current transcript without appending a user message.
func (h *Harness) Continue(ctx context.Context) (LoopResult, error) {
	if h == nil {
		return LoopResult{}, fmt.Errorf("agent harness is nil")
	}
	return h.run(ctx, nil, "", true)
}

func (h *Harness) run(ctx context.Context, prompt *Prompt, runID string, continuing bool) (LoopResult, error) {
	h.mu.Lock()
	if h.active {
		h.mu.Unlock()
		return LoopResult{}, ErrHarnessRunning
	}
	if continuing && !h.hasPrompt {
		h.mu.Unlock()
		return LoopResult{}, ErrNothingToContinue
	}
	transcript := h.messages
	completion := h.config.Completion
	renderer := h.promptFields
	runConfig := cloneLoopConfig(h.config)
	runConfig.RunID = runID
	model := h.model
	tools := append([]core.Tool(nil), h.tools...)
	runCtx, cancel := context.WithCancel(ctx)
	h.active = true
	h.cancel = cancel
	h.mu.Unlock()

	messages := closeHarnessCompletion(CloneMessages(transcript), completion)
	if prompt != nil {
		message, err := prompt.message(renderer)
		if err != nil {
			cancel()
			h.mu.Lock()
			h.active = false
			h.cancel = nil
			h.mu.Unlock()
			return LoopResult{}, err
		}
		messages = append(messages, message)
		h.mu.Lock()
		h.hasPrompt = true
		h.mu.Unlock()
	}

	result, runErr := RunLoop(runCtx, model, tools, messages, runConfig)
	cancel()

	persisted := result.Messages
	if persisted == nil {
		persisted = messages
	}
	persisted = CloneMessages(persisted)
	h.mu.Lock()
	h.messages = persisted
	h.active = false
	h.cancel = nil
	h.mu.Unlock()
	return result, runErr
}

// Messages returns an ownership-safe transcript snapshot.
func (h *Harness) Messages() []Message {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	messages := h.messages
	h.mu.Unlock()
	return CloneMessages(messages)
}

// Cancel cancels the active run.
func (h *Harness) Cancel() error {
	if h == nil {
		return fmt.Errorf("agent harness is nil")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.active || h.cancel == nil {
		return ErrHarnessIdle
	}
	h.cancel()
	return nil
}

func closeHarnessCompletion(messages []Message, completion CompletionConfig) []Message {
	if len(messages) == 0 || (completion.Mode != CompletionModeFinish && completion.Mode != CompletionModeFinishOrText) {
		return messages
	}
	last := messages[len(messages)-1]
	if last.Role != RoleAssistant {
		return messages
	}
	finishName := strings.TrimSpace(completion.FinishName)
	for _, call := range last.ToolCalls {
		if !strings.EqualFold(strings.TrimSpace(call.Name), finishName) {
			continue
		}
		result := core.ToolResult{
			Data: "Completion acknowledged.",
			Metadata: map[string]any{
				core.ToolResultSyntheticMeta: true,
			},
		}
		messages = append(messages, NewToolResultMessage(call.ID, call.Name, result))
	}
	return messages
}

func (h *Harness) replaceInstructions(source InstructionSource) error {
	if h == nil {
		return fmt.Errorf("agent harness is nil")
	}
	messages, err := source.InstructionMessages()
	if err != nil {
		return err
	}
	renderer, _ := source.(promptFieldRenderer)
	baseMessages := CloneMessages(messages)
	currentMessages := CloneMessages(messages)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.active || h.hasPrompt {
		return fmt.Errorf("cannot replace instructions after harness execution has started")
	}
	h.baseMessages = baseMessages
	h.messages = currentMessages
	h.promptFields = renderer
	return nil
}

func (h *Harness) replaceModel(model Model) error {
	if h == nil {
		return fmt.Errorf("agent harness is nil")
	}
	if model == nil {
		return fmt.Errorf("model is required")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.active {
		return ErrHarnessRunning
	}
	h.model = model
	return nil
}

func (h *Harness) fresh() *Harness {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	model := h.model
	tools := append([]core.Tool(nil), h.tools...)
	config := cloneLoopConfig(h.config)
	renderer := h.promptFields
	base := h.baseMessages
	h.mu.Unlock()
	base = CloneMessages(base)
	return &Harness{
		model: model, tools: tools, config: config,
		promptFields: renderer, baseMessages: CloneMessages(base), messages: base,
	}
}

func cloneLoopConfig(config LoopConfig) LoopConfig {
	cloned := config
	cloned.Options = append([]core.GenerateOption(nil), config.Options...)
	return cloned
}
