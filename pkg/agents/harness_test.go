package agents

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHarness_WithSystemPromptOwnsStateAndContinues(t *testing.T) {
	model := &loopTestModel{responses: []ModelResponse{
		{Message: NewTextMessage(RoleAssistant, "first answer")},
		{Message: NewTextMessage(RoleAssistant, "continued answer")},
	}}
	harness, err := NewHarness(model, nil,
		WithSystemPrompt("You are careful."),
		WithHarnessCompletion(TextCompletion()),
	)
	require.NoError(t, err)

	result, err := harness.Prompt(context.Background(), Prompt{Text: "first question"})
	require.NoError(t, err)
	assert.Equal(t, "first answer", result.FinalAnswer)
	require.Len(t, model.requests, 1)
	assert.Equal(t, RoleSystem, model.requests[0].Messages[0].Role)
	assert.Equal(t, "You are careful.", model.requests[0].Messages[0].TextContent())
	assert.Equal(t, "first question", model.requests[0].Messages[1].TextContent())

	result, err = harness.Continue(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "continued answer", result.FinalAnswer)
	require.Len(t, model.requests, 2)
	assert.Equal(t, "first answer", model.requests[1].Messages[2].TextContent())

	messages := harness.Messages()
	require.Len(t, messages, 4)
	messages[0].Content[0].Text = "mutated"
	assert.Equal(t, "You are careful.", harness.Messages()[0].TextContent())
	assert.ErrorIs(t, harness.Cancel(), ErrHarnessIdle)
}

func TestHarness_WithInstructionsSnapshotsSignatureAndOptimizedDemos(t *testing.T) {
	module := &instructionTestModule{
		signature: core.NewSignature(
			[]core.InputField{{Field: core.NewField("task", core.WithDescription("work to perform"))}},
			[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("final result"))}},
		).WithInstruction("Act as an optimized coding agent."),
		demos: []core.Example{{
			Inputs: map[string]any{"task": "demo task"}, Outputs: map[string]any{"answer": "demo answer"},
		}},
	}
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil,
		WithInstructions(module), WithHarnessCompletion(TextCompletion()),
	)
	require.NoError(t, err)
	assert.Equal(t, int32(0), module.processCalls.Load(), "constructors must not execute modules")

	module.signature.Instruction = "changed"
	module.demos[0].Inputs["task"] = "changed demo"
	_, err = harness.Prompt(context.Background(), Prompt{Text: "real task"})
	require.NoError(t, err)
	assert.Equal(t, int32(0), module.processCalls.Load())
	require.Len(t, model.requests, 1)
	messages := model.requests[0].Messages
	require.Len(t, messages, 4)
	assert.Contains(t, messages[0].TextContent(), "Act as an optimized coding agent.")
	assert.Contains(t, messages[0].TextContent(), "work to perform")
	assert.Equal(t, "task: demo task", messages[1].TextContent())
	assert.Equal(t, "answer: demo answer", messages[2].TextContent())
	assert.Equal(t, "real task", messages[3].TextContent())
}

func TestHarness_WithInstructionsPreservesMultimodalDemosAndPrefixes(t *testing.T) {
	image := core.NewImageBlock([]byte{1, 2}, "image/png")
	audio := core.NewAudioBlock([]byte{3, 4}, "audio/wav")
	module := &instructionTestModule{
		signature: core.NewSignature(
			[]core.InputField{
				{Field: core.NewImageField("image")},
				{Field: core.NewAudioField("audio")},
			},
			[]core.OutputField{
				{Field: core.NewField("answer", core.WithCustomPrefix("RESULT>"))},
				{Field: core.NewField("notes", core.WithNoPrefix())},
			},
		).WithInstruction("Inspect media."),
		demos: []core.Example{{
			Inputs:  map[string]any{"image": image, "audio": audio},
			Outputs: map[string]any{"answer": "recognized", "notes": "plain notes"},
		}},
	}
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil, WithInstructions(module), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	image.Data[0] = 9
	audio.Data[0] = 9
	_, err = harness.Prompt(context.Background(), Prompt{Text: "real task"})
	require.NoError(t, err)

	messages := model.requests[0].Messages
	assert.Contains(t, messages[0].TextContent(), `answer [text] prefix "RESULT>"`)
	assert.Contains(t, messages[0].TextContent(), "notes [text] without a prefix")
	require.Len(t, messages[1].Content, 2)
	assert.Equal(t, core.FieldTypeImage, messages[1].Content[0].Type)
	assert.Equal(t, []byte{1, 2}, messages[1].Content[0].Data)
	assert.Equal(t, "image", messages[1].Content[0].Metadata["dspy_field"])
	assert.Equal(t, core.FieldTypeAudio, messages[1].Content[1].Type)
	assert.Equal(t, []byte{3, 4}, messages[1].Content[1].Data)
	assert.Equal(t, "RESULT> recognized\nplain notes", messages[2].TextContent())
}

func TestHarness_WithInstructionsAcceptsSignatureAndSingleModuleProgram(t *testing.T) {
	signature := core.NewSignature(nil, nil).WithInstruction("signature policy")
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil,
		WithInstructions(signature), WithHarnessCompletion(TextCompletion()),
	)
	require.NoError(t, err)
	_, err = harness.Prompt(context.Background(), Prompt{Text: "task"})
	require.NoError(t, err)
	assert.Equal(t, "signature policy", model.requests[0].Messages[0].TextContent())

	module := &instructionTestModule{signature: signature}
	program := core.NewProgram(map[string]core.Module{"policy": module}, nil)
	_, err = NewHarness(&loopTestModel{}, nil, WithInstructions(program), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)

	executableProgram := core.NewProgram(map[string]core.Module{"policy": module}, func(context.Context, map[string]any) (map[string]any, error) {
		return nil, nil
	})
	_, err = NewHarness(&loopTestModel{}, nil, WithInstructions(executableProgram), WithHarnessCompletion(TextCompletion()))
	require.ErrorContains(t, err, "executable Forward behavior")

	program.Modules["other"] = &instructionTestModule{signature: signature}
	_, err = NewHarness(&loopTestModel{}, nil, WithInstructions(program), WithHarnessCompletion(TextCompletion()))
	require.ErrorContains(t, err, "exactly one module")
}

func TestHarness_WithInstructionsAcceptsCompatiblePredictAndRejectsRuntimeBehavior(t *testing.T) {
	signature, err := core.ParseSignature("task -> answer")
	require.NoError(t, err)
	signature = signature.WithInstruction("Predict policy")
	predict := modules.NewPredict(signature)
	predict.Demos = []core.Example{{Inputs: map[string]any{"task": "demo"}, Outputs: map[string]any{"answer": "result"}}}
	model := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	harness, err := NewHarness(model, nil, WithInstructions(predict), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	_, err = harness.Prompt(context.Background(), Prompt{Text: "real task"})
	require.NoError(t, err)
	assert.Contains(t, model.requests[0].Messages[0].TextContent(), "task [text]")
	assert.Contains(t, model.requests[0].Messages[0].TextContent(), "answer [text]")

	zeroValuePredict := modules.NewPredict(core.Signature{
		Inputs:  []core.InputField{{Field: core.Field{Name: "task"}}},
		Outputs: []core.OutputField{{Field: core.Field{Name: "answer"}}},
	}.WithInstruction("Literal policy"))
	zeroModel := &loopTestModel{responses: []ModelResponse{{Message: NewTextMessage(RoleAssistant, "done")}}}
	zeroHarness, err := NewHarness(zeroModel, nil, WithInstructions(zeroValuePredict), WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	_, err = zeroHarness.Prompt(context.Background(), Prompt{Text: "real task"})
	require.NoError(t, err)
	assert.Contains(t, zeroModel.requests[0].Messages[0].TextContent(), "task [text]")
	assert.Contains(t, zeroModel.requests[0].Messages[0].TextContent(), "answer [text]")

	predict.WithDefaultOptions(core.WithGenerateOptions(core.WithMaxTokens(10)))
	_, err = NewHarness(&loopTestModel{}, nil, WithInstructions(predict), WithHarnessCompletion(TextCompletion()))
	require.ErrorContains(t, err, "default generation options")

	structured := modules.NewPredict(signature).WithStructuredOutput()
	_, err = NewHarness(&loopTestModel{}, nil, WithInstructions(structured), WithHarnessCompletion(TextCompletion()))
	require.ErrorContains(t, err, "output/interceptor behavior")
}

func TestHarness_RejectsUnrepresentableModuleAndConflictingInstructions(t *testing.T) {
	module := &nonInstructionModule{signature: core.NewSignature(nil, nil).WithInstruction("hidden behavior")}
	_, err := NewHarness(&loopTestModel{}, nil,
		WithInstructions(module), WithHarnessCompletion(TextCompletion()),
	)
	require.ErrorContains(t, err, "cannot be represented as instruction artifacts")

	_, err = NewHarness(&loopTestModel{}, nil,
		WithSystemPrompt("one"),
		WithInstructions(core.NewSignature(nil, nil).WithInstruction("two")),
		WithHarnessCompletion(TextCompletion()),
	)
	require.ErrorContains(t, err, "already configured")

	var typedNil *instructionTestModule
	_, err = NewHarness(&loopTestModel{}, nil,
		WithInstructions(typedNil), WithHarnessCompletion(TextCompletion()),
	)
	require.ErrorContains(t, err, "instruction provider is required")
}

func TestHarness_RejectsOverlapAndCancelsActiveRun(t *testing.T) {
	model := newBlockingHarnessModel()
	harness, err := NewHarness(model, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)

	resultCh := make(chan error, 1)
	go func() {
		_, runErr := harness.Prompt(context.Background(), Prompt{Text: "block"})
		resultCh <- runErr
	}()
	<-model.started

	_, err = harness.Prompt(context.Background(), Prompt{Text: "overlap"})
	assert.ErrorIs(t, err, ErrHarnessRunning)
	_, err = harness.Continue(context.Background())
	assert.ErrorIs(t, err, ErrHarnessRunning)
	require.NoError(t, harness.Cancel())
	assert.ErrorIs(t, <-resultCh, context.Canceled)
	assert.ErrorIs(t, harness.Cancel(), ErrHarnessIdle)

	messages := harness.Messages()
	require.Len(t, messages, 1)
	assert.Equal(t, "block", messages[0].TextContent())
}

func TestHarness_FollowUpClosesPriorFinishCall(t *testing.T) {
	model := &loopTestModel{responses: []ModelResponse{
		{Message: Message{Role: RoleAssistant, ToolCalls: []core.ToolCall{{
			ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "first"},
		}}}},
		{Message: NewTextMessage(RoleAssistant, "second")},
	}}
	harness, err := NewHarness(model, nil)
	require.NoError(t, err)
	_, err = harness.Prompt(context.Background(), Prompt{Text: "first task"})
	require.NoError(t, err)
	_, err = harness.Prompt(context.Background(), Prompt{Text: "follow up"})
	require.NoError(t, err)

	require.Len(t, model.requests, 2)
	messages := model.requests[1].Messages
	require.Len(t, messages, 4)
	require.NotNil(t, messages[2].ToolResult)
	assert.Equal(t, "finish-1", messages[2].ToolResult.ToolCallID)
	assert.True(t, messages[2].ToolResult.Synthetic)
	assert.Equal(t, "follow up", messages[3].TextContent())
}

func TestHarness_ContinueRequiresPrompt(t *testing.T) {
	harness, err := NewHarness(&loopTestModel{}, nil, WithHarnessCompletion(TextCompletion()))
	require.NoError(t, err)
	_, err = harness.Continue(context.Background())
	assert.ErrorIs(t, err, ErrNothingToContinue)
	_, err = harness.Prompt(context.Background(), Prompt{})
	require.EqualError(t, err, "prompt content is required")
}

type instructionTestModule struct {
	signature    core.Signature
	demos        []core.Example
	processCalls atomic.Int32
}

func (m *instructionTestModule) Process(context.Context, map[string]any, ...core.Option) (map[string]any, error) {
	m.processCalls.Add(1)
	return nil, errors.New("instruction module must not execute")
}
func (m *instructionTestModule) GetSignature() core.Signature          { return m.signature }
func (m *instructionTestModule) SetSignature(signature core.Signature) { m.signature = signature }
func (*instructionTestModule) SetLLM(core.LLM)                         {}
func (m *instructionTestModule) Clone() core.Module {
	return &instructionTestModule{signature: m.signature, demos: cloneExamples(m.demos)}
}
func (*instructionTestModule) GetDisplayName() string { return "instruction-test" }
func (*instructionTestModule) GetModuleType() string  { return "instruction-test" }
func (m *instructionTestModule) GetDemos() []core.Example {
	return cloneExamples(m.demos)
}
func (m *instructionTestModule) InstructionArtifacts() (core.Signature, []core.Example, error) {
	return m.signature, m.GetDemos(), nil
}

type nonInstructionModule struct {
	signature core.Signature
}

func (*nonInstructionModule) Process(context.Context, map[string]any, ...core.Option) (map[string]any, error) {
	return nil, errors.New("not representable")
}
func (m *nonInstructionModule) GetSignature() core.Signature          { return m.signature }
func (m *nonInstructionModule) SetSignature(signature core.Signature) { m.signature = signature }
func (*nonInstructionModule) SetLLM(core.LLM)                         {}
func (m *nonInstructionModule) Clone() core.Module {
	return &nonInstructionModule{signature: m.signature}
}
func (*nonInstructionModule) GetDisplayName() string { return "non-instruction" }
func (*nonInstructionModule) GetModuleType() string  { return "non-instruction" }

type blockingHarnessModel struct {
	started chan struct{}
	once    sync.Once
}

func newBlockingHarnessModel() *blockingHarnessModel {
	return &blockingHarnessModel{started: make(chan struct{})}
}

func (m *blockingHarnessModel) Complete(ctx context.Context, _ ModelRequest) (ModelResponse, error) {
	m.once.Do(func() { close(m.started) })
	<-ctx.Done()
	return ModelResponse{}, ctx.Err()
}
func (*blockingHarnessModel) ModelID() string      { return "blocking" }
func (*blockingHarnessModel) ProviderName() string { return "test" }
