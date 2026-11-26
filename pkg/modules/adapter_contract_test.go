package modules

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/adapters"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The cases in this file are the adversarial replies collected while reviewing
// the text parser: values it mangled (a lost minus sign, a dropped asterisk, a
// stripped fence) and replies it accepted with a field left nil. They are
// re-targeted at the adapters. Every case checks the output contract: a call
// returns every declared field, or a typed output error; it never succeeds
// with a field missing.

// scriptedLLM is a text-only model that answers each call with the next reply,
// repeating the last one. It streams in five-byte chunks, so a tag or a JSON
// string is always split across chunks.
type scriptedLLM struct {
	core.LLM
	mu      sync.Mutex
	replies []string
	calls   int
}

func (m *scriptedLLM) ModelID() string      { return "scripted" }
func (m *scriptedLLM) ProviderName() string { return "test" }
func (m *scriptedLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *scriptedLLM) next() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	reply := m.replies[min(m.calls, len(m.replies)-1)]
	m.calls++
	return reply
}

func (m *scriptedLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: m.next()}, nil
}

func (m *scriptedLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	reply := m.next()
	chunks := make(chan core.StreamChunk, len(reply)/5+2)
	for len(reply) > 0 {
		n := min(5, len(reply))
		chunks <- core.StreamChunk{Content: reply[:n]}
		reply = reply[n:]
	}
	chunks <- core.StreamChunk{Done: true}
	close(chunks)
	return &core.StreamResponse{ChunkChannel: chunks, Cancel: func() {}}, nil
}

// contractMode is one way to run a Predict call through an adapter.
type contractMode struct {
	name      string
	adapter   func() core.Adapter
	streaming bool
}

func xmlOnly() core.Adapter { return adapters.NewXMLAdapter().WithJSONFallback(false) }
func xmlFallback() core.Adapter {
	return adapters.NewXMLAdapter()
}
func jsonOnly() core.Adapter { return adapters.NewJSONAdapter() }

var contractModes = []contractMode{
	{name: "xml", adapter: xmlOnly},
	{name: "xml streaming", adapter: xmlOnly, streaming: true},
	{name: "xml with JSON fallback", adapter: xmlFallback},
	{name: "xml with JSON fallback streaming", adapter: xmlFallback, streaming: true},
	{name: "json", adapter: jsonOnly},
	{name: "json streaming", adapter: jsonOnly, streaming: true},
}

// runContract sends reply through Predict with the mode's adapter and checks
// the output contract. It returns the outputs and error for further checks.
func runContract(t *testing.T, mode contractMode, signature core.Signature, replies ...string) (map[string]any, error) {
	t.Helper()
	llm := &scriptedLLM{replies: replies}
	predict := NewPredict(signature).WithAdapter(mode.adapter())
	predict.SetLLM(llm)

	var opts []core.Option
	streamed := map[string]string{}
	if mode.streaming {
		opts = append(opts, core.WithStreamHandler(func(chunk core.StreamChunk) error {
			if chunk.Content != "" {
				streamed[chunk.Field] += chunk.Content
			}
			return nil
		}))
	}
	outputs, err := predict.Process(context.Background(), map[string]any{"question": "When was Masakazu Katsura born?"}, opts...)
	if err != nil {
		assert.Nil(t, outputs, "a failed call returns no outputs")
		assert.True(t, dspyerrors.IsOutputError(err), "an unreadable reply is an output error, got %T: %v", err, err)
		return outputs, err
	}
	assert.Empty(t, signature.MissingOutputs(outputs), "a successful call has every declared output")
	for name := range outputs {
		assert.True(t, hasOutput(signature, name), "undeclared output %q", name)
	}
	if mode.streaming {
		// The handler received each field's text, and nothing else.
		want := map[string]string{}
		for name, value := range outputs {
			if text, _ := value.(string); text != "" {
				want[name] = text
			}
		}
		assert.Equal(t, want, streamed, "the streamed field text is the returned field text")
	}
	return outputs, nil
}

func hasOutput(signature core.Signature, name string) bool {
	for _, field := range signature.Outputs {
		if field.Name == name {
			return true
		}
	}
	return false
}

// renderReply returns the reply a model would write for outputs by following
// the adapter's own format: the assistant turn the adapter renders for a
// demonstration with those outputs. Cases built with it show that what the
// adapter shows the model, it can read back unchanged. They do not show that a
// reply written another way is read correctly; the hand-written replies in
// this file cover that.
func renderReply(t *testing.T, adapter core.Adapter, signature core.Signature, outputs map[string]any) string {
	t.Helper()
	inputs := map[string]any{"question": "q"}
	messages, err := adapter.Format(signature, []core.Example{{Inputs: inputs, Outputs: outputs}}, inputs)
	require.NoError(t, err)
	for _, message := range messages {
		if message.Role != "assistant" {
			continue
		}
		var text strings.Builder
		for _, block := range message.Content {
			text.WriteString(block.Text)
		}
		return text.String()
	}
	require.FailNow(t, "the adapter rendered no assistant turn for the demonstration")
	return ""
}

// adversarialValues are the field values the text parser changed or dropped.
var adversarialValues = []string{
	"-42",
	"-0.5",
	"*ptr",
	"**-42**",
	"- Paris",
	"-42\n-43",
	"*first\n*second",
	"-1 step\n-2 step",
	"```\n-42\n```",
	"```markdown\n-0.5\n```",
	"```json\n{\"answer\":-42}\n```",
	"answer:",
	"answer: 42",
	"answer: answer: 42",
	"The answer: 42",
	"reasoning:\nanswer:",
	"I cannot determine this.\nanswer:",
	"**answer:** Paris",
	"<response><other>42</other></response>",
	"if a < b && c > d { return \"</done>\" }",
	"",
}

func TestAdapterContract_AdversarialValuesRoundTripExactly(t *testing.T) {
	single := qaSignature("answer")
	for _, mode := range contractModes {
		for _, value := range adversarialValues {
			t.Run(fmt.Sprintf("%s/%q", mode.name, value), func(t *testing.T) {
				want := map[string]any{"answer": value}
				reply := renderReply(t, mode.adapter(), single, want)

				got, err := runContract(t, mode, single, reply)
				require.NoError(t, err)
				assert.Equal(t, want, got)
			})
		}
	}
}

// TestAdapterContract_LabelBoundaryValuesRoundTripExactly re-targets the 120
// label-boundary cases: a first field whose value looks like markdown or a
// number, followed by a second field written with a repeated, bold, bulleted
// or later label. Under an adapter those labels are ordinary text, with the
// standard prefixes and with custom ones.
func TestAdapterContract_LabelBoundaryValuesRoundTripExactly(t *testing.T) {
	bodies := []string{"-1 step\n-2 step", "*first\n*second", "```\n-42\n```", " \n-42", "\n\n", "**-42**"}
	for _, custom := range []bool{false, true} {
		signature := qaSignature("reasoning", "answer")
		first, last := "reasoning:", "answer:"
		if custom {
			first, last = "R>", "A>"
			signature.Outputs[0].Prefix = first
			signature.Outputs[1].Prefix = last
		}
		tails := []string{last + " -7", last + "\n-7", "**" + last + "** -7", "-" + last + " -7", first + "\n" + last + " -7"}
		for _, mode := range contractModes {
			for bodyIndex, body := range bodies {
				for tailIndex, tail := range tails {
					name := fmt.Sprintf("custom=%t/%s/body=%d/tail=%d", custom, mode.name, bodyIndex, tailIndex)
					t.Run(name, func(t *testing.T) {
						want := map[string]any{"reasoning": body, "answer": tail}
						reply := renderReply(t, mode.adapter(), signature, want)

						got, err := runContract(t, mode, signature, reply)
						require.NoError(t, err)
						assert.Equal(t, want, got)
					})
				}
			}
		}
	}
}

// legacyReplies are the replies of the original cases, written for the text
// parser's label format. None of them follows an adapter's format.
var legacyReplies = []string{
	// Single-output replies.
	"answer: 1962",
	"answer:\n1962",
	"1962",
	"Paris\nis the capital.",
	"The answer: 42",
	"- Paris",
	"Sure, here it is.\nanswer: Paris",
	"answer: answer: 42",
	"Answer: 42",
	"**answer:** Paris",
	"",
	"  \n \n",
	"Final Answer: 42",
	"-42",
	"answer: -42",
	"-0.5",
	"*ptr",
	"answer:",
	"reasoning:\nanswer:",
	"answer: ",
	"I cannot determine this.\nanswer:",
	"```\n-42\n```",
	"answer: ```\n-42\n```",
	"```markdown\n-0.5\n```",
	"-42\n-43",
	"answer: -42\n-43",
	"*first\n*second",
	"\n \t-42",
	"**-42**",
	"Sure.\n**answer:**",
	"```\nanswer:\n```",
	"```\n-42\n```\nanswer: 99",
	// Multi-output replies.
	"reasoning: Born in 1962.\nanswer: 1962",
	"answer: 1962\nreasoning: Born in 1962.",
	"reasoning: answer: maybe 1961, check.\nanswer: 1962",
	"Born in 1962.\nanswer: 1962",
	"Computed 6 * 7.\nanswer: 42",
	"preamble\nanswer:",
	// Replies in the old XML and ReAct formats.
	"<response><answer>42",
	"<response><other>42</other></response>",
	"thought: The previous tool result gives 579.\naction: <action><tool_name>Finish</tool_name></action>",
	"thought:\nThe previous tool result gives the sum, 579.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\n\nanswer:",
	"thought:\nThe prior tool result gives 77 + 88 = 165.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\nanswer:\n",
}

func TestAdapterContract_RepliesInAnotherFormatAreErrorsNotEmptyFields(t *testing.T) {
	signatures := map[string]core.Signature{
		"one output":  qaSignature("answer"),
		"two outputs": qaSignature("reasoning", "answer"),
	}
	for signatureName, signature := range signatures {
		for _, mode := range contractModes {
			for _, reply := range legacyReplies {
				t.Run(fmt.Sprintf("%s/%s/%q", signatureName, mode.name, reply), func(t *testing.T) {
					outputs, err := runContract(t, mode, signature, reply)
					require.Error(t, err, "outputs: %v", outputs)
					require.ErrorIs(t, err, dspyerrors.ErrParseFailed)

					var parseErr *dspyerrors.ParseError
					require.ErrorAs(t, err, &parseErr)
					assert.Equal(t, reply, parseErr.Raw, "the error keeps the reply")
				})
			}
		}
	}
}

// TestAdapterContract_RepliesTheFormatAllows covers the original replies that
// one adapter's format does allow. Each has one acceptable value.
func TestAdapterContract_RepliesTheFormatAllows(t *testing.T) {
	single := qaSignature("answer")
	xml := []contractMode{contractModes[0], contractModes[1], contractModes[2], contractModes[3]}
	json := []contractMode{contractModes[4], contractModes[5]}
	cases := []struct {
		name  string
		modes []contractMode
		reply string
		want  string
	}{
		{name: "tags without line breaks", modes: xml, reply: "<answer>-42</answer>", want: "-42"},
		{name: "a multi-line value written by hand", modes: xml, reply: "<answer>\n-42\n-43\n</answer>", want: "-42\n-43"},
		{name: "a fenced value written by hand", modes: xml, reply: "<answer>\n```\n-42\n```\n</answer>", want: "```\n-42\n```"},
		{name: "a JSON string with escapes written by hand", modes: json, reply: `{"answer": "*first\n*second"}`, want: "*first\n*second"},
		{name: "prose before the tag", modes: xml, reply: "Sure, here it is.\n<answer>*ptr</answer>", want: "*ptr"},
		{name: "a JSON object", modes: json, reply: `{"answer": "-42"}`, want: "-42"},
		{name: "a fenced JSON object", modes: json, reply: "```json\n{\"answer\": \"1962\"}\n```", want: "1962"},
		{name: "prose around a JSON object", modes: json, reply: "Sure.\n{\"answer\": \"*ptr\"}\nDone.", want: "*ptr"},
	}
	for _, tc := range cases {
		for _, mode := range tc.modes {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				got, err := runContract(t, mode, single, tc.reply)
				require.NoError(t, err)
				assert.Equal(t, map[string]any{"answer": tc.want}, got)
			})
		}
	}

	t.Run("the JSON fallback reads a JSON reply the tag parser rejected", func(t *testing.T) {
		for _, mode := range []contractMode{contractModes[2], contractModes[3]} {
			got, err := runContract(t, mode, single, "-42", `{"answer": "-42"}`)
			require.NoError(t, err, mode.name)
			assert.Equal(t, map[string]any{"answer": "-42"}, got, mode.name)
		}
	})

	// These replies are outside the format, and an adapter may still read
	// them. If it does, only one value is right; runContract checks the rest.
	lenient := []struct {
		name  string
		modes []contractMode
		reply string
		want  string
	}{
		{name: "a JSON number for a text field", modes: json, reply: "```json\n{\"answer\":-42}\n```", want: "-42"},
		{name: "the old response wrapper", modes: xml, reply: "<response><answer>-42</answer></response>", want: "-42"},
		{name: "the old wrapper with a rationale", modes: xml, reply: "<response><rationale>Born 1962.</rationale><answer>1962</answer></response>", want: "1962"},
	}
	for _, tc := range lenient {
		for _, mode := range tc.modes {
			t.Run(tc.name+" gives its value or an error/"+mode.name, func(t *testing.T) {
				got, err := runContract(t, mode, single, tc.reply)
				if err == nil {
					assert.Equal(t, map[string]any{"answer": tc.want}, got)
				}
			})
		}
	}
}

// TestAdapterContract_PartialRepliesNameTheFieldThatIsMissing re-targets the
// replies that skipped a label or left one empty: the reply follows the
// adapter's format and lacks one field.
func TestAdapterContract_PartialRepliesNameTheFieldThatIsMissing(t *testing.T) {
	signature := qaSignature("reasoning", "answer")
	xml, json := contractModes[:4], contractModes[4:]
	cases := []struct {
		name    string
		modes   []contractMode
		reply   string
		missing string
	}{
		{name: "first tag skipped", modes: xml, reply: "<answer>1962</answer>", missing: "reasoning"},
		{name: "prose in place of the first tag", modes: xml, reply: "Born in 1962.\n<answer>1962</answer>", missing: "reasoning"},
		{name: "second tag absent", modes: xml, reply: "<reasoning>Born in 1962.</reasoning>", missing: "answer"},
		{name: "second tag null", modes: xml, reply: "<reasoning>Born in 1962.</reasoning>\n<answer />", missing: "answer"},
		{name: "first key skipped", modes: json, reply: `{"answer": "1962"}`, missing: "reasoning"},
		{name: "second key absent", modes: json, reply: `{"reasoning": "Born in 1962."}`, missing: "answer"},
		{name: "second key null", modes: json, reply: `{"reasoning": "Born in 1962.", "answer": null}`, missing: "answer"},
	}
	for _, tc := range cases {
		for _, mode := range tc.modes {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				outputs, err := runContract(t, mode, signature, tc.reply)
				require.Error(t, err, "outputs: %v", outputs)

				var parseErr *dspyerrors.ParseError
				require.ErrorAs(t, err, &parseErr)
				_, invalid := parseErr.InvalidFields[tc.missing]
				assert.True(t, invalid || assert.ObjectsAreEqual([]string{tc.missing}, parseErr.MissingFields),
					"the error names %q: missing %v, invalid %v", tc.missing, parseErr.MissingFields, parseErr.InvalidFields)
				assert.Equal(t, tc.reply, parseErr.Raw, "the error keeps the reply")
			})
		}
	}
}

// finishWithoutAnswer are final ReAct replies that name Finish and leave the
// answer empty: one written for review and five saved from live runs. The
// text loop returns them as a success with a nil answer.
var finishWithoutAnswer = []string{
	"thought: The previous tool result gives 579.\naction: <action><tool_name>Finish</tool_name></action>",
	"thought:\nThe previous tool result gives the sum, 579.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\n\nanswer:",
	"thought:\nThe multiplication result is 126.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\n\nanswer:",
	"thought:\nThe prior tool result gives 77 + 88 = 165.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\nanswer:\n",
	"thought:\nThe previous tool result is 400.\n\naction:\n<action><tool_name>Finish</tool_name></action>\n\nobservation:\n\nanswer:",
	"thought:\nThe recorded tool result gives the sum as 165.\naction:\n<action><tool_name>Finish</tool_name></action>\nobservation:\n\nanswer:",
}

func TestAdapterContract_ReActFinishWithoutAnAnswerIsAnError(t *testing.T) {
	signature := qaSignature("answer")
	inputs := map[string]any{"question": "What is 123 plus 456?"}
	requireMissingAnswer := func(t *testing.T, outputs map[string]any, err error) {
		t.Helper()
		assert.Nil(t, outputs)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	}

	for index, reply := range finishWithoutAnswer {
		t.Run(fmt.Sprintf("reply %d/model with tool calling", index), func(t *testing.T) {
			model := &toolChatLLM{
				BaseLLM:   core.NewBaseLLM("test", "tools", []core.Capability{core.CapabilityChat, core.CapabilityToolCalling}, nil),
				responses: []*core.ChatResponse{{Content: reply, FinishReason: core.FinishReasonStop}},
			}
			react := NewReAct(signature, addOnlyRegistry(t), 2).WithAdapter(adapters.NewXMLAdapter())
			react.SetLLM(model)

			outputs, err := react.Process(context.Background(), inputs)
			requireMissingAnswer(t, outputs, err)
		})

		t.Run(fmt.Sprintf("reply %d/model without tool calling", index), func(t *testing.T) {
			react := NewReAct(signature, addOnlyRegistry(t), 2).WithAdapter(adapters.NewXMLAdapter())
			react.SetLLM(&scriptedLLM{replies: []string{reply}})

			outputs, trace, err := react.ProcessWithTrace(context.Background(), inputs)
			requireMissingAnswer(t, outputs, err)
			var parseErr *dspyerrors.ParseError
			require.ErrorAs(t, err, &parseErr)
			assert.Equal(t, []string{"answer"}, parseErr.MissingFields)
			assert.Equal(t, "finish", trace.TerminationCause)
			assert.Nil(t, trace.Output)
		})

		t.Run(fmt.Sprintf("reply %d/no adapter keeps the nil answer", index), func(t *testing.T) {
			react := NewReAct(signature, addOnlyRegistry(t), 2)
			react.SetLLM(&scriptedLLM{replies: []string{reply}})

			outputs, err := react.Process(context.Background(), inputs)
			require.NoError(t, err)
			assert.Nil(t, outputs["answer"])
		})
	}

	t.Run("a Finish that carries the answer succeeds without tool calling", func(t *testing.T) {
		react := NewReAct(signature, addOnlyRegistry(t), 2).WithAdapter(adapters.NewXMLAdapter())
		react.SetLLM(&scriptedLLM{replies: []string{
			"thought: The tool returned 579.\naction: <action><tool_name>Finish</tool_name></action>\nanswer: 579",
		}})

		outputs, err := react.Process(context.Background(), inputs)
		require.NoError(t, err)
		assert.Equal(t, "579", outputs["answer"])
	})
}

func TestAdapterContract_ReActOutOfIterationsWithoutAnAnswerIsAnError(t *testing.T) {
	// One tool call uses the only iteration; the fallback extraction then
	// replies in prose, so it yields no answer.
	replies := []string{
		"thought: I need to add.\naction: <action><tool_name>add</tool_name><arguments><arg key=\"a\">2</arg><arg key=\"b\">3</arg></arguments></action>",
		"I could not work it out.",
	}
	inputs := map[string]any{"question": "What is 2 plus 3?"}

	react := NewReAct(qaSignature("answer"), addOnlyRegistry(t), 1).WithAdapter(adapters.NewXMLAdapter())
	react.SetLLM(&scriptedLLM{replies: replies})
	outputs, trace, err := react.ProcessWithTrace(context.Background(), inputs)
	assert.Nil(t, outputs)
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	assert.Equal(t, "max_iterations", trace.TerminationCause)

	react = NewReAct(qaSignature("answer"), addOnlyRegistry(t), 1)
	react.SetLLM(&scriptedLLM{replies: replies})
	outputs, err = react.Process(context.Background(), inputs)
	require.NoError(t, err, "without an adapter the result is unchanged")
	assert.Nil(t, outputs["answer"])
}

func TestAdapterContract_ReActFinishChecksTheCurrentSignature(t *testing.T) {
	react := NewReAct(qaSignature("answer"), addOnlyRegistry(t), 2).WithAdapter(adapters.NewXMLAdapter())
	live := react.GetSignature()
	live.Outputs = append(live.Outputs, core.OutputField{Field: core.NewField("source")})
	react.SetSignature(live)
	react.SetLLM(&scriptedLLM{replies: []string{
		"thought: Done.\naction: <action><tool_name>Finish</tool_name></action>\nanswer: 579",
	}})

	_, err := react.Process(context.Background(), map[string]any{"question": "What is 123 plus 456?"})
	var parseErr *dspyerrors.ParseError
	require.ErrorAs(t, err, &parseErr)
	assert.Equal(t, []string{"source"}, parseErr.MissingFields)
}

func TestReActAdapter_IsSavedAndRestored(t *testing.T) {
	newProgram := func(react *ReAct) *core.Program {
		program := core.NewProgram(map[string]core.Module{"agent": react}, nil)
		return &program
	}
	path := filepath.Join(t.TempDir(), "program.json")
	saved := NewReAct(qaSignature("answer"), addOnlyRegistry(t), 2).WithAdapter(adapters.NewJSONAdapter())
	require.NoError(t, core.SaveProgram(newProgram(saved), path))

	restored := NewReAct(qaSignature("answer"), addOnlyRegistry(t), 2)
	require.NoError(t, core.LoadProgram(newProgram(restored), path))
	name, version := restored.GetAdapterInfo()
	assert.Equal(t, "json", name)
	assert.Equal(t, "v1", version)

	plainPath := filepath.Join(t.TempDir(), "plain.json")
	require.NoError(t, core.SaveProgram(newProgram(NewReAct(qaSignature("answer"), addOnlyRegistry(t), 2)), plainPath))
	raw, err := os.ReadFile(plainPath)
	require.NoError(t, err)
	assert.NotContains(t, string(raw), "adapter", "a program without an adapter records none")
}

func addOnlyRegistry(t *testing.T) *tools.InMemoryToolRegistry {
	t.Helper()
	registry := tools.NewInMemoryToolRegistry()
	require.NoError(t, registry.Register(tools.NewFuncTool("add", "Add two integers a and b.",
		models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{
			"a": {Type: "integer", Description: "first"},
			"b": {Type: "integer", Description: "second"},
		}},
		func(context.Context, map[string]any) (*models.CallToolResult, error) {
			return &models.CallToolResult{Content: []models.Content{models.TextContent{Type: "text", Text: "579"}}}, nil
		})))
	return registry
}
