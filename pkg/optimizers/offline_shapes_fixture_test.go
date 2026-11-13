package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"math/rand"
	"regexp"
	"slices"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

const offlineCopy = "Copy the query text into completion unchanged, preserving the order of all labels."

// Extend only the provider protocol: optimizers still execute real modules and
// whole programs. In the pipeline, route owns the codebook and write can only
// copy its query. Changing the last predictor cannot solve the routing task.
type offlineShapeLLM struct {
	*offlineLLM
	shape       string
	late        bool
	proposals   int
	refinements int
}

func newOfflineShapeLLM(shape string, late bool) *offlineShapeLLM {
	return &offlineShapeLLM{offlineLLM: newOfflineLLM(), shape: shape, late: late}
}

func (m *offlineShapeLLM) Generate(ctx context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if strings.HasPrefix(prompt, "Given the fields ") {
		response, err := m.completeShape(prompt, false)
		return &core.LLMResponse{Content: response}, err
	}
	// Writer proposals remain copy instructions, including proposals generated
	// from its signature alone. They never enter the route candidate domain.
	if strings.Contains(prompt, offlineCopy) || strings.Contains(prompt, `Convert "query" into "completion"`) {
		return &core.LLMResponse{Content: strings.Repeat(offlineCopy+"\n", 5)}, nil
	}
	if m.late {
		if response, ok := m.lateProposal(prompt); ok {
			return &core.LLMResponse{Content: response}, nil
		}
	}
	if m.shape == "cot" && strings.HasPrefix(prompt, "You are improving a GEPA instruction") {
		return &core.LLMResponse{Content: m.propose(offlineShapeGEPARefine(prompt))}, nil
	}
	response, err := m.respond(prompt)
	if err != nil {
		m.unexpected = append(m.unexpected, err.Error())
	}
	return &core.LLMResponse{Content: response}, err
}

// Only proposal availability depends on the round. Task completions never do:
// every flat instruction is an ordinary constant-label codebook scoring .25.
func (m *offlineShapeLLM) lateProposal(prompt string) (string, bool) {
	flat := func(n int) string { return fmt.Sprintf("%s Revision %d.", offlineBaseline, n) }
	switch {
	case strings.HasPrefix(prompt, "You are an expert prompt engineer specializing"):
		return m.propose(flat(0), flat(1), flat(2), flat(3), flat(4)), true
	case strings.HasPrefix(prompt, "You are an expert prompt engineer analyzing previous instruction attempts"):
		m.refinements++
		if m.refinements <= 3 {
			return m.propose(flat(10*m.refinements), flat(10*m.refinements+1), flat(10*m.refinements+2), flat(10*m.refinements+3), flat(10*m.refinements+4)), true
		}
		return m.propose(offlineCorrect, offlinePartial, offlineDecoy, offlineOther, flat(100)), true
	case strings.HasPrefix(prompt, "Generate a variation of the following instruction"),
		strings.HasPrefix(prompt, "Given the following instruction and rule"):
		m.proposals++
		// SIMBA defaults: incumbent plus five new candidates per step.
		if m.proposals <= 15 {
			return m.propose(flat(0)), true
		}
		return m.propose(offlineCorrect), true
	case strings.HasPrefix(prompt, "You are improving a GEPA instruction"):
		m.proposals++
		if m.proposals <= 3 {
			return m.propose(flat(m.proposals)), true
		}
		// Retain the existing requirement for real input/expected/actual evidence.
		return m.propose(offlineGEPARefine(prompt)), true
	default:
		return "", false
	}
}

var offlineCompletionField = regexp.MustCompile(`"completion":("(?:\\.|[^"\\])*")`)

// Reflection evidence is human-readable text, not a JSON wire protocol. CoT
// adds rationale and raw XML metadata that can exceed GEPA's rendering limit.
// Read a complete answer field even when later metadata is truncated; never
// invent a missing/truncated answer or drop the existing input/expected checks.
func offlineShapeGEPARefine(prompt string) string {
	lines := strings.Split(prompt, "\n")
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "Actual: ") {
			match := offlineCompletionField.FindStringSubmatch(line)
			var answer string
			if len(match) == 2 && json.Unmarshal([]byte(match[1]), &answer) == nil {
				lines[i] = `   Actual: {"completion":` + match[1] + `}`
			}
		}
	}
	return offlineGEPARefine(strings.Join(lines, "\n"))
}

func offlineFields(block string) map[string]string {
	fields := make(map[string]string)
	for _, line := range strings.Split(block, "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok {
			fields[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	return fields
}

func (m *offlineShapeLLM) completeShape(prompt string, teacher bool) (string, error) {
	blocks := strings.Split(prompt, "---\n\n")
	if len(blocks) < 2 {
		return "", fmt.Errorf("missing Predict input block")
	}
	input := offlineFields(blocks[len(blocks)-1])
	inputKey, outputKey := "prompt", "completion"
	if _, copyQuery := input["query"]; copyQuery {
		inputKey = "query"
	} else if m.shape == "pipeline" {
		outputKey = "query"
	}
	// Do not silently repair program-level demos installed on a local module.
	// Rationale is optional, but each demo must have that module's I/O fields.
	for i, block := range blocks[1 : len(blocks)-1] {
		fields := offlineFields(block)
		if fields[inputKey] == "" || fields[inputKey] == "<nil>" || fields[outputKey] == "" || fields[outputKey] == "<nil>" {
			return "", fmt.Errorf("invalid %s -> %s demonstration: %q", inputKey, outputKey, block)
		}
		blocks[i+1] = "prompt: " + fields[inputKey] + "\ncompletion: " + fields[outputKey] + "\n\n"
	}
	var answer string
	switch {
	case inputKey == "query":
		answer = input["query"]
	case teacher:
		var err error
		answer, err = offlineTeacherAnswer(input["prompt"])
		if err != nil {
			return "", err
		}
	default:
		blocks[len(blocks)-1] = "prompt: " + input["prompt"]
		response, err := m.complete(strings.Join(blocks, "---\n\n"))
		if err != nil {
			return "", err
		}
		answer = strings.TrimPrefix(response, "completion: ")
	}
	if teacher && inputKey == "prompt" {
		m.demos = append(m.demos, core.Example{Inputs: map[string]any{inputKey: input[inputKey]}, Outputs: map[string]any{outputKey: answer}})
	}
	if m.shape == "cot" {
		// ChainOfThought's default multi-output mode requests XML. Exercise its
		// real XML interceptor rather than bypassing it with text-only output.
		return "<response><rationale>Apply the symbol rules in input order.</rationale><completion>" + html.EscapeString(answer) + "</completion></response>", nil
	}
	return outputKey + ": " + answer, nil
}

type offlineShapeTeacher struct{ *offlineShapeLLM }

func (m *offlineShapeTeacher) Generate(ctx context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	response, err := m.completeShape(prompt, true)
	return &core.LLMResponse{Content: response}, err
}

func offlineShapeProgram(model *offlineShapeLLM) core.Program {
	output := "completion"
	if model.shape == "pipeline" {
		output = "query"
	}
	signature := core.NewSignature([]core.InputField{{Name: "prompt"}},
		[]core.OutputField{{Name: output, Prefix: output + ":"}}).WithInstruction(offlineBaseline)
	var route core.Module = modules.NewPredict(signature)
	if model.shape == "cot" {
		route = modules.NewChainOfThought(signature)
	}
	route.SetLLM(model)
	moduleMap := map[string]core.Module{"route": route}
	if model.shape == "pipeline" {
		writer := modules.NewPredict(core.NewSignature([]core.InputField{{Name: "query"}},
			[]core.OutputField{{Name: "completion", Prefix: "completion:"}}).WithInstruction(offlineCopy))
		writer.SetLLM(model)
		moduleMap["write"] = writer
	}
	return core.NewProgramWithForwardFactory(moduleMap, func(bound map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			output, err := bound["route"].(core.InterceptableModule).ProcessWithInterceptors(ctx, inputs, nil)
			if err != nil || bound["write"] == nil {
				return output, err
			}
			return bound["write"].(core.InterceptableModule).ProcessWithInterceptors(ctx, map[string]any{"query": output["query"]}, nil)
		}
	})
}

func (m *offlineShapeLLM) randomProgram(base core.Program, seed int64, demos bool) core.Program {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !demos {
		choices := make([]string, 0, len(m.instructions))
		for instruction := range m.instructions {
			choices = append(choices, instruction)
		}
		slices.Sort(choices)
		return offlineWithInstruction(base, choices[rand.New(rand.NewSource(seed)).Intn(len(choices))])
	}
	program := base.Clone()
	observed := slices.Clone(m.demos)
	if len(observed) > 0 {
		slices.SortFunc(observed, func(a, b core.Example) int {
			return strings.Compare(a.Inputs["prompt"].(string), b.Inputs["prompt"].(string))
		})
		program.Modules["route"].(interface{ SetDemos([]core.Example) }).SetDemos([]core.Example{observed[rand.New(rand.NewSource(seed)).Intn(len(observed))]})
	}
	return program
}

// TPE is not a Program optimizer. This is the existing gate's 25-trial adapter,
// generalized only to avoid assuming route is a top-level *Predict. In particular
// it does not add a rebindability guard that the real SearchStrategy cannot own.
type offlineShapeTPE struct {
	seed  int64
	model *offlineShapeLLM
}

func (o offlineShapeTPE) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	strategy := NewTPEOptimizer(TPEConfig{Seed: o.seed})
	choices := []string{offlineDecoy, offlineCorrect, offlinePartial}
	const trials = 25
	if err := strategy.Initialize(SearchConfig{ParamSpace: map[string][]any{"instruction": {0, 1, 2}}, MaxTrials: trials, Seed: o.seed}); err != nil {
		return program, err
	}
	examples := core.MaterializeDataset(dataset)
	for range trials {
		params, err := strategy.SuggestParams(ctx)
		if err != nil {
			return program, err
		}
		instruction := choices[params["instruction"].(int)]
		o.model.mu.Lock()
		o.model.propose(instruction)
		o.model.mu.Unlock()
		score, _, err := offlineScore(ctx, offlineWithInstruction(program, instruction), examples, metric)
		if err != nil {
			return program, err
		}
		if err := strategy.UpdateResults(params, score); err != nil {
			return program, err
		}
	}
	params, _ := strategy.GetBestParams()
	return offlineWithInstruction(program, choices[params["instruction"].(int)]), nil
}

// MIPRO generates its instruction pool before trials. Delay program proposals,
// not model completions: the first three trials explicitly select the original
// instruction and no demos; subsequent proposals use its real seeded TPE search.
type offlineDelayedSearch struct {
	SearchStrategy
	trials int
}

func (s *offlineDelayedSearch) SuggestParams(ctx context.Context) (map[string]any, error) {
	params, err := s.SearchStrategy.SuggestParams(ctx)
	s.trials++
	if s.trials <= 3 {
		for key := range params {
			params[key] = float64(0)
		}
	}
	return params, err
}
