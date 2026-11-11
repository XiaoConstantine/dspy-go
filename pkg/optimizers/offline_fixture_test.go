package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

const (
	offlineBaseline = "Route tickets by symbol using a=amber b=amber c=amber d=amber."
	offlinePartial  = "Route tickets by symbol using a=amber b=blue c=coral d=amber."
	offlineCorrect  = "Route tickets by symbol using a=amber b=blue c=coral d=dusk."
	offlineDecoy    = "Explain routing with a=blue b=coral c=dusk d=amber. Include a metaphor and a detailed analytical justification."
	offlineOther    = "Explain routing with a=coral b=dusk c=amber d=blue. Include a creative analogy and a comprehensive structured analysis."
)

var offlineRule = regexp.MustCompile(`\b([abcd])=(amber|blue|coral|dusk)\b`)

// offlineLLM is a scripted provider, not a mocked optimizer or metric. Task
// completions depend only on the rendered instruction, demos, and current input.
// It has no real provider, transport, credentials, sleeps, or timing-dependent
// responses. Unimplemented LLM capabilities panic rather than reaching a network.
type offlineLLM struct {
	core.LLM
	mu           sync.Mutex
	instructions map[string]bool
	demos        []core.Example
	singleCalls  int
	unexpected   []string
}

func newOfflineLLM() *offlineLLM {
	return &offlineLLM{instructions: map[string]bool{offlineBaseline: true}}
}

func (m *offlineLLM) ModelID() string { return "offline-codebook" }

func (m *offlineLLM) ProviderName() string { return "scripted" }

func (m *offlineLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *offlineLLM) Generate(ctx context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Serialize only the script bookkeeping, never optimizer execution. Identical
	// concurrent requests return identical responses; only MIPRO's sequential
	// instruction-generation protocol consumes a response sequence.
	m.mu.Lock()
	defer m.mu.Unlock()
	response, err := m.respond(prompt)
	if err != nil {
		m.unexpected = append(m.unexpected, err.Error())
		return nil, err
	}
	return &core.LLMResponse{Content: response}, nil
}

func (m *offlineLLM) propose(instructions ...string) string {
	for _, instruction := range instructions {
		m.instructions[instruction] = true
	}
	return strings.Join(instructions, "\n")
}

func (m *offlineLLM) respond(prompt string) (string, error) {
	switch {
	case strings.HasPrefix(prompt, "Given the fields "):
		return m.complete(prompt)
	case strings.HasPrefix(prompt, "ticket="):
		// MIPRO's teacher protocol is a raw input rather than a Predict prompt.
		return offlineTeacherAnswer(prompt)
	case strings.HasPrefix(prompt, "Generate an instruction for the following signature:"):
		choices := []string{offlineDecoy, offlineCorrect, offlinePartial}
		instruction := choices[m.singleCalls%len(choices)]
		m.singleCalls++
		return m.propose(instruction), nil
	case strings.HasPrefix(prompt, "You are an expert prompt engineer specializing"):
		return m.propose(offlineDecoy, offlinePartial, offlineOther, offlineBaseline, offlineDecoy), nil
	case strings.HasPrefix(prompt, "You are an expert prompt engineer analyzing previous instruction attempts"):
		// Refinement must receive a useful parent, not just make another LLM call.
		high, _, _ := strings.Cut(strings.SplitN(prompt, "HIGH-PERFORMING INSTRUCTIONS:", 2)[1], "LOW-PERFORMING INSTRUCTIONS:")
		return m.propose(offlineOther, offlineRefine(high), offlineDecoy), nil
	case strings.HasPrefix(prompt, "Generate ") && strings.Contains(prompt, "diverse variations of the instruction"):
		var count int
		if _, err := fmt.Sscanf(prompt, "Generate %d", &count); err != nil {
			return "", err
		}
		// Seed a refinement problem, not an already-solved population. The short,
		// useful parent competes with verbose but incorrect heuristic attractors.
		choices := []string{offlineDecoy, offlinePartial, offlineOther}
		variations := make([]string, count)
		for i := range variations {
			variations[i] = choices[i%len(choices)]
		}
		return m.propose(variations...), nil
	case strings.HasPrefix(prompt, "Generate a variation of the following instruction"):
		// Rephrasing the initial instruction finds a partial codebook. Rephrasing
		// it again does not fix its missing rule; trajectory evidence is needed.
		if strings.Contains(prompt, offlineBaseline) {
			return m.propose(offlinePartial), nil
		}
		if strings.Contains(prompt, offlineCorrect) {
			return m.propose(offlineCorrect), nil
		}
		return m.propose(offlinePartial), nil
	case strings.HasPrefix(prompt, "Analyze the following successful"):
		return "RULE: Use the observed routing labels rather than a constant label.", nil
	case strings.HasPrefix(prompt, "Given the following instruction and rule"):
		return m.propose(offlineRefine(prompt)), nil
	case strings.HasPrefix(prompt, "Analyze the following optimization progress"):
		return "Preserve rules supported by the examples and compare candidate accuracy.", nil
	case strings.HasPrefix(prompt, "As an expert prompt engineer, critically analyze"),
		strings.HasPrefix(prompt, "# Prompt Candidate Analysis and Critique"):
		return "STRENGTHS:\n- Explicit routing rules.\nWEAKNESSES:\n- Some labels disagree with the examples.\nSUGGESTIONS:\n- Correct the missing label in a useful codebook.\nCONFIDENCE: 1.0", nil
	case strings.HasPrefix(prompt, "Apply a "):
		// Inspect the parent, not expected answers in the reflection evidence.
		parent := strings.SplitN(prompt, "\n\n", 2)[0]
		return m.propose(offlineRefine(parent)), nil
	case strings.HasPrefix(prompt, "You are improving a GEPA instruction"):
		return m.propose(offlineGEPARefine(prompt)), nil
	default:
		return "", fmt.Errorf("offline script does not recognize prompt: %.160s", prompt)
	}
}

// GEPA can refine the original seed without paraphrase initialization, but only
// from concrete input/expected/actual evidence. Bad parents remain decoys; seeing
// an expected answer somewhere in the prompt is not sufficient to improve them.
func offlineGEPARefine(prompt string) string {
	parent, _, _ := strings.Cut(prompt, "MODULE:")
	_, evidence, _ := strings.Cut(prompt, "EXAMPLE-LEVEL EVIDENCE:")
	lines := strings.Split(evidence, "\n")
	labels := map[byte]string{'a': "amber", 'b': "blue", 'c': "coral", 'd': "dusk"}
	for i, line := range lines {
		_, inputJSON, ok := strings.Cut(line, "Inputs: ")
		if !ok || i+2 >= len(lines) {
			continue
		}
		var input, expected, actual map[string]string
		if json.Unmarshal([]byte(inputJSON), &input) != nil ||
			json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(lines[i+1]), "Expected: ")), &expected) != nil ||
			json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(lines[i+2]), "Actual: ")), &actual) != nil {
			continue
		}
		_, symbols, err := offlineInput(input["prompt"])
		want, got := strings.Fields(expected["completion"]), strings.Fields(actual["completion"])
		if err != nil || len(want) != 4 || len(got) != 4 {
			continue
		}
		valid := true
		for j := range symbols {
			valid = valid && want[j] == labels[symbols[j]]
		}
		if !valid {
			continue
		}
		if strings.Contains(parent, offlineBaseline) {
			return offlinePartial
		}
		return offlineRefine(parent)
	}
	return offlineDecoy
}

func offlineRefine(parent string) string {
	if strings.Contains(parent, offlinePartial) || strings.Contains(parent, offlineCorrect) {
		return offlineCorrect
	}
	return offlineDecoy
}

// The teacher is fallible: even-numbered training tickets get a rotated (wrong)
// codebook. Bootstrap must filter those demos using the supplied metric.
func offlineTeacherAnswer(input string) (string, error) {
	id, symbols, err := offlineInput(input)
	if err != nil {
		return "", err
	}
	labels := []string{"amber", "blue", "coral", "dusk"}
	var answers []string
	for _, symbol := range symbols {
		index := int(symbol - 'a')
		if id%2 == 0 {
			index = (index + 1) % len(labels)
		}
		answers = append(answers, labels[index])
	}
	return strings.Join(answers, " "), nil
}

func offlineInput(input string) (int, string, error) {
	var id int
	var symbols string
	if _, err := fmt.Sscanf(strings.TrimSpace(input), "ticket=%d symbols=%s", &id, &symbols); err != nil {
		return 0, "", err
	}
	if len(symbols) != 4 || strings.ContainsAny(symbols, " \n") {
		return 0, "", fmt.Errorf("invalid symbols %q", symbols)
	}
	for _, symbol := range symbols {
		if symbol < 'a' || symbol > 'd' {
			return 0, "", fmt.Errorf("invalid symbol %q", symbol)
		}
	}
	return id, symbols, nil
}

func (m *offlineLLM) complete(prompt string) (string, error) {
	blocks := strings.Split(prompt, "---\n\n")
	if len(blocks) < 2 {
		return "", fmt.Errorf("missing Predict input block")
	}
	input := strings.TrimSpace(strings.TrimPrefix(blocks[len(blocks)-1], "prompt: "))
	_, symbols, err := offlineInput(input)
	if err != nil {
		return "", err
	}
	rules := map[byte]string{}
	for _, match := range offlineRule.FindAllStringSubmatch(blocks[0], -1) {
		rules[match[1][0]] = match[2]
	}
	for _, block := range blocks[1 : len(blocks)-1] {
		demoInput, demoOutput, ok := strings.Cut(block, "\ncompletion: ")
		if !ok {
			return "", fmt.Errorf("invalid demonstration block %q", block)
		}
		_, demoSymbols, err := offlineInput(strings.TrimPrefix(demoInput, "prompt: "))
		if err != nil {
			return "", err
		}
		labels := strings.Fields(demoOutput)
		if len(labels) != len(demoSymbols) {
			return "", fmt.Errorf("invalid demonstration labels %q", demoOutput)
		}
		for i := range demoSymbols {
			rules[demoSymbols[i]] = labels[i]
		}
	}
	var answer []string
	for i := range symbols {
		label, ok := rules[symbols[i]]
		if !ok {
			return "", fmt.Errorf("instruction has no rule for %c", symbols[i])
		}
		answer = append(answer, label)
	}
	if strings.Contains(blocks[0], "Explain routing") {
		return "completion: A creative metaphor and comprehensive structured analysis of " + input +
			" suggests these routing labels: " + strings.Join(answer, " ") +
			". This detailed explanation is deliberately not the requested four-label answer.", nil
	}
	return "completion: " + strings.Join(answer, " "), nil
}

type offlineTeacher struct{ *offlineLLM }

func (m *offlineTeacher) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	blocks := strings.Split(prompt, "---\n\n")
	input := strings.TrimSpace(strings.TrimPrefix(blocks[len(blocks)-1], "prompt: "))
	answer, err := offlineTeacherAnswer(input)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.demos = append(m.demos, core.Example{
		Inputs: map[string]any{"prompt": input}, Outputs: map[string]any{"completion": answer},
	})
	m.mu.Unlock()
	return &core.LLMResponse{Content: "completion: " + answer}, nil
}

func offlineProgram(model core.LLM) core.Program {
	signature := core.NewSignature(
		[]core.InputField{{Name: "prompt"}},
		[]core.OutputField{{Name: "completion", Prefix: "completion:"}},
	).WithInstruction(offlineBaseline)
	predict := modules.NewPredict(signature)
	predict.SetLLM(model)
	return core.NewProgramWithForwardFactory(map[string]core.Module{"route": predict},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				// Exercise GEPA's actual installed interceptors, including its parent
				// selection objectives; plain text Predict.Process bypasses them.
				return modules["route"].(core.InterceptableModule).ProcessWithInterceptors(ctx, inputs, nil)
			}
		})
}

func offlineWithInstruction(base core.Program, instruction string) core.Program {
	program := base.Clone()
	module := program.Modules["route"]
	module.SetSignature(module.GetSignature().WithInstruction(instruction))
	return program
}

func offlineExamples(seed int64, offset int) []core.Example {
	rng := rand.New(rand.NewSource(seed))
	labels := map[byte]string{'a': "amber", 'b': "blue", 'c': "coral", 'd': "dusk"}
	examples := make([]core.Example, 12)
	for i := range examples {
		symbols := []byte("abcd")
		rng.Shuffle(len(symbols), func(i, j int) { symbols[i], symbols[j] = symbols[j], symbols[i] })
		var answer []string
		for _, symbol := range symbols {
			answer = append(answer, labels[symbol])
		}
		examples[i] = core.Example{
			Inputs:  map[string]any{"prompt": "ticket=" + strconv.Itoa(offset+i) + " symbols=" + string(symbols)},
			Outputs: map[string]any{"completion": strings.Join(answer, " ")},
		}
	}
	return examples
}

// The metric scores the four routing decisions independently. Every example
// contains all symbols, so minibatch composition cannot inflate its score.
func offlineMetric(expected, actual map[string]any) float64 {
	want, ok := expected["completion"].(string)
	if !ok {
		return 0
	}
	got, ok := actual["completion"].(string)
	if !ok {
		return 0
	}
	wants, gots := strings.Fields(want), strings.Fields(got)
	if len(wants) != 4 || len(gots) != len(wants) {
		return 0
	}
	var score float64
	for i := range wants {
		if wants[i] == gots[i] {
			score += 0.25
		}
	}
	return score
}

func (m *offlineLLM) randomCandidate(base core.Program, seed int64, demos bool) core.Program {
	m.mu.Lock()
	defer m.mu.Unlock()
	rng := rand.New(rand.NewSource(seed))
	if demos {
		// Sort observations so goroutine completion order cannot affect the mutant.
		observed := slices.Clone(m.demos)
		if len(observed) == 0 {
			return base.Clone()
		}
		slices.SortFunc(observed, func(a, b core.Example) int {
			return strings.Compare(a.Inputs["prompt"].(string), b.Inputs["prompt"].(string))
		})
		program := base.Clone()
		program.Modules["route"].(*modules.Predict).SetDemos([]core.Example{observed[rng.Intn(len(observed))]})
		return program
	}
	choices := make([]string, 0, len(m.instructions))
	for instruction := range m.instructions {
		choices = append(choices, instruction)
	}
	slices.Sort(choices)
	return offlineWithInstruction(base, choices[rng.Intn(len(choices))])
}
