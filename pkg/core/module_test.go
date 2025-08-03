package core

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// // MockLLM is a mock implementation of the LLM interface for testing.
type MockLLM struct{}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	return &LLMResponse{Content: "mock response"}, nil
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"response": "mock response"}, nil
}

func (m *MockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *MockLLM) CreateEmbedding(ctx context.Context, input string, options ...EmbeddingOption) (*EmbeddingResult, error) {
	return &EmbeddingResult{
		// Using float32 for the vector as embeddings are typically floating point numbers
		Vector: []float32{0.1, 0.2, 0.3},
		// Include token count to simulate real embedding behavior
		TokenCount: len(strings.Fields(input)),
		// Add metadata to simulate real response
		Metadata: map[string]interface{}{},
	}, nil
}

func (m *MockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...EmbeddingOption) (*BatchEmbeddingResult, error) {
	opts := NewEmbeddingOptions()
	for _, opt := range options {
		opt(opts)
	}

	// Create mock results for each input
	embeddings := make([]EmbeddingResult, len(inputs))
	for i, input := range inputs {
		embeddings[i] = EmbeddingResult{
			// Each embedding gets slightly different values to simulate real behavior
			Vector:     []float32{0.1 * float32(i+1), 0.2 * float32(i+1), 0.3 * float32(i+1)},
			TokenCount: len(strings.Fields(input)),
			Metadata: map[string]interface{}{
				"model":        opts.Model,
				"input_length": len(input),
				"batch_index":  i,
			},
		}
	}

	// Return the batch result
	return &BatchEmbeddingResult{
		Embeddings: embeddings,
		Error:      nil,
		ErrorIndex: -1, // -1 indicates no error
	}, nil
}

func (m *MockLLM) StreamGenerate(ctx context.Context, prompt string, opts ...GenerateOption) (*StreamResponse, error) {
	return nil, nil
}

func (m *MockLLM) ProviderName() string {
	return "mock"
}

func (m *MockLLM) ModelID() string {
	return "mock"
}

func (m *MockLLM) Capabilities() []Capability {
	return []Capability{}
}

func (m *MockLLM) GenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*LLMResponse, error) {
	// Convert content blocks to a simple text representation for mock response
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}
	return &LLMResponse{Content: "mock response for content: " + textContent}, nil
}

func (m *MockLLM) StreamGenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*StreamResponse, error) {
	// Create a simple mock stream response
	chunkChan := make(chan StreamChunk, 1)
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}

	chunkChan <- StreamChunk{
		Content: "mock stream response for content: " + textContent,
		Done:    true,
	}
	close(chunkChan)

	return &StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       func() {},
	}, nil
}

// // TestBaseModule tests the BaseModule struct and its methods.
func TestBaseModule(t *testing.T) {
	sig := NewSignature(
		[]InputField{{Field: Field{Name: "input"}}},
		[]OutputField{{Field: Field{Name: "output"}}},
	)
	bm := NewModule(sig)

	if !reflect.DeepEqual(bm.GetSignature(), sig) {
		t.Error("GetSignature did not return the correct signature")
	}

	mockLLM := &MockLLM{}
	bm.SetLLM(mockLLM)
	if bm.LLM != mockLLM {
		t.Error("SetLLM did not set the LLM correctly")
	}

	_, err := bm.Process(context.Background(), map[string]any{"input": "test"})
	if err == nil || err.Error() != "Process method not implemented" {
		t.Error("Expected 'Process method not implemented' error")
	}

	clone := bm.Clone()
	if !reflect.DeepEqual(clone.GetSignature(), bm.GetSignature()) {
		t.Error("Cloned module does not have the same signature")
	}
}

// TestModuleChain tests the ModuleChain struct and its methods.
func TestModuleChain(t *testing.T) {
	module1 := NewModule(NewSignature(
		[]InputField{{Field: Field{Name: "input1"}}},
		[]OutputField{{Field: Field{Name: "output1"}}},
	))
	module2 := NewModule(NewSignature(
		[]InputField{{Field: Field{Name: "input2"}}},
		[]OutputField{{Field: Field{Name: "output2"}}},
	))

	chain := NewModuleChain(module1, module2)

	if len(chain.Modules) != 2 {
		t.Errorf("Expected 2 modules in chain, got %d", len(chain.Modules))
	}

	sig := chain.GetSignature()
	if len(sig.Inputs) != 1 || sig.Inputs[0].Name != "input1" {
		t.Error("Chain signature inputs are incorrect")
	}
	if len(sig.Outputs) != 1 || sig.Outputs[0].Name != "output2" {
		t.Error("Chain signature outputs are incorrect")
	}
}

// Test types for typed module functionality.
type TestQAInputs struct {
	Question string `dspy:"question,required" description:"The question to answer"`
	Context  string `dspy:"context,required" description:"Context for answering"`
}

type TestQAOutputs struct {
	Answer     string `dspy:"answer" description:"The generated answer"`
	Confidence int    `dspy:"confidence" description:"Confidence score"`
}

// Enhanced MockLLM for testing typed modules.
type MockTypedModule struct {
	*BaseModule
}

func NewMockTypedModule() *MockTypedModule {
	signature := NewSignature(
		[]InputField{
			{Field: NewTextField("question", WithDescription("The question"))},
			{Field: NewTextField("context", WithDescription("Context"))},
		},
		[]OutputField{
			{Field: NewTextField("answer", WithDescription("The answer"))},
			{Field: NewTextField("confidence", WithDescription("Confidence"))},
		},
	)

	return &MockTypedModule{
		BaseModule: NewModule(signature),
	}
}

func (m *MockTypedModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	// Simple mock implementation with proper type assertion checking
	question, ok := inputs["question"].(string)
	if !ok {
		return nil, errors.New("mock input 'question' is not a string or is missing")
	}
	contextStr, ok := inputs["context"].(string)
	if !ok {
		return nil, errors.New("mock input 'context' is not a string or is missing")
	}

	return map[string]any{
		"answer":     "Based on " + contextStr + ", the answer to '" + question + "' is mocked",
		"confidence": 85,
	}, nil
}

func TestProcessTyped(t *testing.T) {
	ctx := context.Background()
	module := NewMockTypedModule()

	inputs := TestQAInputs{
		Question: "What is AI?",
		Context:  "AI is artificial intelligence",
	}

	// Test type-safe processing
	outputs, err := ProcessTyped[TestQAInputs, TestQAOutputs](ctx, module, inputs)

	if err != nil {
		t.Fatalf("ProcessTyped failed: %v", err)
	}

	if !strings.Contains(outputs.Answer, "What is AI?") {
		t.Errorf("Expected answer to contain question, got: %s", outputs.Answer)
	}

	if !strings.Contains(outputs.Answer, "AI is artificial intelligence") {
		t.Errorf("Expected answer to contain context, got: %s", outputs.Answer)
	}

	if outputs.Confidence != 85 {
		t.Errorf("Expected confidence 85, got: %d", outputs.Confidence)
	}
}

func TestProcessTypedWithValidation(t *testing.T) {
	ctx := context.Background()
	module := NewMockTypedModule()

	tests := []struct {
		name    string
		inputs  TestQAInputs
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid inputs",
			inputs: TestQAInputs{
				Question: "What is AI?",
				Context:  "AI is artificial intelligence",
			},
			wantErr: false,
		},
		{
			name: "missing required field",
			inputs: TestQAInputs{
				Question: "What is AI?",
				// Context missing
			},
			wantErr: true,
			errMsg:  "required input field 'context' cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputs, err := ProcessTypedWithValidation[TestQAInputs, TestQAOutputs](ctx, module, tt.inputs)

			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("Expected error to contain '%s', got: %s", tt.errMsg, err.Error())
				}
				if outputs.Answer != "" {
					t.Errorf("Expected empty answer on error, got: %s", outputs.Answer)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if outputs.Answer == "" {
					t.Errorf("Expected non-empty answer")
				}
				if outputs.Confidence != 85 {
					t.Errorf("Expected confidence 85, got: %d", outputs.Confidence)
				}
			}
		})
	}
}

func TestProcessTypedWithMapInputs(t *testing.T) {
	ctx := context.Background()
	module := NewMockTypedModule()

	// Test that legacy map inputs still work
	legacyInputs := map[string]any{
		"question": "What is deep learning?",
		"context":  "Deep learning uses neural networks",
	}

	outputs, err := ProcessTyped[map[string]any, TestQAOutputs](ctx, module, legacyInputs)

	if err != nil {
		t.Fatalf("ProcessTyped with map inputs failed: %v", err)
	}

	if !strings.Contains(outputs.Answer, "What is deep learning?") {
		t.Errorf("Expected answer to contain question, got: %s", outputs.Answer)
	}

	if outputs.Confidence != 85 {
		t.Errorf("Expected confidence 85, got: %d", outputs.Confidence)
	}
}

func TestProcessTypedWithMapOutputs(t *testing.T) {
	ctx := context.Background()
	module := NewMockTypedModule()

	inputs := TestQAInputs{
		Question: "What is reinforcement learning?",
		Context:  "RL is learning through interaction",
	}

	// Test that legacy map outputs still work
	outputs, err := ProcessTyped[TestQAInputs, map[string]any](ctx, module, inputs)

	if err != nil {
		t.Fatalf("ProcessTyped with map outputs failed: %v", err)
	}

	answer, exists := outputs["answer"]
	if !exists {
		t.Errorf("Expected 'answer' field in outputs")
	}

	if answerStr, ok := answer.(string); !ok {
		t.Errorf("Expected answer to be string, got %T", answer)
	} else if !strings.Contains(answerStr, "What is reinforcement learning?") {
		t.Errorf("Expected answer to contain question, got: %s", answerStr)
	}

	confidence, exists := outputs["confidence"]
	if !exists {
		t.Errorf("Expected 'confidence' field in outputs")
	}

	if confInt, ok := confidence.(int); !ok {
		t.Errorf("Expected confidence to be int, got %T", confidence)
	} else if confInt != 85 {
		t.Errorf("Expected confidence 85, got: %d", confInt)
	}
}

// Benchmark to ensure performance is acceptable.
func BenchmarkProcessTyped(b *testing.B) {
	ctx := context.Background()
	module := NewMockTypedModule()

	inputs := TestQAInputs{
		Question: "What is AI?",
		Context:  "AI is artificial intelligence",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ProcessTyped[TestQAInputs, TestQAOutputs](ctx, module, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProcessTypedWithValidation(b *testing.B) {
	ctx := context.Background()
	module := NewMockTypedModule()

	inputs := TestQAInputs{
		Question: "What is AI?",
		Context:  "AI is artificial intelligence",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ProcessTypedWithValidation[TestQAInputs, TestQAOutputs](ctx, module, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}
