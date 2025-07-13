package optimizers

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func init() {
	mockLLM := new(testutil.MockLLM)

	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: `answer: 
	Paris`}, nil)
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(map[string]interface{}{"answer": "Paris"}, nil)

	core.GlobalConfig.DefaultLLM = mockLLM
	core.GlobalConfig.TeacherLLM = mockLLM
	core.GlobalConfig.ConcurrencyLevel = 1
}

func createProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	))

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {

		ctx, span := core.StartSpan(ctx, "Forward")
		defer core.EndSpan(ctx)
		span.WithAnnotation("inputs", inputs)
		outputs, err := predict.Process(ctx, inputs)
		if err != nil {
			span.WithError(err)
			return nil, err
		}
		span.WithAnnotation("outputs", outputs)
		return outputs, nil
	}

	return core.NewProgram(map[string]core.Module{"predict": predict}, forwardFunc)
}

func TestBootstrapFewShot(t *testing.T) {
	student := createProgram()
	// Create training set as core.Examples
	trainExamples := []core.Example{
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of France?"},
			Outputs: map[string]interface{}{"answer": "Paris"},
		},
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of Germany?"},
			Outputs: map[string]interface{}{"answer": "Berlin"},
		},
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of Italy?"},
			Outputs: map[string]interface{}{"answer": "Rome"},
		},
	}

	// Create dataset
	trainDataset := datasets.NewSimpleDataset(trainExamples)

	// Define metric functions
	boolMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
		return true // Always return true for this test
	}
	floatMetric := func(expected, actual map[string]interface{}) float64 {
		return 1.0 // Always return 1.0 for this test
	}

	// Create BootstrapFewShot optimizer
	maxBootstrapped := 2
	optimizer := NewBootstrapFewShot(boolMetric, maxBootstrapped)

	ctx := core.WithExecutionState(context.Background())

	// Compile the program
	optimizedProgram, _ := optimizer.Compile(ctx, student, trainDataset, floatMetric)

	// Check if the optimized program has the correct number of demonstrations
	optimizedPredict, ok := optimizedProgram.Modules["predict"].(*modules.Predict)
	assert.True(t, ok)
	assert.Equal(t, maxBootstrapped, len(optimizedPredict.Demos))

	// Check if the demonstrations are correct
	for _, demo := range optimizedPredict.Demos {
		assert.Contains(t, demo.Inputs, "question")
		assert.Contains(t, demo.Outputs, "answer")
		assert.Equal(t, "Paris", demo.Outputs["answer"])
	}
	// Verify the trace structure
	spans := core.CollectSpans(ctx)
	require.NotEmpty(t, spans, "Expected spans to be recorded")
	rootSpan := spans[0]
	assert.Equal(t, "Compilation", rootSpan.Operation, "Expected Compilation as root span")

	// Find Example spans (should be direct children of Compilation)
	var exampleSpans []*core.Span
	for _, span := range spans {
		if span.Operation == "Example" {
			exampleSpans = append(exampleSpans, span)
		}
	}
	assert.Equal(t, maxBootstrapped, len(exampleSpans),
		"Expected number of Example spans to match maxBootstrapped")

	// Verify span structure and content
	var compilationSpan *core.Span
	for _, span := range spans {
		if span.Operation == "Compilation" {
			compilationSpan = span
		}
	}

	// Verify compilation span
	require.NotNil(t, compilationSpan, "Expected to find compilation span")
	assert.NotZero(t, compilationSpan.StartTime)
	assert.Nil(t, compilationSpan.Error)

}

func TestBootstrapFewShotEdgeCases(t *testing.T) {

	trainExamples := []core.Example{
		{
			Inputs:  map[string]interface{}{"question": "Q1"},
			Outputs: map[string]interface{}{"answer": "A1"},
		},
		{
			Inputs:  map[string]interface{}{"question": "Q2"},
			Outputs: map[string]interface{}{"answer": "A2"},
		},
		{
			Inputs:  map[string]interface{}{"question": "Q3"},
			Outputs: map[string]interface{}{"answer": "A3"},
		},
	}
	trainDataset := datasets.NewSimpleDataset(trainExamples)
	dummyMetric := func(expected, actual map[string]interface{}) float64 { return 1.0 }

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool { return true }, 0)
		ctx := context.Background()

		optimized, err := optimizer.Compile(ctx, createProgram(), trainDataset, dummyMetric)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("MaxBootstrapped Large", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool {
			return true
		}, 100)
		ctx := context.Background()

		optimized, err := optimizer.Compile(ctx, createProgram(), trainDataset, dummyMetric)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}
		demoCount := len(optimized.Modules["predict"].(*modules.Predict).Demos)
		assert.Equal(t, len(trainExamples), demoCount)
	})

	t.Run("Metric Rejects All", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool { return false }, 2)
		ctx := context.Background()
		optimized, _ := optimizer.Compile(ctx, createProgram(), trainDataset, dummyMetric)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})
}

// Benchmark tests for BootstrapFewShot optimizer using shared benchmark utilities

// BenchmarkBootstrapFewShot runs comprehensive benchmarks for BootstrapFewShot optimizer.
func BenchmarkBootstrapFewShot(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	configs := testutil.StandardBenchmarkConfigs()

	testCases := []struct {
		name       string
		datasetKey string
		configKey  string
	}{
		{"Fast_Tiny", "tiny", "fast"},
		{"Fast_Small", "small", "fast"},
		{"Standard_Small", "small", "standard"},
		{"Standard_Medium", "medium", "standard"},
		{"Comprehensive_Medium", "medium", "comprehensive"},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			dataset := datasets[tc.datasetKey]
			config := configs[tc.configKey]

			// Create program
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			// Create BootstrapFewShot optimizer with config
			boolMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
				// Convert to float64 metric for compatibility
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.5 // Accept if accuracy > 50%
			}

			bootstrap := NewBootstrapFewShot(boolMetric, config.MaxSteps) // Use MaxSteps as MaxBootstrapped

			// Setup mock LLM
			mockLLM := setupBootstrapBenchmarkLLM()

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := bootstrap.Compile(ctx, program, benchDataset, nil)
				if err != nil {
					b.Fatalf("BootstrapFewShot compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkBootstrapFewShotDatasetScaling tests performance across different dataset sizes.
func BenchmarkBootstrapFewShotDatasetScaling(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	config := testutil.StandardBenchmarkConfigs()["fast"] // Use fast config for scaling tests

	for name, dataset := range datasets {
		b.Run(fmt.Sprintf("Dataset_%s_%d", name, dataset.Size), func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			boolMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.5
			}

			bootstrap := NewBootstrapFewShot(boolMetric, config.MaxSteps)

			mockLLM := setupBootstrapBenchmarkLLM()

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := bootstrap.Compile(ctx, program, benchDataset, nil)
				if err != nil {
					b.Fatalf("BootstrapFewShot compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkBootstrapFewShotParameterTuning tests performance with different parameter configurations.
func BenchmarkBootstrapFewShotParameterTuning(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"] // Use small dataset for parameter testing

	parameterTests := []struct {
		name            string
		maxBootstrapped int
		metricThreshold float64
	}{
		{"MaxBootstrap3_Threshold50", 3, 0.5},
		{"MaxBootstrap5_Threshold50", 5, 0.5},
		{"MaxBootstrap8_Threshold50", 8, 0.5},
		{"MaxBootstrap5_Threshold30", 5, 0.3},
		{"MaxBootstrap5_Threshold70", 5, 0.7},
	}

	for _, pt := range parameterTests {
		b.Run(pt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			boolMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > pt.metricThreshold
			}

			bootstrap := NewBootstrapFewShot(boolMetric, pt.maxBootstrapped)

			mockLLM := setupBootstrapBenchmarkLLM()

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := bootstrap.Compile(ctx, program, benchDataset, nil)
				if err != nil {
					b.Fatalf("BootstrapFewShot compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkBootstrapFewShotMetricVariations tests performance with different metric behaviors.
func BenchmarkBootstrapFewShotMetricVariations(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"]

	metricTests := []struct {
		name        string
		metricFunc  func(example, prediction map[string]interface{}, ctx context.Context) bool
		description string
	}{
		{
			"AlwaysAccept",
			func(example, prediction map[string]interface{}, ctx context.Context) bool { return true },
			"Metric that always accepts predictions",
		},
		{
			"AlwaysReject",
			func(example, prediction map[string]interface{}, ctx context.Context) bool { return false },
			"Metric that always rejects predictions",
		},
		{
			"Selective50",
			func(example, prediction map[string]interface{}, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.5
			},
			"Metric that accepts when accuracy > 50%",
		},
		{
			"Selective80",
			func(example, prediction map[string]interface{}, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.8
			},
			"Metric that accepts when accuracy > 80%",
		},
	}

	for _, mt := range metricTests {
		b.Run(mt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			bootstrap := NewBootstrapFewShot(mt.metricFunc, 5)

			mockLLM := setupBootstrapBenchmarkLLM()

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := bootstrap.Compile(ctx, program, benchDataset, nil)
				if err != nil {
					b.Fatalf("BootstrapFewShot compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkBootstrapFewShotConcurrency tests performance with different concurrency levels.
func BenchmarkBootstrapFewShotConcurrency(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"]

	concurrencyTests := []struct {
		name             string
		concurrencyLevel int
	}{
		{"Concurrency1", 1},
		{"Concurrency2", 2},
		{"Concurrency4", 4},
		{"Concurrency8", 8},
	}

	for _, ct := range concurrencyTests {
		b.Run(ct.name, func(b *testing.B) {
			// Set concurrency level for this test
			originalConcurrency := core.GlobalConfig.ConcurrencyLevel
			core.GlobalConfig.ConcurrencyLevel = ct.concurrencyLevel
			defer func() {
				core.GlobalConfig.ConcurrencyLevel = originalConcurrency
			}()

			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			boolMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.5
			}

			bootstrap := NewBootstrapFewShot(boolMetric, 5)

			mockLLM := setupBootstrapBenchmarkLLM()

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := bootstrap.Compile(ctx, program, benchDataset, nil)
				if err != nil {
					b.Fatalf("BootstrapFewShot compilation failed: %v", err)
				}
			}
		})
	}
}

// setupBootstrapBenchmarkLLM creates a mock LLM configured for BootstrapFewShot benchmarks.
func setupBootstrapBenchmarkLLM() *MockLLM {
	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "test response"}, nil)
	mockLLM.On("GetModelName").Return("benchmark-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	mockLLM.On("ProviderName").Return("benchmark")
	mockLLM.On("ModelID").Return("benchmark-model-id")
	return mockLLM
}
