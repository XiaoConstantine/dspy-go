package optimizers

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func createProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Name: "question"}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	))

	return core.NewProgramWithForwardFactory(
		map[string]core.Module{"predict": predict},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				ctx, span := core.StartSpan(ctx, "Forward")
				defer core.EndSpan(ctx)
				span.WithAnnotation("inputs", inputs)
				outputs, err := modules["predict"].Process(ctx, inputs)
				if err != nil {
					span.WithError(err)
					return nil, err
				}
				span.WithAnnotation("outputs", outputs)
				return outputs, nil
			}
		},
	)
}

func TestBootstrapFewShot(t *testing.T) {
	setupTestMockLLM(t)
	student := createProgram()
	// Create training set as core.Examples
	trainExamples := []core.Example{
		{
			Inputs:  map[string]any{"question": "What is the capital of France?"},
			Outputs: map[string]any{"answer": "Paris"},
		},
		{
			Inputs:  map[string]any{"question": "What is the capital of Germany?"},
			Outputs: map[string]any{"answer": "Berlin"},
		},
		{
			Inputs:  map[string]any{"question": "What is the capital of Italy?"},
			Outputs: map[string]any{"answer": "Rome"},
		},
	}

	// Create dataset
	trainDataset := datasets.NewSimpleDataset(trainExamples)

	// Define metric functions
	boolMetric := func(example, prediction map[string]any, ctx context.Context) bool {
		return true // Always return true for this test
	}
	floatMetric := func(expected, actual map[string]any) float64 {
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

	spansByID := make(map[string]*core.Span, len(spans))
	for _, span := range spans {
		spansByID[span.ID] = span
	}
	teacherPredictions := 0
	addedDemonstrations := 0
	for _, tracedSpan := range spans {
		switch tracedSpan.Operation {
		case "Example":
			assert.Equal(t, compilationSpan.ID, tracedSpan.ParentID)
		case "TeacherPrediction":
			teacherPredictions++
			parent := spansByID[tracedSpan.ParentID]
			require.NotNil(t, parent, "TeacherPrediction should have a traced parent")
			assert.Equal(t, "Example", parent.Operation)
		case "AddDemonstrations":
			addedDemonstrations++
			assert.Equal(t, compilationSpan.ID, tracedSpan.ParentID)
		}
	}
	assert.Equal(t, maxBootstrapped, teacherPredictions)
	assert.Equal(t, maxBootstrapped, addedDemonstrations)
}

func TestBootstrapFewShotEdgeCases(t *testing.T) {
	setupTestMockLLM(t)

	trainExamples := []core.Example{
		{
			Inputs:  map[string]any{"question": "Q1"},
			Outputs: map[string]any{"answer": "A1"},
		},
		{
			Inputs:  map[string]any{"question": "Q2"},
			Outputs: map[string]any{"answer": "A2"},
		},
		{
			Inputs:  map[string]any{"question": "Q3"},
			Outputs: map[string]any{"answer": "A3"},
		},
	}
	trainDataset := datasets.NewSimpleDataset(trainExamples)
	dummyMetric := func(expected, actual map[string]any) float64 { return 1.0 }

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]any, _ context.Context) bool { return true }, 0)
		ctx := context.Background()

		optimized, err := optimizer.Compile(ctx, createProgram(), trainDataset, dummyMetric)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("MaxBootstrapped Large", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]any, _ context.Context) bool {
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
		optimizer := NewBootstrapFewShot(func(_, _ map[string]any, _ context.Context) bool { return false }, 2)
		ctx := context.Background()
		optimized, _ := optimizer.Compile(ctx, createProgram(), trainDataset, dummyMetric)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("CompileWithScorer", func(t *testing.T) {
		type callerKey struct{}
		ctx := context.WithValue(context.Background(), callerKey{}, true)
		// Q1 scores 1, Q2 scores 0.5, and Q3 cannot be scored.
		scorer := core.ScorerFunc(func(ctx context.Context, example core.Example, _ map[string]any) (float64, error) {
			if ctx.Value(callerKey{}) == nil {
				t.Error("scorer did not receive the caller's context")
			}
			if _, mixed := example.Inputs["answer"]; mixed || example.Outputs["answer"] == nil {
				t.Errorf("scorer received example %v; want inputs and expected outputs kept apart", example)
			}
			switch example.Inputs["question"] {
			case "Q1":
				return 1, nil
			case "Q2":
				return 0.5, nil
			}
			return 0, fmt.Errorf("judge unavailable")
		})
		// The constructor metric would keep everything; the scorer decides.
		keepAll := func(_, _ map[string]any, _ context.Context) bool { return true }
		questions := func(optimized core.Program) []any {
			var kept []any
			for _, demo := range optimized.Modules["predict"].(*modules.Predict).Demos {
				kept = append(kept, demo.Inputs["question"])
			}
			return kept
		}

		optimizer := NewBootstrapFewShot(keepAll, 100)
		optimized, err := optimizer.CompileWithScorer(ctx, createProgram(), trainExamples, scorer)
		require.NoError(t, err)
		assert.ElementsMatch(t, []any{"Q1", "Q2"}, questions(optimized), "any score above zero is kept; an unscorable example is not")

		optimizer.MetricThreshold = 1
		optimized, err = optimizer.CompileWithScorer(ctx, createProgram(), trainExamples, scorer)
		require.NoError(t, err)
		assert.Equal(t, []any{"Q1"}, questions(optimized))

		_, err = optimizer.CompileWithScorer(ctx, createProgram(), trainExamples, core.MetricScorer(nil))
		assert.Error(t, err)

		// A score of exactly zero is kept only by a negative threshold.
		zero := core.ScorerFunc(func(context.Context, core.Example, map[string]any) (float64, error) { return 0, nil })
		optimizer.MetricThreshold = 0
		optimized, err = optimizer.CompileWithScorer(ctx, createProgram(), trainExamples, zero)
		require.NoError(t, err)
		assert.Empty(t, questions(optimized))
		optimizer.MetricThreshold = -1
		optimized, err = optimizer.CompileWithScorer(ctx, createProgram(), trainExamples, zero)
		require.NoError(t, err)
		assert.Len(t, questions(optimized), len(trainExamples))
	})

	t.Run("CompileWithScorer reports cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		// The scorer keeps Q1, then the run is cancelled while scoring.
		scorer := core.ScorerFunc(func(ctx context.Context, example core.Example, _ map[string]any) (float64, error) {
			if example.Inputs["question"] == "Q1" {
				return 1, nil
			}
			cancel()
			return 0, ctx.Err()
		})
		_, err := NewBootstrapFewShot(nil, 100).CompileWithScorer(ctx, createProgram(), trainExamples, scorer)
		assert.ErrorIs(t, err, context.Canceled)
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
			boolMetric := func(example, prediction map[string]any, ctx context.Context) bool {
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

			boolMetric := func(example, prediction map[string]any, ctx context.Context) bool {
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

			boolMetric := func(example, prediction map[string]any, ctx context.Context) bool {
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
		metricFunc  func(example, prediction map[string]any, ctx context.Context) bool
		description string
	}{
		{
			"AlwaysAccept",
			func(example, prediction map[string]any, ctx context.Context) bool { return true },
			"Metric that always accepts predictions",
		},
		{
			"AlwaysReject",
			func(example, prediction map[string]any, ctx context.Context) bool { return false },
			"Metric that always rejects predictions",
		},
		{
			"Selective50",
			func(example, prediction map[string]any, ctx context.Context) bool {
				score := testutil.BenchmarkAccuracyMetric(example, prediction)
				return score > 0.5
			},
			"Metric that accepts when accuracy > 50%",
		},
		{
			"Selective80",
			func(example, prediction map[string]any, ctx context.Context) bool {
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

			boolMetric := func(example, prediction map[string]any, ctx context.Context) bool {
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

// TestBootstrapFewShotBootstrapsChainOfThought checks that demonstrations
// reach the Predict module inside a composite. Before the fix the optimizer
// saw no module with demonstrations and returned without calling the teacher.
func TestBootstrapFewShotBootstrapsChainOfThought(t *testing.T) {
	mockLLM := setupTestMockLLM(t)
	cot := modules.NewChainOfThought(core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	))
	student := core.NewProgramWithForwardFactory(
		map[string]core.Module{"cot": cot},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				return modules["cot"].Process(ctx, inputs)
			}
		},
	)
	train := []core.Example{
		{Inputs: map[string]any{"question": "Capital of France?"}, Outputs: map[string]any{"answer": "Paris"}},
		{Inputs: map[string]any{"question": "Capital of Germany?"}, Outputs: map[string]any{"answer": "Berlin"}},
		{Inputs: map[string]any{"question": "Capital of Italy?"}, Outputs: map[string]any{"answer": "Rome"}},
	}

	optimizer := NewBootstrapFewShot(nil, 2)
	compiled, err := optimizer.CompileWithScorer(core.WithExecutionState(context.Background()), student, train,
		core.ScorerFunc(func(context.Context, core.Example, map[string]any) (float64, error) { return 1, nil }))
	require.NoError(t, err)

	compiledCoT, ok := compiled.Modules["cot"].(*modules.ChainOfThought)
	require.True(t, ok)
	demos := compiledCoT.Predict.GetDemos()
	require.Len(t, demos, 2)
	assert.Equal(t, "Capital of France?", demos[0].Inputs["question"])
	assert.Equal(t, "Capital of Germany?", demos[1].Inputs["question"])
	mockLLM.AssertNumberOfCalls(t, "Generate", 2)
	assert.Empty(t, cot.Predict.GetDemos(), "the student program must not be modified")
}

func TestBootstrapFewShotKeepsLookingPastRejectedExamples(t *testing.T) {
	mockLLM := setupTestMockLLM(t)
	var train []core.Example
	for i := range 6 {
		train = append(train, core.Example{
			Inputs:  map[string]any{"question": fmt.Sprintf("q%d", i)},
			Outputs: map[string]any{"answer": "Paris"},
		})
	}
	// The first two examples fail the metric; the next two pass.
	scorer := core.ScorerFunc(func(_ context.Context, example core.Example, _ map[string]any) (float64, error) {
		if question := example.Inputs["question"]; question == "q0" || question == "q1" {
			return 0, nil
		}
		return 1, nil
	})

	compiled, err := NewBootstrapFewShot(nil, 2).CompileWithScorer(
		core.WithExecutionState(context.Background()), createProgram(), train, scorer)
	require.NoError(t, err)

	demos := compiled.Modules["predict"].(*modules.Predict).GetDemos()
	require.Len(t, demos, 2)
	assert.Equal(t, "q2", demos[0].Inputs["question"])
	assert.Equal(t, "q3", demos[1].Inputs["question"])
	mockLLM.AssertNumberOfCalls(t, "Generate", 4)
}

// bootstrapQuestions returns the questions of the demos on the compiled
// single-predictor program.
func bootstrapQuestions(t *testing.T, compiled core.Program) []any {
	t.Helper()
	var questions []any
	for _, demo := range compiled.Modules["predictor"].(*modules.Predict).GetDemos() {
		questions = append(questions, demo.Inputs["question"])
	}
	return questions
}

func keepEverything(_, _ map[string]any, _ context.Context) bool { return true }

// bootstrapTrainset returns n legacy train examples asking q<i>.
func bootstrapTrainset(n int) []map[string]any {
	trainset := make([]map[string]any, n)
	for i, example := range scriptQAExamples(n) {
		trainset[i] = map[string]any{"question": example.Inputs["question"], "answer": example.Outputs["answer"]}
	}
	return trainset
}

func TestBootstrapFewShotKeepsModuleLLMWithoutTeacherLLM(t *testing.T) {
	answering := func(answer string) *scriptLLM {
		return &scriptLLM{behave: func(_, _ string) (string, error) { return "answer: " + answer, nil }}
	}
	moduleLLM, defaultLLM, teacherLLM := answering("from-module"), answering("from-default"), answering("from-teacher")
	newProgram := func() core.Program {
		predictor := modules.NewPredict(scriptQASignature("base"))
		predictor.SetLLM(moduleLLM)
		return coproSinglePredictorProgram(predictor)
	}
	compile := func(t *testing.T) any {
		program := newProgram()
		compiled, err := NewBootstrapFewShot(keepEverything, 2).CompileLegacy(context.Background(), program, program, bootstrapTrainset(3))
		require.NoError(t, err)
		demos := compiled.Modules["predictor"].(*modules.Predict).GetDemos()
		require.Len(t, demos, 2)
		return demos[0].Outputs["answer"]
	}

	useScriptLLM(t, defaultLLM)
	assert.Equal(t, "from-module", compile(t), "the default LLM does not replace a module's own LLM")
	assert.Empty(t, defaultLLM.sent(""))

	core.GlobalConfig.DefaultLLM = nil
	assert.Equal(t, "from-module", compile(t), "no default LLM is needed when modules have their own")

	core.GlobalConfig.TeacherLLM = teacherLLM
	assert.Equal(t, "from-teacher", compile(t), "a configured teacher LLM is used")
}

func TestBootstrapFewShotSkipsFailedTeacherCalls(t *testing.T) {
	compile := func(t *testing.T, fails func(question string) bool) (core.Program, error) {
		llm := &scriptLLM{behave: func(_, prompt string) (string, error) {
			question := promptInput(prompt, "question")
			if fails(question) {
				return "", fmt.Errorf("transient provider error")
			}
			return "answer: gold-" + question, nil
		}}
		useScriptLLM(t, llm)
		program := coproSinglePredictorProgram(modules.NewPredict(scriptQASignature("base")))
		return NewBootstrapFewShot(keepEverything, 4).CompileLegacy(context.Background(), program, program, bootstrapTrainset(10))
	}

	t.Run("one failure is skipped", func(t *testing.T) {
		compiled, err := compile(t, func(question string) bool { return question == "q1" })
		require.NoError(t, err)
		assert.Equal(t, []any{"q0", "q2", "q3", "q4"}, bootstrapQuestions(t, compiled))
	})

	t.Run("a teacher that always fails is an error", func(t *testing.T) {
		_, err := compile(t, func(string) bool { return true })
		assert.ErrorContains(t, err, "every teacher call failed")
		assert.ErrorContains(t, err, "transient provider error")
	})
}

func TestBootstrapFewShotRecompileReplacesExistingDemos(t *testing.T) {
	llm := &scriptLLM{behave: func(_, prompt string) (string, error) {
		return "answer: gold-" + promptInput(prompt, "question"), nil
	}}
	useScriptLLM(t, llm)
	predictor := modules.NewPredict(scriptQASignature("base"))
	stale := []core.Example{
		{Inputs: map[string]any{"question": "old1"}, Outputs: map[string]any{"answer": "stale1"}},
		{Inputs: map[string]any{"question": "old2"}, Outputs: map[string]any{"answer": "stale2"}},
	}
	predictor.SetDemos(stale)
	program := coproSinglePredictorProgram(predictor)

	compiled, err := NewBootstrapFewShot(keepEverything, 2).CompileLegacy(context.Background(), program, program, bootstrapTrainset(10))

	require.NoError(t, err)
	assert.Equal(t, []any{"q0", "q1"}, bootstrapQuestions(t, compiled))
	assert.Equal(t, stale, predictor.GetDemos(), "the student program must not be modified")
}

func TestBootstrapFewShotGivesEachPredictorItsOwnDemos(t *testing.T) {
	llm := &scriptLLM{behave: func(_, prompt string) (string, error) {
		if question := promptInput(prompt, "question"); question != "" {
			return "<response><rationale>why-" + question + "</rationale><query>search-" + question + "</query></response>", nil
		}
		return "answer: gold-" + strings.TrimPrefix(promptInput(prompt, "query"), "search-"), nil
	}}
	useScriptLLM(t, llm)
	// The first stage is a ChainOfThought, so its predictor is nested.
	rewrite := modules.NewChainOfThought(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "query", Prefix: "query:"}}},
	))
	answer := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "query"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer", Prefix: "answer:"}}},
	))
	program := core.NewProgramWithForwardFactory(map[string]core.Module{"a_rewrite": rewrite, "b_answer": answer},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				rewritten, err := modules["a_rewrite"].Process(ctx, inputs)
				if err != nil {
					return nil, err
				}
				// The second stage also receives a key its signature lacks.
				return modules["b_answer"].Process(ctx, map[string]any{"query": rewritten["query"], "question": ""})
			}
		})
	// The metric rejects q1, so neither predictor learns from that run.
	keep := func(example, _ map[string]any, _ context.Context) bool { return example["question"] != "q1" }

	compiled, err := NewBootstrapFewShot(keep, 2).CompileLegacy(context.Background(), program, program, bootstrapTrainset(4))

	require.NoError(t, err)
	assert.Equal(t, []core.Example{
		{Inputs: map[string]any{"question": "q0"}, Outputs: map[string]any{"rationale": "why-q0", "query": "search-q0"}},
		{Inputs: map[string]any{"question": "q2"}, Outputs: map[string]any{"rationale": "why-q2", "query": "search-q2"}},
	}, compiled.Modules["a_rewrite"].(*modules.ChainOfThought).Predict.GetDemos())
	assert.Equal(t, []core.Example{
		{Inputs: map[string]any{"query": "search-q0"}, Outputs: map[string]any{"answer": "gold-q0"}},
		{Inputs: map[string]any{"query": "search-q2"}, Outputs: map[string]any{"answer": "gold-q2"}},
	}, compiled.Modules["b_answer"].(*modules.Predict).GetDemos())
	assert.Empty(t, rewrite.Predict.GetDemos(), "the student program must not be modified")

	llm.prompts = nil
	result, err := compiled.Execute(context.Background(), map[string]any{"question": "new"})
	require.NoError(t, err)
	assert.Equal(t, "gold-new", result["answer"])
	prompts := llm.sent("")
	require.Len(t, prompts, 2)
	for _, prompt := range prompts {
		assert.NotContains(t, prompt, "<nil>", "every demonstration field is filled")
	}
	assert.Contains(t, prompts[0], "question: q0\nrationale: why-q0\nquery: search-q0\n")
	assert.Contains(t, prompts[1], "query: search-q0\nanswer: gold-q0\n")
}

func TestBootstrapFewShotRequiresMatchingTeacherPredictors(t *testing.T) {
	llm := &scriptLLM{behave: func(_, prompt string) (string, error) {
		if question := promptInput(prompt, "question"); question != "" {
			return "query: search-" + question, nil
		}
		return "answer: gold-" + strings.TrimPrefix(promptInput(prompt, "query"), "search-"), nil
	}}
	useScriptLLM(t, llm)
	student := scriptTwoStageProgram(llm)
	optimizer := NewBootstrapFewShot(keepEverything, 2)

	t.Run("a teacher with other predictors is rejected", func(t *testing.T) {
		teacher := coproSinglePredictorProgram(modules.NewPredict(scriptQASignature("base")))
		_, err := optimizer.CompileLegacy(context.Background(), student, teacher, bootstrapTrainset(4))
		assert.ErrorContains(t, err, "must have the same names")
		assert.Empty(t, llm.sent(""), "the mismatch is rejected before any teacher call")
	})

	t.Run("single predictors pair up under different names", func(t *testing.T) {
		single := modules.NewPredict(scriptQASignature("base"))
		single.SetLLM(&scriptLLM{behave: func(_, prompt string) (string, error) {
			return "answer: gold-" + promptInput(prompt, "question"), nil
		}})
		teacher := core.NewProgramWithForwardFactory(map[string]core.Module{"teacher": single},
			func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
				return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
					return modules["teacher"].Process(ctx, inputs)
				}
			})
		compiled, err := optimizer.CompileLegacy(context.Background(),
			coproSinglePredictorProgram(modules.NewPredict(scriptQASignature("base"))), teacher, bootstrapTrainset(4))
		require.NoError(t, err)
		assert.Equal(t, []any{"q0", "q1"}, bootstrapQuestions(t, compiled))
	})
}
