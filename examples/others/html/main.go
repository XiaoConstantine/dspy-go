package main

import (
	"context"
	"flag"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

// SimpleDataset implements the core.Dataset interface for MIPRO.
type SimpleDataset struct {
	Examples []core.Example
	Index    int
}

func NewSimpleDataset(examples []core.Example) *SimpleDataset {
	return &SimpleDataset{
		Examples: examples,
		Index:    0,
	}
}

func (d *SimpleDataset) Next() (core.Example, bool) {
	if d.Index < len(d.Examples) {
		example := d.Examples[d.Index]
		d.Index++
		return example, true
	}
	return core.Example{}, false
}

func (d *SimpleDataset) Reset() {
	d.Index = 0
}

func cleanMarkdownCodeBlocks(input string) string {
	// Remove markdown code fence markers
	input = strings.TrimPrefix(input, "```html")
	input = strings.TrimPrefix(input, "```")
	input = strings.TrimSuffix(input, "```")
	input = strings.TrimSpace(input)
	return input
}

func createMetricFunc() func(example, prediction map[string]interface{}, ctx context.Context) bool {
	// Create a signature for our judge module
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "document"}},
			{Field: core.Field{Name: "predicted_metadata"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "rationale", Prefix: "rationale"}},
			{Field: core.Field{Name: "is_correct", Prefix: "is_correct"}},
		},
	).WithInstruction(`You are evaluating whether the predicted metadata correctly extracts information from the HTML document.
    First, analyze the document and identify the key information.
    Then, compare this with the predicted metadata.
    Explain your reasoning step by step, and finally determine if the prediction is correct.`)

	// Create a ChainOfThought module with this signature
	judge := modules.NewChainOfThought(signature)

	// Return a metric function that uses this module
	return func(example, prediction map[string]interface{}, ctx context.Context) bool {
		// Create inputs for the judge module
		inputs := map[string]interface{}{
			"document":           example["document"],
			"predicted_metadata": prediction,
		}

		// Execute the judge module
		result, err := judge.Process(ctx, inputs)
		if err != nil {
			// Handle error - in a metric, we might want to log and return false
			logging.GetLogger().Error(ctx, "Error in metric evaluation: %v", err)
			return false
		}

		// Extract and convert the is_correct field to a boolean
		isCorrectStr, ok := result["is_correct"].(string)
		if !ok {
			logging.GetLogger().Error(ctx, "Invalid is_correct type in result: %T", result["is_correct"])
			return false
		}

		// Parse the boolean string - could be more robust with regex or trimming
		isCorrect := strings.Contains(strings.ToLower(isCorrectStr), "true") ||
			strings.Contains(strings.ToLower(isCorrectStr), "yes")

		return isCorrect
	}
}

func main() {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)
	apiKey := flag.String("api-key", "", "Anthropic API Key")
	flag.Parse()
	ctx := core.WithExecutionState(context.Background())

	llms.EnsureFactory()

	// Configure the default LLM
	err := core.ConfigureDefaultLLM(*apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to configure default LLM: %v", err)
	}

	// Q1: Take a HTML page and generate structured data?
	// Create a signature for extracting structured data from HTML
	extractSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "document"}}},
		[]core.OutputField{
			{Field: core.NewField("title")},
			{Field: core.NewField("headings")},
			{Field: core.NewField("entity_info")},
		},
	)

	// Create a ChainOfThought module for extraction
	extract := modules.NewChainOfThought(extractSignature)

	// Q2: But I don't have HTML documents to use for optimization.
	// Create a module to synthesize topics
	synthesizeTopicsSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "count"}}},
		[]core.OutputField{{Field: core.NewField("random_technical_topics")}},
	).WithInstruction(`Generate a list of diverse technical topics.
Each topic should be something that could have a web page with headings and entities.
Format your response as a list, DO NOT wrap response in json format`)
	synthesize_topics := modules.NewPredict(synthesizeTopicsSignature)

	// Create a module to synthesize HTML documents from topics
	synthesizeDocSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "topic"}}},
		[]core.OutputField{{Field: core.NewField("html_document")}},
	).WithInstruction(`Generate a realistic HTML document about the given topic.
The HTML should include a title tag in the head, multiple heading tags (h1, h2, etc.),
several paragraphs of content, and references to entity names (people, companies, technologies).
Use proper HTML structure with html, head, and body tags.`)
	synthesize_doc := modules.NewPredict(synthesizeDocSignature)

	// Generate topics
	topicsResult, err := synthesize_topics.Process(ctx, map[string]interface{}{
		"count": 5,
	})
	if err != nil {
		logger.Fatalf(ctx, "Failed to generate topics: %v", err)
	}
	// Parse topics from the result
	var topics []string

	if topicsContent, ok := topicsResult["random_technical_topics"]; ok && topicsContent != nil {
		logger.Info(ctx, "\nTopic content: %v", topicsContent)

		// Handle the case where it's a string
		if topicsString, ok := topicsContent.(string); ok {
			logger.Info(ctx, "Parsing topics from multiline string...")

			// Split by lines and take each non-empty line as a topic
			lines := strings.Split(topicsString, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				// Skip empty lines and lines that appear to be field names
				if line != "" && !strings.HasPrefix(strings.ToLower(line), "random_technical_topics") {
					topics = append(topics, line)
				}
			}
		} else if topicsList, ok := topicsContent.([]interface{}); ok {
			// Handle the case where it's already a list
			logger.Info(ctx, "Topics already in list format, extracting...")
			for _, t := range topicsList {
				if topic, ok := t.(string); ok {
					topics = append(topics, topic)
				}
			}
		}
	}
	// Generate HTML documents for each topic
	var trainingExamples []core.Example
	for _, t := range topics {
		docResult, err := synthesize_doc.Process(ctx, map[string]interface{}{
			"topic": t,
		})
		if err != nil {
			logger.Error(ctx, "Failed to generate document for topic %v: %v", t, err)
			continue
		}
		// Extract HTML document from result
		if htmlDoc, ok := docResult["html_document"]; ok {
			htmlString, ok := htmlDoc.(string)
			if ok && htmlString != "" {
				// Clean up markdown code fences if present
				htmlString = cleanMarkdownCodeBlocks(htmlString)

				// Create a training example with this document
				example := core.Example{
					Inputs: map[string]interface{}{
						"document": htmlString,
					},
					// We're leaving Outputs empty since we want the model to generate them
					Outputs: make(map[string]interface{}),
				}

				trainingExamples = append(trainingExamples, example)
				logger.Info(ctx, "Created training example from %s document (%d chars)",
					t, len(htmlString))
			} else {
				logger.Error(ctx, "Generated document was empty or not a string for %s", t)
			}
		} else {
			logger.Error(ctx, "No html_document field in result for %s: %v", t, docResult)
		}
	}
	// Make sure we have at least one example
	if len(trainingExamples) == 0 {
		logger.Fatal(ctx, "Failed to generate any valid training examples")
	}

	logger.Info(ctx, "Created %d training examples\n", len(trainingExamples))

	metricFunc := createMetricFunc()

	// Q3: How can I teach LLM to be better at this task?
	// Define a metric function to evaluate extraction quality
	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
		if metricFunc(example, prediction, ctx) {
			return 1.0
		}
		logger := logging.GetLogger()
		logger.Info(ctx, "======= METRIC FALLBACK EVALUATION STARTED =======")
		_, ok := example["document"].(string)
		if !ok {
			logger.Error(ctx, "Document not found in example")
			return 0.0
		}
		// Check if required fields exist
		title, hasTitle := prediction["title"].(string)
		headings, hasHeadings := prediction["headings"].(string)
		entityInfo, hasEntityInfo := prediction["entity_info"].(string)

		if !hasTitle || !hasHeadings || !hasEntityInfo {
			logger.Warn(ctx, "Missing expected fields in prediction")
			return 0.0
		}

		// Simple presence checks
		var score = 0.0

		// If we have a title that's not a placeholder
		if hasTitle && !strings.Contains(strings.ToLower(title), "no title") {
			score += 0.3
			logger.Debug(ctx, "Added 0.3 for having title: %s", title)
		}

		// If we have headings that aren't placeholders
		if hasHeadings && !strings.Contains(strings.ToLower(headings), "no headings") {
			score += 0.3
			logger.Debug(ctx, "Added 0.3 for having headings: %s", headings)
		}

		// If we have entity info that's not a placeholder
		if hasEntityInfo && !strings.Contains(strings.ToLower(entityInfo), "no entities") {
			score += 0.4
			logger.Debug(ctx, "Added 0.4 for having entity info: %s", entityInfo)
		}

		logger.Debug(ctx, "Final metric score: %.2f", score)

		return score
	}

	dataset := NewSimpleDataset(trainingExamples)

	// Create a program using the extract module
	program := core.NewProgram(
		map[string]core.Module{"extract": extract},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return extract.Process(ctx, inputs)
		},
	)

	// Optimize the extraction module using MIPROv2
	optimizer := optimizers.NewMIPRO(metric,
		optimizers.WithNumTrials(10),
		optimizers.WithMaxBootstrappedDemos(5),
		optimizers.WithVerbose(true),
	)
	coreMetric := func(expected, actual map[string]interface{}) float64 {
		return metric(expected, actual, ctx)
	}

	optimized_program, err := optimizer.Compile(ctx, program, dataset, coreMetric)
	if err != nil {
		logger.Fatalf(ctx, "Optimization failed: %v", err)
	}

	// Q4: How do I use this now?
	// Use the optimized program to extract data from a new HTML document
	result, err := optimized_program.Execute(ctx, map[string]interface{}{
		"document": `<html lang="en">...</html>`,
	})

	if err != nil {
		logger.Error(ctx, "Extraction failed: %v", err)
	}

	logger.Info(ctx, "Title: %v\n", result["title"])
	logger.Info(ctx, "Headings: %v\n", result["headings"])
	logger.Info(ctx, "Entity Info: %v\n", result["entity_info"])
}
