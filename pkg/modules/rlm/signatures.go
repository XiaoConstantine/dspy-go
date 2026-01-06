package rlm

import (
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// =============================================================================
// DSPy-Native Signature Definitions
// =============================================================================

// RLMSignature creates the main RLM module signature.
// This is the outer interface: takes context + query, returns answer.
func RLMSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context",
				core.WithDescription("The context data to analyze (can be very large)"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("The question to answer about the context"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("answer",
				core.WithDescription("The final answer to the query"),
			)},
		},
	).WithInstruction("Analyze the context using iterative code exploration to answer the query.")
}

// IterationSignature defines the signature for each RLM iteration.
// This powers the inner loop where the LLM decides what to do next.
func IterationSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context_info",
				core.WithDescription("Summary of the context (type, size, preview)"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("The original question to answer"),
			)},
			{Field: core.NewField("history",
				core.WithDescription("Previous code executions and their results"),
			)},
			{Field: core.NewField("repl_state",
				core.WithDescription("Current REPL variable state"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("reasoning",
				core.WithDescription("Step-by-step thinking about what to do next"),
				core.WithCustomPrefix("Reasoning:"),
			)},
			{Field: core.NewField("action",
				core.WithDescription("The action type: 'explore', 'query', 'compute', or 'final'"),
				core.WithCustomPrefix("Action:"),
			)},
			{Field: core.NewField("code",
				core.WithDescription("Go code to execute (if action is explore/query/compute)"),
				core.WithCustomPrefix("Code:"),
			)},
			{Field: core.NewField("answer",
				core.WithDescription("The final answer (if action is 'final')"),
				core.WithCustomPrefix("Answer:"),
			)},
		},
	).WithInstruction(`You are exploring a large context using a Go REPL. Available functions:
- Query(prompt string) string: Query a sub-LLM with the prompt
- QueryBatched(prompts []string) []string: Query multiple prompts in parallel
- Standard Go: fmt, strings, regexp, strconv

Sub-LLM Capacity: Sub-LLMs can handle ~500K characters. For efficiency, batch ~200K characters per Query call.
IMPORTANT: REPL outputs are truncated. Use Query() to analyze full content rather than printing large outputs.
Make sure to explicitly look through the entire context before answering.

Actions:
- explore: Write code to examine the context (len, preview, structure)
- query: Write code to call Query/QueryBatched for analysis
- compute: Write code to process/combine results
- final: Provide the answer (no more code needed)

Always show your reasoning before deciding on an action.`)
}

// SubQuerySignature defines the signature for sub-LLM queries.
// This is used by Query() and QueryBatched() internally.
func SubQuerySignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("prompt",
				core.WithDescription("The analysis prompt"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("response",
				core.WithDescription("The analysis result"),
			)},
		},
	).WithInstruction("Analyze the given content and provide a concise, accurate response.")
}

// ChunkAnalysisSignature for analyzing individual chunks of large contexts.
func ChunkAnalysisSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("chunk",
				core.WithDescription("A portion of the larger context"),
			)},
			{Field: core.NewField("query",
				core.WithDescription("What to look for in this chunk"),
			)},
			{Field: core.NewField("chunk_index",
				core.WithDescription("Which chunk this is (e.g., '3 of 10')"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("findings",
				core.WithDescription("Relevant findings from this chunk"),
			)},
			{Field: core.NewField("confidence",
				core.WithDescription("Confidence level: high, medium, low"),
			)},
		},
	).WithInstruction("Analyze this chunk for information relevant to the query. Be concise but thorough.")
}

// SynthesisSignature for combining results from multiple chunk analyses.
func SynthesisSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("query",
				core.WithDescription("The original question"),
			)},
			{Field: core.NewField("chunk_results",
				core.WithDescription("Results from analyzing each chunk"),
			)},
		},
		[]core.OutputField{
			{Field: core.NewField("synthesis",
				core.WithDescription("Combined analysis of all chunks"),
			)},
			{Field: core.NewField("answer",
				core.WithDescription("The final answer based on all evidence"),
			)},
		},
	).WithInstruction("Synthesize the chunk analysis results into a coherent answer.")
}

// =============================================================================
// Example Demos for Few-Shot Learning
// =============================================================================

// IterationDemos provides few-shot examples for the iteration module.
func IterationDemos() []core.Example {
	return []core.Example{
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "This is a large context (150K chars). I should first explore its structure before diving into analysis.",
				"action":    "explore",
				"code":      "fmt.Println(\"Length:\", len(context))\nfmt.Println(\"Preview:\", context[:500])",
				"answer":    "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "Explored: Length=150000, Preview shows JSON array of messages",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Context is too large to process at once. I'll chunk it and use QueryBatched for parallel analysis.",
				"action":    "query",
				"code": `chunkSize := len(context) / 5
var prompts []string
for i := 0; i < 5; i++ {
    start, end := i*chunkSize, (i+1)*chunkSize
    if i == 4 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Find any secret codes in this text. Return ONLY the code if found, or 'none' if not found: %s", context[start:end]))
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Chunk %d: %s\n", i, r) }`,
				"answer": "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 150000 chars",
				"query":        "What is the secret code?",
				"history":      "Chunk 0: none\nChunk 1: ALPHA-7892\nChunk 2: none\nChunk 3: none\nChunk 4: none",
				"repl_state":   "context: <loaded>, results: [none, ALPHA-7892, none, none, none]",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Found the secret code in chunk 1: ALPHA-7892. This is the answer.",
				"action":    "final",
				"code":      "",
				"answer":    "ALPHA-7892",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Count all error messages in the logs",
				"history":      "",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Context is 800K chars. Sub-LLMs handle ~500K, so I'll use regex to filter first, then batch ~200K per query for efficiency.",
				"action":    "explore",
				"code": `errorRe := regexp.MustCompile("(?i)error|exception|failed")
matches := errorRe.FindAllString(context, -1)
fmt.Println("Total potential errors:", len(matches))
fmt.Println("Sample:", context[:1000])`,
				"answer": "",
			},
		},
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 1200000 chars",
				"query":        "Summarize the main themes in this document collection",
				"history":      "Explored: 1.2M chars, appears to be multiple documents separated by ---",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "At 1.2M chars, I need to chunk strategically. I'll split by document separator and batch ~200K chars per Query call for efficiency.",
				"action":    "query",
				"code": `docs := strings.Split(context, "---")
fmt.Println("Found", len(docs), "documents")

var prompts []string
var batch string
for _, doc := range docs {
    if len(batch)+len(doc) > 200000 && batch != "" {
        prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
        batch = ""
    }
    batch += doc + "\n---\n"
}
if batch != "" {
    prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Batch %d themes: %s\n", i, r) }`,
				"answer": "",
			},
		},
	}
}

// SubQueryDemos provides few-shot examples for sub-LLM queries.
func SubQueryDemos() []core.Example {
	return []core.Example{
		{
			Inputs: map[string]interface{}{
				"prompt": "Find any email addresses in this text: Contact us at support@example.com or sales@example.com",
			},
			Outputs: map[string]interface{}{
				"response": "support@example.com, sales@example.com",
			},
		},
		{
			Inputs: map[string]interface{}{
				"prompt": "What is the main topic of this text? Return one word: The quick brown fox jumps over the lazy dog.",
			},
			Outputs: map[string]interface{}{
				"response": "animals",
			},
		},
	}
}

