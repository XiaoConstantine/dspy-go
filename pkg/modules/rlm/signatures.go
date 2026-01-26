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

QUERY FUNCTIONS (choose based on context size):
- Query(prompt string) string: Auto-prepends FULL context to prompt. Use for small contexts (<50K chars).
- QueryRaw(prompt string) string: Sends prompt AS-IS without context. Use when you've already included context in prompt.
- QueryWith(contextSlice, prompt string) string: Prepends ONLY the slice you specify. Use for large contexts.
- QueryBatched(prompts []string) []string: Parallel Query() calls. Each gets full context prepended.
- QueryBatchedRaw(prompts []string) []string: Parallel QueryRaw() calls. No context prepended.

CONTEXT ACCESS FUNCTIONS:
- FindRelevant(query string, topK int) []string: Semantic search - returns top-K relevant chunks
- GetChunk(id int) string: Get chunk by ID (chunks are ~4KB each)
- GetContext(startLine, endLine int) string: Get specific line range
- ChunkCount() int: Number of chunks
- LineCount() int: Number of lines

COMPLETION FUNCTIONS:
- FINAL(value string): Signal completion with a direct value
- FINAL_VAR(varName string): Signal completion with a variable's value

STANDARD GO: fmt, strings, regexp, strconv, encoding/json, sort

CRITICAL - CONTEXT SIZE HANDLING:
- Context < 50K chars: Use Query(prompt + context) directly
- Context 50K-200K chars: Use QueryWith(context, prompt) to control what's sent
- Context > 200K chars: MUST chunk! Use FindRelevant() or split manually, then QueryWith() or QueryRaw()

WARNING: Query() auto-prepends the ENTIRE context (~4 chars = 1 token). For a 500K char context:
  Query("analyze: " + context[:100000])  // WRONG! Sends 500K + 100K = 600K chars (overflow!)
  QueryWith(context[:100000], "analyze") // CORRECT! Sends only 100K chars

LARGE CONTEXT PATTERN:
// For contexts > 200K chars, chunk and query separately:
chunks := FindRelevant("main entry points", 5)  // Get relevant chunks
var results []string
for _, chunk := range chunks {
    r := QueryWith(chunk, "Find entry points in this code")
    results = append(results, r)
}
combined := strings.Join(results, "\n")
answer := QueryRaw("Synthesize these findings: " + combined)
FINAL(answer)

CRITICAL CODE RULES (violations cause errors):
- DO NOT use 'import' statements - packages are already imported
- DO NOT use 'type' declarations - use map[string]interface{} or inline structs
- DO NOT use 'func' declarations at top level - use closures if needed
- Write ONLY executable statements (assignments, function calls, loops, conditionals)
- EVERY variable must be declared with := before use
- Keep code blocks SHORT (under 15 lines)

CRITICAL - SIGNALING COMPLETION:
When you have the answer, IMMEDIATELY call FINAL() in the SAME code block!

Actions:
- explore: Write code to examine the context (len, preview, structure)
- query: Write code to call Query/QueryWith/QueryRaw for analysis
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
		// Example demonstrating FINAL() called immediately after Query()
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 5000 chars",
				"query":        "What is the label: correct or incorrect?",
				"history":      "Explored: Length=5000, Preview shows labeled data",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "Small context that fits in one Query call. I'll ask the sub-LLM to determine the label and IMMEDIATELY call FINAL with the result.",
				"action":    "query",
				"code": `answer := Query("What is the label in this text? Return ONLY 'correct' or 'incorrect': " + context)
FINAL(answer)`,
				"answer": "",
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
				"reasoning": "At 1.2M chars, I MUST use QueryWith or QueryRaw - Query() would overflow! I'll split by document separator and use QueryWith for each batch.",
				"action":    "query",
				"code": `docs := strings.Split(context, "---")
fmt.Println("Found", len(docs), "documents")

var results []string
var batch string
for _, doc := range docs {
    if len(batch)+len(doc) > 150000 && batch != "" {
        // Use QueryWith to send ONLY this batch, not full context
        r := QueryWith(batch, "Identify main themes in these documents")
        results = append(results, r)
        batch = ""
    }
    batch += doc + "\n---\n"
}
if batch != "" {
    r := QueryWith(batch, "Identify main themes in these documents")
    results = append(results, r)
}
for i, r := range results { fmt.Printf("Batch %d: %s\n", i, r) }`,
				"answer": "",
			},
		},
		// Example using FindRelevant for semantic search on large context
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Find all authentication-related code",
				"history":      "Explored: 800K chars of Go source code, too large for single Query",
				"repl_state":   "context: <loaded>",
			},
			Outputs: map[string]interface{}{
				"reasoning": "At 800K chars, I cannot use Query() as it would overflow. I'll use FindRelevant() to get semantically relevant chunks, then QueryWith() to analyze each.",
				"action":    "query",
				"code": `// Find chunks most relevant to authentication
chunks := FindRelevant("authentication login password session token", 10)
fmt.Println("Found", len(chunks), "relevant chunks")

var authCode []string
for i, chunk := range chunks {
    // Use QueryWith to analyze just this chunk
    result := QueryWith(chunk, "Extract any authentication-related code (login, password, session, token handling). Return the code snippets or 'none' if not found.")
    if result != "none" && result != "" {
        authCode = append(authCode, fmt.Sprintf("Chunk %d:\n%s", i, result))
    }
}
fmt.Println("Auth code found in", len(authCode), "chunks")
for _, code := range authCode { fmt.Println(code) }`,
				"answer": "",
			},
		},
		// Example showing synthesis with QueryRaw
		{
			Inputs: map[string]interface{}{
				"context_info": "string, 800000 chars",
				"query":        "Find all authentication-related code",
				"history":      "Found auth code in 4 chunks: JWT token validation, password hashing, session middleware, OAuth handler",
				"repl_state":   "context: <loaded>, authCode: [4 code snippets]",
			},
			Outputs: map[string]interface{}{
				"reasoning": "I have 4 authentication code snippets. Now I'll synthesize them using QueryRaw (no context needed) and immediately FINAL the answer.",
				"action":    "query",
				"code": `summary := strings.Join(authCode, "\n\n---\n\n")
// Use QueryRaw since we're passing our own content, not the original context
answer := QueryRaw("Summarize these authentication code findings into a coherent explanation:\n\n" + summary)
FINAL(answer)`,
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

