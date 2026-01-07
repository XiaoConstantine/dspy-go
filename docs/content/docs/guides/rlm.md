---
title: "RLM Module"
description: "Recursive Language Model for large context exploration"
summary: "Enable LLMs to programmatically explore contexts exceeding token limits through a sandboxed Go REPL"
date: 2025-01-06T00:00:00+00:00
lastmod: 2025-01-06T00:00:00+00:00
draft: false
weight: 360
toc: true
seo:
  title: "RLM Module - dspy-go"
  description: "Complete guide to Recursive Language Model for large context analysis in dspy-go"
  canonical: ""
  noindex: false
---

# RLM Module (Recursive Language Model)

The **RLM Module** enables LLMs to programmatically explore large contexts through a sandboxed Go REPL. The LLM iteratively writes and executes code, making sub-LLM queries until it reaches a final answer.

## Why RLM?

Traditional LLM approaches fail when:
- **Context exceeds token limits** - Documents too large to fit in a single prompt
- **Complex analysis required** - Multi-step exploration needing programmatic logic
- **Iterative refinement** - Questions requiring multiple rounds of investigation

RLM solves this by:
- **Recursive Exploration**: LLM writes Go code to explore data
- **Sandboxed Execution**: Yaegi interpreter with restricted stdlib (no os, net, syscall)
- **Sub-LLM Queries**: `Query()` and `QueryBatched()` functions for parallel LLM calls
- **Token Tracking**: Separate tracking for root and sub-LLM token usage

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RLM Module                                │
├─────────────────────────────────────────────────────────────┤
│  Large Context (100K+ chars) + Query                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Root LLM                                  │
├─────────────────────────────────────────────────────────────┤
│  Generates Go code to explore context                       │
│  Uses Query()/QueryBatched() for sub-LLM calls              │
│  Calls FINAL(answer) when done                              │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Iteration 1 │   │   Iteration 2 │   │   Iteration N │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ Execute code  │   │ Execute code  │   │ FINAL(answer) │
│ Get output    │   │ Get output    │   │ Return result │
│ Continue...   │   │ Continue...   │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## Quick Start

### Basic Usage

```go
import "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"

// Create RLM module from any LLM
rlmModule := rlm.NewFromLLM(llm,
    rlm.WithMaxIterations(10),       // Max exploration iterations
    rlm.WithVerbose(true),           // Enable detailed logging
    rlm.WithTimeout(5*time.Minute),  // Overall timeout
)

// Large context (e.g., 100K+ characters)
document := loadLargeDocument()
query := "What percentage of reviews are positive?"

// Execute exploration
result, err := rlmModule.Complete(ctx, document, query)
if err != nil {
    log.Fatal(err)
}

// Access results
fmt.Printf("Answer: %s\n", result.Response)
fmt.Printf("Iterations: %d\n", result.Iterations)
fmt.Printf("Total Tokens: %d\n", result.Usage.TotalTokens)
```

### Token Tracking

```go
// Get detailed token tracking
tracker := rlmModule.GetTokenTracker()

// Root LLM usage (orchestration)
rootUsage := tracker.GetRootUsage()
fmt.Printf("Root LLM: %d tokens\n", rootUsage.TotalTokens)

// Sub-LLM calls (from Query/QueryBatched)
subCalls := tracker.GetSubCalls()
fmt.Printf("Sub-LLM calls: %d\n", len(subCalls))

for i, call := range subCalls {
    fmt.Printf("  Call %d: %d tokens\n", i+1, call.TotalTokens)
}

// Total across all calls
totalUsage := tracker.GetTotalUsage()
fmt.Printf("Total: %d tokens\n", totalUsage.TotalTokens)
```

---

## Configuration Options

```go
rlmModule := rlm.NewFromLLM(llm,
    // Iteration control
    rlm.WithMaxIterations(10),       // Maximum exploration steps
    rlm.WithTimeout(5*time.Minute),  // Overall timeout

    // Logging
    rlm.WithVerbose(true),           // Detailed console output
    rlm.WithJSONLPath("session.jsonl"), // Log to JSONL file

    // Model configuration
    rlm.WithTemperature(0.7),        // Response temperature
    rlm.WithMaxTokens(4096),         // Max tokens per response

    // Custom instructions
    rlm.WithSystemPrompt(customPrompt), // Override system prompt
)
```

### Configuration Reference

| Option | Description | Default |
|--------|-------------|---------|
| `WithMaxIterations` | Maximum exploration iterations | `10` |
| `WithTimeout` | Overall execution timeout | `5m` |
| `WithVerbose` | Enable detailed console logging | `false` |
| `WithJSONLPath` | Path for JSONL session log | `""` |
| `WithTemperature` | LLM temperature | `0.7` |
| `WithMaxTokens` | Max tokens per response | `4096` |

---

## JSONL Session Logging

RLM supports detailed session logging in JSONL format for debugging and analysis.

### Enable Logging

```go
rlmModule := rlm.NewFromLLM(llm,
    rlm.WithVerbose(true),
    rlm.WithJSONLPath("session.jsonl"),
)
```

### View Logs with CLI

The dspy-cli includes a powerful viewer for RLM session logs:

```bash
# View log with statistics
./dspy-cli view session.jsonl --stats

# Interactive navigation mode
./dspy-cli view -i session.jsonl

# Watch live as log is written
./dspy-cli view -w session.jsonl

# Search and filter
./dspy-cli view -s "error" session.jsonl --errors

# Export to markdown
./dspy-cli view --export report.md session.jsonl
```

### Log Format

Each log entry contains:

```json
{
  "timestamp": "2025-01-06T10:30:00Z",
  "type": "iteration",
  "iteration": 1,
  "code": "// Go code executed",
  "output": "Execution result",
  "tokens": {
    "prompt": 1000,
    "completion": 500,
    "total": 1500
  }
}
```

---

## Available Functions

Within the RLM sandbox, the LLM can use these functions:

### Query(prompt string) string

Make a sub-LLM call with a single prompt:

```go
// In RLM-generated code
result := Query("Summarize this section: " + section)
fmt.Println(result)
```

### QueryBatched(prompts []string) []string

Make parallel sub-LLM calls for efficiency:

```go
// In RLM-generated code
prompts := []string{
    "Analyze section 1: " + section1,
    "Analyze section 2: " + section2,
    "Analyze section 3: " + section3,
}
results := QueryBatched(prompts)

for i, result := range results {
    fmt.Printf("Section %d: %s\n", i+1, result)
}
```

### FINAL(answer string)

Signal completion with the final answer:

```go
// In RLM-generated code
FINAL("The positive review percentage is 72%")
```

### Context Access

The large context is available as the `Context` variable:

```go
// In RLM-generated code
lines := strings.Split(Context, "\n")
fmt.Printf("Document has %d lines\n", len(lines))

// Search for specific content
for i, line := range lines {
    if strings.Contains(line, "revenue") {
        fmt.Printf("Line %d: %s\n", i, line)
    }
}
```

---

## Sandbox Security

The RLM sandbox uses the Yaegi interpreter with restricted stdlib access:

### Allowed Packages

- `fmt` - Formatting and printing
- `strings` - String manipulation
- `strconv` - String conversion
- `math` - Mathematical functions
- `sort` - Sorting algorithms
- `regexp` - Regular expressions
- `time` - Time operations (limited)
- `encoding/json` - JSON parsing

### Blocked Packages

- `os` - No filesystem access
- `net` - No network access
- `syscall` - No system calls
- `unsafe` - No unsafe operations
- `runtime` - Limited runtime access

This ensures the LLM-generated code cannot:
- Access the filesystem
- Make network requests
- Execute system commands
- Escape the sandbox

---

## Use Cases

### Large Document Analysis

```go
// Analyze a 500-page document
doc := loadPDFText("large_report.pdf")
query := "What are the key financial trends mentioned?"

result, _ := rlmModule.Complete(ctx, doc, query)
fmt.Println(result.Response)
```

### Multi-Step Research

```go
// Research requiring iterative exploration
context := loadResearchPapers()
query := "Compare the methodologies used across all papers and identify common patterns"

result, _ := rlmModule.Complete(ctx, context, query)
```

### Data Extraction

```go
// Extract structured data from large datasets
data := loadCSVData()
query := "Calculate the average sales by region and identify outliers"

result, _ := rlmModule.Complete(ctx, data, query)
```

### Code Analysis

```go
// Analyze a large codebase
codebase := loadCodebaseText()
query := "Identify all API endpoints and their authentication requirements"

result, _ := rlmModule.Complete(ctx, codebase, query)
```

---

## Advanced Patterns

### Oolong Strategy

The oolong strategy pattern enables more sophisticated exploration:

```go
// See examples/rlm_oolong for full implementation
rlmModule := rlm.NewFromLLM(llm,
    rlm.WithMaxIterations(15),
    rlm.WithOolongStrategy(true), // Enable oolong strategy
)
```

### Custom System Prompts

```go
customPrompt := `You are analyzing a financial document.
Focus on revenue, expenses, and profit margins.
Use Query() for detailed analysis of specific sections.
Use QueryBatched() to analyze multiple sections in parallel.
Call FINAL() with your comprehensive analysis.`

rlmModule := rlm.NewFromLLM(llm,
    rlm.WithSystemPrompt(customPrompt),
)
```

### Combining with Other Modules

```go
// Use RLM for exploration, then ChainOfThought for reasoning
rlmResult, _ := rlmModule.Complete(ctx, largeDoc, "Extract key facts")

cotModule := modules.NewChainOfThought(reasoningSignature)
finalResult, _ := cotModule.Process(ctx, map[string]interface{}{
    "facts": rlmResult.Response,
    "question": "What conclusions can we draw?",
})
```

---

## Performance Considerations

### Token Efficiency

- Use `QueryBatched()` instead of multiple `Query()` calls
- Keep iteration count reasonable (10-15 max)
- Use specific, focused queries

### Memory Management

- RLM loads the full context into memory
- For very large documents, consider chunking
- Monitor `result.Usage.TotalTokens` for cost tracking

### Timeout Handling

```go
// Set appropriate timeout based on document size
timeout := time.Duration(len(document)/10000) * time.Minute
if timeout < 2*time.Minute {
    timeout = 2 * time.Minute
}

rlmModule := rlm.NewFromLLM(llm,
    rlm.WithTimeout(timeout),
)
```

---

## Examples

### Basic RLM

```bash
cd examples/rlm
go run main.go --api-key YOUR_API_KEY
```

### RLM Oolong Strategy

```bash
cd examples/rlm_oolong
go run main.go --api-key YOUR_API_KEY
```

**[RLM Examples](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/rlm)**

---

## Next Steps

- **[Core Concepts](core-concepts/)** - Understand modules and signatures
- **[Building Agents](agents/)** - Combine RLM with agent patterns
- **[CLI Reference](../reference/cli/)** - Full CLI documentation for log viewing
