# A2A Deep Research Agent Example

This example demonstrates Google's **Agent-to-Agent (A2A) protocol** implementation in dspy-go with a real-world **deep research system** using actual LLMs and multi-agent composition.

## Overview

Unlike simple demonstration agents, this example features a sophisticated research system with **three specialized LLM-powered agents** coordinated by a parent orchestrator, showcasing the full power of A2A agent composition.

### What This Demonstrates

✅ **Multi-agent hierarchical composition** using A2A protocol
✅ **Real LLM-powered agents** using dspy-go modules and signatures
✅ **Complex multi-step workflows** (search → analyze → synthesize)
✅ **In-process agent communication** (no HTTP overhead)
✅ **Agent capability discovery** and registration
✅ **Production-ready patterns** for agent systems

## Architecture

```
ResearchOrchestrator (Parent)
├── SearchAgent (LLM-powered)
│   └── Generates search queries & gathers information
├── AnalysisAgent (LLM-powered)
│   └── Extracts insights, patterns, contradictions
└── SynthesisAgent (LLM-powered)
    └── Creates comprehensive research reports
```

### Agent Specifications

#### 1. SearchAgent
- **Purpose**: Information gathering and query formulation
- **Input**: Research topic
- **Output**: Targeted search queries + simulated search results
- **LLM Module**: `modules.NewPredict` with search signature
- **Prompting**: Generates 3-5 specific queries with diverse perspectives

#### 2. AnalysisAgent
- **Purpose**: Deep analysis of gathered information
- **Input**: Topic + search results
- **Output**: Key findings, patterns, contradictions, gaps
- **LLM Module**: `modules.NewPredict` with analytical signature
- **Prompting**: Critical, evidence-based analysis

#### 3. SynthesisAgent
- **Purpose**: Report generation and knowledge synthesis
- **Input**: Topic + analysis results
- **Output**: Executive summary, detailed report, conclusions, recommendations
- **LLM Module**: `modules.NewPredict` with synthesis signature
- **Prompting**: Professional, structured reporting

#### 4. ResearchOrchestrator
- **Purpose**: Workflow coordination
- **Behavior**: Manages 3-step pipeline (search → analysis → synthesis)
- **A2A Role**: Parent agent coordinating sub-agents via A2A messages
- **Output**: Complete research report with all sections

## Running the Example

### Prerequisites

- Go 1.21 or later
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Basic Usage

```bash
# With default model (Gemini 2.0 Flash)
go run main.go --api-key YOUR_GEMINI_API_KEY

# With Gemini Pro
go run main.go --api-key YOUR_GEMINI_API_KEY --model gemini-pro

# With Gemini Flash
go run main.go --api-key YOUR_GEMINI_API_KEY --model gemini-flash
```

### Command-Line Flags

- `--api-key` (required): Your Google Gemini API key
- `--model` (optional): Gemini model to use (default: `gemini-2.0-flash-exp`)

### Supported Gemini Models

```go
// Recommended models
"gemini-2.0-flash-exp"  // Default - Fast and cost-effective
"gemini-pro"            // More capable, higher quality
"gemini-flash"          // Fast responses

// Experimental models
"gemini-exp-1206"       // Latest experimental features
```

> **Note**: This example is optimized for Google Gemini models. While dspy-go supports other providers (Claude, OpenAI), the example uses Gemini-specific model IDs.

## Sample Output

```
╔════════════════════════════════════════════════════════════════╗
║       A2A Deep Research Agent - Multi-Agent Composition        ║
╚════════════════════════════════════════════════════════════════╝

⚙️  Configuring LLM: gemini-2.0-flash-exp
🔧 Initializing research agents...
✓ Research system ready with 3 specialized agents

████████████████████████████████████████████████████████████████
RESEARCH PROJECT 1/2
████████████████████████████████████████████████████████████████

🎯 Research Orchestrator: Starting deep research on:
   What are the latest advancements in quantum computing...
================================================================================

Step 1/3: Information Gathering
--------------------------------------------------------------------------------
🔍 SearchAgent: Gathering information...
✓ Search completed. Found 2 result sets.

Step 2/3: Information Analysis
--------------------------------------------------------------------------------
📊 AnalysisAgent: Analyzing search results...
✓ Analysis completed. Identified key findings and patterns.

Step 3/3: Report Synthesis
--------------------------------------------------------------------------------
📝 SynthesisAgent: Creating research report...
✓ Report synthesis completed.

╔════════════════════════════════════════════════════════════════╗
║                    RESEARCH REPORT                             ║
╚════════════════════════════════════════════════════════════════╝

📌 Topic: What are the latest advancements in quantum computing...

📋 EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
[2-3 paragraph summary of key findings...]

📄 DETAILED REPORT
--------------------------------------------------------------------------------
[Comprehensive research report with sections...]

💡 CONCLUSIONS
--------------------------------------------------------------------------------
[Evidence-based conclusions...]

🎯 RECOMMENDATIONS
--------------------------------------------------------------------------------
[Actionable recommendations...]
```

## Key Code Patterns

### 1. Creating LLM-Powered Agents

```go
func NewSearchAgent() (*SearchAgent, error) {
    // Define input/output schema with descriptions
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.Field{
                Name: "topic",
                Description: "The research topic or question",
            }},
        },
        []core.OutputField{
            {Field: core.Field{
                Name: "search_queries",
                Description: "List of 3-5 specific search queries",
                Prefix: "search queries:",
            }},
        },
    ).WithInstruction("You are a skilled research assistant...")

    return &SearchAgent{
        searchModule: modules.NewPredict(signature),
    }, nil
}
```

### 2. Agent Execution with LLM

```go
func (s *SearchAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    logger := logging.GetLogger()
    logger.Info(ctx, "🔍 SearchAgent: Gathering information...")

    // LLM call happens here via the module
    result, err := s.searchModule.Process(ctx, input)
    if err != nil {
        return nil, fmt.Errorf("search failed: %w", err)
    }

    return result, nil
}
```

### 3. A2A Agent Composition

```go
// Create LLM-powered agents
searchAgent, _ := NewSearchAgent()
analysisAgent, _ := NewAnalysisAgent()
synthesisAgent, _ := NewSynthesisAgent()

// Wrap with A2A executors
searchExec := a2a.NewExecutorWithConfig(searchAgent, a2a.ExecutorConfig{
    Name: "SearchAgent",
})

// Create orchestrator and compose
_, orchestratorExec := NewResearchOrchestrator()
orchestratorExec.WithSubAgent("search", searchExec).
                WithSubAgent("analysis", analysisExec).
                WithSubAgent("synthesis", synthesisExec)
```

### 4. Multi-Step Workflow Coordination

```go
func (r *ResearchOrchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    // Step 1: Search
    searchResult, err := r.executor.CallSubAgent(ctx, "search",
        a2a.NewUserMessage(topic))

    // Step 2: Analyze
    analysisInput := a2a.NewMessage(a2a.RoleUser,
        a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
        a2a.NewTextPartWithMetadata(searchResults, map[string]interface{}{"field": "search_results"}),
    )
    analysisResult, err := r.executor.CallSubAgent(ctx, "analysis", analysisInput)

    // Step 3: Synthesize
    synthesisResult, err := r.executor.CallSubAgent(ctx, "synthesis", synthesisInput)

    return compiledOutput, nil
}
```

## A2A Message Flow

```
User Request (topic: "quantum computing advancements")
    ↓
ResearchOrchestrator.Execute()
    │
    ├─→ CallSubAgent("search", topic)
    │   └─→ SearchAgent processes with LLM
    │       └─→ Returns Artifact(search_queries, search_results)
    │
    ├─→ CallSubAgent("analysis", topic + search_results)
    │   └─→ AnalysisAgent analyzes with LLM
    │       └─→ Returns Artifact(key_findings, patterns, contradictions, gaps)
    │
    └─→ CallSubAgent("synthesis", topic + analysis_results)
        └─→ SynthesisAgent synthesizes with LLM
            └─→ Returns Artifact(executive_summary, detailed_report, conclusions, recommendations)
```

## Understanding the A2A Protocol

### Message Format

Messages contain typed parts with metadata:

```go
msg := a2a.NewMessage(a2a.RoleUser,
    a2a.NewTextPartWithMetadata(content, map[string]interface{}{
        "field": "fieldname",
    }),
)
```

### Artifact Format

Agents return artifacts with structured parts:

```go
artifact := Artifact{
    Parts: []Part{
        {Type: "text", Text: "...", Metadata: {"field": "summary"}},
        {Type: "text", Text: "...", Metadata: {"field": "details"}},
    },
}
```

### Key Benefits

1. **Type Safety**: All communication uses structured messages
2. **Metadata**: Parts carry semantic meaning via metadata
3. **Composability**: Agents work together without tight coupling
4. **Discoverability**: Agent capabilities are self-describing
5. **Interoperability**: Compatible with Python ADK agents

## Advanced Patterns

### Add New Specialized Agents

```go
// Create a fact-checking agent
type FactCheckerAgent struct {
    module core.Module
}

func NewFactCheckerAgent() (*FactCheckerAgent, error) {
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.Field{Name: "claims"}},
        },
        []core.OutputField{
            {Field: core.Field{Name: "verified_facts", Prefix: "verified:"}},
            {Field: core.Field{Name: "unverified_claims", Prefix: "unverified:"}},
        },
    ).WithInstruction("Verify the accuracy of claims using reliable sources...")

    return &FactCheckerAgent{
        module: modules.NewPredict(signature),
    }, nil
}

// Add to orchestrator
factCheckerExec := a2a.NewExecutorWithConfig(factChecker, a2a.ExecutorConfig{
    Name: "FactCheckerAgent",
})
orchestratorExec.WithSubAgent("factchecker", factCheckerExec)
```

### Multi-Round Research

```go
// Implement iterative refinement
func (r *ResearchOrchestrator) DeepResearch(ctx context.Context, topic string, maxRounds int) {
    for round := 0; round < maxRounds; round++ {
        searchResult := r.executor.CallSubAgent(ctx, "search", ...)
        analysisResult := r.executor.CallSubAgent(ctx, "analysis", ...)

        // Check if we need more information
        gaps := extractGaps(analysisResult)
        if len(gaps) == 0 {
            break // Research complete
        }

        // Refine search based on gaps
        topic = refineTopicBasedOnGaps(topic, gaps)
    }
}
```

### Add Real Tools

```go
// Integrate WebSearch tool
import "github.com/XiaoConstantine/dspy-go/pkg/tools"

func (s *SearchAgent) GetCapabilities() []core.Tool {
    return []core.Tool{
        tools.NewWebSearchTool(apiKey),
    }
}
```

## Performance Considerations

- **API Costs**: Each agent makes Gemini API calls; consider using `gemini-flash` for cost optimization
- **Model Selection**:
  - `gemini-2.0-flash-exp`: Best balance of speed and quality (recommended)
  - `gemini-pro`: Higher quality for complex analysis
  - `gemini-flash`: Fastest responses, lower cost
- **Caching**: Enable with `ctx = core.WithExecutionState(ctx)` to cache LLM responses
- **In-Process**: Sub-agent calls have zero serialization overhead
- **Parallelization**: Independent agents can be called concurrently (future enhancement)

## Comparison with Python ADK

This dspy-go implementation is equivalent to Python ADK:

```python
# Python ADK
from google.generativeai import adk

search_agent = LLMAgent(model=llm, instruction="...")
analysis_agent = LLMAgent(model=llm, instruction="...")
synthesis_agent = LLMAgent(model=llm, instruction="...")

orchestrator = LLMAgent(
    model=llm,
    sub_agents=[search_agent, analysis_agent, synthesis_agent]
)
```

```go
// dspy-go A2A
searchExec := a2a.NewExecutorWithConfig(searchAgent, ...)
analysisExec := a2a.NewExecutorWithConfig(analysisAgent, ...)
synthesisExec := a2a.NewExecutorWithConfig(synthesisAgent, ...)

orchestratorExec.WithSubAgent("search", searchExec).
                WithSubAgent("analysis", analysisExec).
                WithSubAgent("synthesis", synthesisExec)
```

## Related Examples

- `examples/agents/` - Advanced agent workflows and orchestration
- `examples/react_agent/` - ReAct agent with tool usage
- `examples/tool_composition/` - Tool chaining patterns
- `pkg/agents/communication/README.md` - A2A protocol documentation

## Next Steps

1. **Experiment with Gemini models** (`gemini-flash` vs `gemini-pro`) to find the best performance/cost tradeoff
2. **Add your own specialized agents** for domain-specific research
3. **Integrate real tools** like WebSearch, databases, APIs
4. **Implement multi-round refinement** for deeper research
5. **Deploy as HTTP service** for cross-language agent interop
6. **Try Gemini's multimodal capabilities** by adding image/document analysis agents

## References

- [Google Gemini API](https://ai.google.dev/gemini-api/docs) - Official Gemini documentation
- [Get Gemini API Key](https://aistudio.google.com/app/apikey) - Free API key
- [Google Agent Developer Kit (ADK)](https://github.com/google/adk-python) - Python agent framework
- [A2A Protocol Specification](https://developers.google.com/agent-developer-kit/docs) - Agent-to-Agent protocol
- [dspy-go Documentation](../../README.md) - Main dspy-go docs
- [dspy-go Modules & Signatures](../../pkg/modules/README.md) - Module system guide
