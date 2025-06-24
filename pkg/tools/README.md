# Smart Tool Registry

The Smart Tool Registry is an intelligent tool management system that provides advanced capabilities for tool selection, performance tracking, and automatic discovery from MCP servers.

## Features

### 🧠 Intelligent Tool Selection
- **Bayesian Tool Selector**: Uses Bayesian inference to score and select the best tools based on intent matching, performance metrics, and capability analysis
- **Multi-factor Scoring**: Combines match score, performance score, and capability score with configurable weights
- **Synonym Recognition**: Intelligent matching of related terms (e.g., "search" matches "find", "query", "lookup")
- **Prior Probability Integration**: Learns from tool usage patterns to improve selection over time

### 📊 Performance Tracking
- **Execution Metrics**: Tracks success rate, average latency, and execution count for each tool
- **Reliability Scoring**: Computes composite reliability scores based on success rate and performance
- **Real-time Updates**: Performance metrics are updated with each tool execution
- **Historical Analysis**: Maintains performance history for trend analysis

### 🔍 Capability Analysis
- **Automatic Capability Extraction**: Infers tool capabilities from metadata and descriptions
- **Capability Matching**: Matches user intents to tool capabilities with confidence scoring
- **Keyword-based Inference**: Extracts capabilities from tool descriptions using keyword analysis

### 🔄 Automatic Discovery
- **MCP Integration**: Automatically discovers tools from connected MCP servers
- **Real-time Updates**: Subscribes to tool updates and dynamically registers new tools
- **Fallback Mechanisms**: Provides fallback tool selection when primary tools fail

## Usage

### Basic Setup

```go
import "github.com/XiaoConstantine/dspy-go/pkg/tools"

// Create a smart tool registry
config := &tools.SmartToolRegistryConfig{
    AutoDiscoveryEnabled:       true,
    PerformanceTrackingEnabled: true,
    FallbackEnabled:           true,
}
registry := tools.NewSmartToolRegistry(config)

// Register tools
searchTool := tools.NewMockTool("search", "Search for information", []string{"search", "query"})
err := registry.Register(searchTool)
```

### Intelligent Tool Selection

```go
// Select the best tool for an intent
ctx := context.Background()
intent := "I need to find user information"
tool, err := registry.SelectBest(ctx, intent)
if err != nil {
    log.Fatal(err)
}

// Execute with performance tracking
params := map[string]interface{}{"query": "user data"}
result, err := registry.ExecuteWithTracking(ctx, tool.Name(), params)
```

### Performance Metrics

```go
// Get performance metrics for a tool
metrics, err := registry.GetPerformanceMetrics("search")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Success Rate: %.2f%%\n", metrics.SuccessRate*100)
fmt.Printf("Average Latency: %v\n", metrics.AverageLatency)
fmt.Printf("Reliability Score: %.2f\n", metrics.ReliabilityScore)
```

### Fallback Configuration

```go
// Add fallback tools for specific intents
err := registry.AddFallback("search data", "backup_search_tool")
if err != nil {
    log.Fatal(err)
}
```

## Architecture

### Components

1. **SmartToolRegistry**: Main registry that orchestrates intelligent tool management
2. **BayesianToolSelector**: Implements Bayesian inference for tool selection
3. **MCPDiscoveryService**: Handles automatic tool discovery from MCP servers
4. **PerformanceMetrics**: Tracks and analyzes tool performance
5. **ToolCapability**: Represents tool capabilities with confidence scores

### Selection Algorithm

The Bayesian tool selector uses a weighted scoring system:

- **Match Score** (40%): How well the tool name/description matches the intent
- **Performance Score** (35%): Based on historical performance metrics
- **Capability Score** (25%): How well tool capabilities match the intent

The final score can be adjusted by prior probabilities based on tool usage patterns.

### Performance Metrics

Each tool maintains the following metrics:
- Execution count and success/failure counts
- Success rate (success_count / total_executions)
- Average latency with exponential moving average
- Reliability score (composite of success rate and latency)

### Capability Inference

Tools' capabilities are extracted from:
1. Explicit capabilities in tool metadata
2. Keyword analysis of tool descriptions
3. Pattern matching against common capability types

## Configuration

### SmartToolRegistryConfig

```go
type SmartToolRegistryConfig struct {
    Selector                   ToolSelector        // Custom tool selector (default: BayesianToolSelector)
    MCPDiscovery              MCPDiscoveryService // MCP discovery service
    AutoDiscoveryEnabled      bool               // Enable automatic tool discovery
    PerformanceTrackingEnabled bool               // Enable performance tracking
    FallbackEnabled           bool               // Enable fallback mechanisms
}
```

### BayesianToolSelector Weights

```go
selector := tools.NewBayesianToolSelector()
selector.MatchWeight = 0.5        // Increase emphasis on name/description matching
selector.PerformanceWeight = 0.3  // Moderate emphasis on performance
selector.CapabilityWeight = 0.2   // Lower emphasis on capabilities
```

## Testing

The Smart Tool Registry includes comprehensive test coverage:

- Unit tests for all major components
- Integration tests for tool selection workflows
- Performance benchmarks for large tool sets
- Mock implementations for testing

Run tests:
```bash
go test ./pkg/tools/ -v
```

Run benchmarks:
```bash
go test ./pkg/tools/ -bench=. -benchmem
```

## Performance

The Smart Tool Registry is designed for high performance:

- Tool selection typically completes in microseconds
- Performance metrics updates are non-blocking
- Memory-efficient scoring algorithms
- Concurrent-safe operations with minimal locking

Benchmark results on typical hardware:
- Tool selection: ~10μs for 100 tools
- Performance update: ~1μs per execution
- Auto-discovery: ~100ms for MCP server polling

## Future Enhancements

- Machine learning-based capability inference
- Advanced performance prediction models
- Tool recommendation system
- Integration with external tool marketplaces
- Real-time tool performance dashboards