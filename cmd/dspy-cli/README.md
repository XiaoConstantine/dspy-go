# DSPy-CLI: Interactive Command-Line Tool for DSPy-Go Optimizers

A powerful command-line interface for exploring, testing, and optimizing DSPy-Go programs with zero boilerplate code.

## 🚀 Quick Start

```bash
# Install the CLI
cd cmd/dspy-cli
go build -o dspy-cli

# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Try an optimizer with sample data
./dspy-cli try bootstrap --dataset gsm8k --max-examples 5

# List all available optimizers
./dspy-cli list

# Get detailed optimizer information
./dspy-cli describe mipro

# Get optimizer recommendations
./dspy-cli recommend advanced
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Commands](#-commands)
- [Optimizers](#-optimizers)
- [Datasets](#-datasets)
- [Examples](#-examples)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **Zero Boilerplate**: Eliminates 60+ lines of setup code per optimizer
- **All Optimizers Supported**: Bootstrap, MIPRO, SIMBA, GEPA, COPRO
- **Sample Datasets**: Built-in GSM8K, HotPotQA, and Simple Q&A datasets
- **Intelligent Recommendations**: Get optimizer suggestions based on your use case
- **Beautiful Output**: Colored, formatted results with progress indicators
- **Dependency Isolation**: CLI dependencies don't pollute the main library
- **Quick Experimentation**: Test optimizers in seconds, not minutes

## 🛠 Installation

### Prerequisites

- Go 1.21 or higher
- Gemini API key (set as `GEMINI_API_KEY` environment variable)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/XiaoConstantine/dspy-go.git
cd dspy-go/cmd/dspy-cli

# Build the CLI
go build -o dspy-cli

# Optional: Install globally
sudo mv dspy-cli /usr/local/bin/
```

### Go Workspace Setup

If you're developing, the CLI uses Go workspaces for seamless integration:

```bash
# The workspace is already configured in go.work
cd dspy-go
go work sync
```

## 📝 Commands

### `list` - Show All Optimizers

```bash
dspy-cli list
```

Displays all available optimizers with their status, complexity, and use cases.

### `describe` - Detailed Optimizer Information

```bash
dspy-cli describe <optimizer>
```

Shows comprehensive information about a specific optimizer including:
- Description and use cases
- Complexity and computational cost
- Best-for scenarios
- Example applications

```bash
# Examples
dspy-cli describe mipro
dspy-cli describe bootstrap
dspy-cli describe gepa
```

### `recommend` - Get Optimizer Suggestions

```bash
dspy-cli recommend <use-case>
```

Get intelligent optimizer recommendations based on your needs:

```bash
dspy-cli recommend beginner      # Simple, quick optimizers
dspy-cli recommend balanced      # Good performance/cost balance
dspy-cli recommend advanced      # Cutting-edge optimizers
dspy-cli recommend multi-module  # Collaborative optimization
```

### `try` - Run Optimizer with Sample Data

```bash
dspy-cli try <optimizer> [flags]
```

**Flags:**
- `--dataset`: Dataset to use (gsm8k, hotpotqa, simple)
- `--max-examples`: Limit number of examples (default: all)
- `--verbose`: Enable detailed logging
- `--api-key`: API key (overrides environment variable)

**Examples:**

```bash
# Basic usage
dspy-cli try bootstrap --dataset gsm8k

# Limit examples for faster testing
dspy-cli try mipro --dataset gsm8k --max-examples 5

# Verbose output for debugging
dspy-cli try simba --dataset hotpotqa --verbose

# All optimizers
dspy-cli try bootstrap --dataset simple
dspy-cli try mipro --dataset gsm8k --max-examples 3
dspy-cli try simba --dataset hotpotqa --max-examples 5
dspy-cli try gepa --dataset gsm8k --max-examples 3
dspy-cli try copro --dataset simple --max-examples 10
```

## 🔧 Optimizers

All optimizers are fully functional and tested:

### ✅ Bootstrap FewShot
- **Complexity**: Low
- **Cost**: Low (fast convergence)
- **Best For**: Getting started, simple tasks, limited budget
- **Use Case**: Quick improvements with minimal computational cost

### ✅ MIPRO
- **Complexity**: Medium
- **Cost**: Medium (systematic search)
- **Best For**: Complex reasoning, systematic optimization, multi-step problems
- **Use Case**: Balanced cost/performance requirements

### ✅ SIMBA
- **Complexity**: High
- **Cost**: Medium-High (introspective analysis)
- **Best For**: Self-reflection tasks, adaptive learning, advanced reasoning
- **Use Case**: Advanced optimization with introspective learning

### ✅ GEPA
- **Complexity**: Very High
- **Cost**: High (evolutionary + Pareto optimization)
- **Best For**: Cutting-edge performance, multi-objective optimization, research
- **Use Case**: State-of-the-art optimization for demanding tasks

### ✅ COPRO
- **Complexity**: Medium-High
- **Cost**: Medium (collaborative approach)
- **Best For**: Multi-module systems, collaborative optimization, complex pipelines
- **Use Case**: Optimizing multiple modules working together

## 📊 Datasets

### GSM8K (Grade School Math 8K)
- **Type**: Math word problems
- **Examples**: Sample math problems with step-by-step solutions
- **Best For**: Reasoning, mathematical problem solving
- **Format**: Question → Answer

### HotPotQA
- **Type**: Multi-hop question answering
- **Examples**: Questions requiring multiple reasoning steps
- **Best For**: Complex reasoning, information synthesis
- **Format**: Question → Answer

### Simple Q&A
- **Type**: Basic question-answer pairs
- **Examples**: Simple factual questions
- **Best For**: Testing, basic optimization, getting started
- **Format**: Question → Answer

## 💡 Examples

### Complete Workflow Example

```bash
# 1. Explore available optimizers
dspy-cli list

# 2. Get recommendations for your use case
dspy-cli recommend balanced

# 3. Learn about a specific optimizer
dspy-cli describe mipro

# 4. Test the optimizer with sample data
dspy-cli try mipro --dataset gsm8k --max-examples 5 --verbose

# 5. Compare with other optimizers
dspy-cli try bootstrap --dataset gsm8k --max-examples 5
dspy-cli try simba --dataset gsm8k --max-examples 5
```

### Performance Comparison

```bash
# Test all optimizers on the same dataset for comparison
for optimizer in bootstrap mipro simba gepa copro; do
  echo "Testing $optimizer..."
  dspy-cli try $optimizer --dataset gsm8k --max-examples 3
  echo "---"
done
```

### Quick Validation

```bash
# Quickly validate that all optimizers are working
dspy-cli try bootstrap --dataset simple --max-examples 2
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required: Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Alternative names (auto-detected)
export GOOGLE_API_KEY="your-api-key-here"
export DSPY_API_KEY="your-api-key-here"
```

### LLM Configuration

The CLI automatically configures:
- **Model**: Google Gemini Flash (fast and cost-effective)
- **Provider**: Google AI Platform
- **Auto-detection**: API key from multiple environment variables

### Logging Levels

```bash
# Normal output
dspy-cli try bootstrap --dataset gsm8k

# Verbose debugging
dspy-cli try bootstrap --dataset gsm8k --verbose
```

## 🏗 Architecture

### Dependency Isolation

The CLI uses a separate Go module to prevent dependency pollution:

```
dspy-go/
├── pkg/                    # Main library (clean dependencies)
├── cmd/dspy-cli/
│   ├── go.mod             # Separate module with CLI dependencies
│   ├── main.go            # CLI entry point
│   └── internal/
│       ├── commands/       # Cobra commands
│       ├── optimizers/     # Optimizer registry
│       ├── runner/         # Execution engine
│       └── samples/        # Sample datasets
└── go.work                # Workspace configuration
```

### Key Components

1. **Command Layer**: Cobra-based CLI commands
2. **Registry**: Optimizer metadata and recommendations
3. **Runner**: Execution engine that eliminates boilerplate
4. **Samples**: Built-in datasets for quick testing
5. **Workspace**: Go workspace for seamless development

### Boilerplate Elimination

The CLI eliminates this typical setup code:

```go
// 60+ lines of boilerplate that the CLI handles automatically
func setupOptimizer() {
    // Context creation
    ctx := core.WithExecutionState(context.Background())

    // LLM configuration
    llms.EnsureFactory()
    core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)

    // Dataset loading and conversion
    examples, _ := datasets.LoadGSM8K()
    dataset := createDataset(examples)

    // Signature and module creation
    signature := core.NewSignature(/* ... */)
    module := modules.NewChainOfThought(signature)

    // Program creation with forward function
    program := core.NewProgram(/* ... */)

    // Optimizer creation and configuration
    optimizer := optimizers.NewMIPRO(/* complex config */)

    // Training/validation split
    trainExamples, valExamples := splitDataset(examples)

    // Metric function definition
    metricFunc := func(/* ... */) float64 { /* ... */ }

    // Compilation and evaluation
    optimizedProgram, _ := optimizer.Compile(ctx, program, dataset, metricFunc)

    // Results calculation and display
    // ...
}
```

## 🐛 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
Error: API key required. Set GEMINI_API_KEY environment variable
```
Solution: Set your Gemini API key in environment variables.

**No Examples in Dataset**
```bash
Error: No examples found in dataset
```
Solution: Check dataset name spelling and network connectivity.

**Optimizer Not Found**
```bash
Error: optimizer 'typo' not found
```
Solution: Use `dspy-cli list` to see available optimizers.

**Import Cycle During Development**
```bash
Error: import cycle detected
```
Solution: Ensure you're using the Go workspace: `go work sync`

### Performance Tips

1. **Start Small**: Use `--max-examples` flag for initial testing
2. **Use Verbose Mode**: Add `--verbose` for debugging
3. **Try Bootstrap First**: Fastest optimizer for initial validation
4. **Compare Results**: Test multiple optimizers on same dataset

### Development Tips

1. **Use Go Workspace**: Enables seamless library development
2. **Separate Dependencies**: CLI deps don't affect main library
3. **Test Locally**: Build and test before committing
4. **Check Registry**: Update optimizer status in registry.go

## 🔗 Related Documentation

- [Main DSPy-Go Documentation](../../README.md)
- [DSPy-Go Core Library](../../pkg/)
- [Optimizer Examples](../../examples/)
- [DSPy Framework](https://github.com/stanfordnlp/dspy)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Test the CLI: `go run main.go try bootstrap --dataset simple`
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

---

**Happy Optimizing! 🚀**

*The DSPy-CLI makes exploring and optimizing DSPy-Go programs effortless. Start with `dspy-cli list` and discover the power of automated prompt optimization.*
