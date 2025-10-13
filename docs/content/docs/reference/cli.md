---
title: "CLI Reference"
description: "Complete command-line interface reference for dspy-cli"
summary: "All commands, flags, and usage examples for the dspy-go CLI tool"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 920
toc: true
seo:
  title: "CLI Reference - dspy-go"
  description: "Complete CLI reference for dspy-go command-line tool"
  canonical: ""
  noindex: false
---

Complete reference for the **dspy-cli** command-line tool. Test optimizers, run experiments, and explore dspy-go without writing code.

---

## Installation

### Build from Source

```bash
cd cmd/dspy-cli
go build -o dspy-cli
```

### Install Globally

```bash
cd cmd/dspy-cli
go install
```

After installing, `dspy-cli` will be available in your `$GOPATH/bin` directory.

---

## Configuration

### API Keys

Set your LLM provider API key:

```bash
# Google Gemini (recommended for multimodal)
export GEMINI_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

## Commands

### `list` - List Available Optimizers

Display all available optimization algorithms with descriptions.

**Usage:**
```bash
dspy-cli list [flags]
```

**Flags:**
| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--detailed` | `-d` | Show detailed information | `false` |
| `--json` | | Output as JSON | `false` |

**Examples:**

```bash
# List all optimizers
dspy-cli list

# Show detailed information
dspy-cli list --detailed

# Output as JSON
dspy-cli list --json
```

**Output:**
```
Available Optimizers:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. bootstrap
   Automated few-shot learning with example selection
   Best for: Quick optimization with limited data

2. gepa
   Multi-objective evolutionary optimization with Pareto selection
   Best for: Highest quality results, complex tasks

3. mipro
   Systematic TPE-based prompt and demonstration optimization
   Best for: Balanced performance and quality

4. simba
   Introspective mini-batch learning
   Best for: Fast iteration with moderate quality

5. copro
   Collaborative multi-module prompt optimization
   Best for: Complex programs with multiple modules
```

---

### `try` - Test an Optimizer

Run an optimizer with a built-in dataset.

**Usage:**
```bash
dspy-cli try <optimizer> [flags]
```

**Arguments:**
| Argument | Description | Required |
|----------|-------------|----------|
| `<optimizer>` | Optimizer name (bootstrap, gepa, mipro, simba, copro) | Yes |

**Flags:**
| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset to use (gsm8k, hotpotqa) | `gsm8k` |
| `--max-examples` | `-n` | Number of examples to use | `10` |
| `--verbose` | `-v` | Verbose output | `false` |
| `--timeout` | `-t` | Timeout in seconds | `300` |
| `--output` | `-o` | Output file for results | stdout |
| `--json` | | Output as JSON | `false` |

**Examples:**

```bash
# Try Bootstrap with default settings
dspy-cli try bootstrap

# Try GEPA with GSM8K dataset
dspy-cli try gepa --dataset gsm8k --max-examples 20

# Try MIPRO with verbose output
dspy-cli try mipro --verbose

# Try SIMBA and save results
dspy-cli try simba --output results.json --json

# Try with custom timeout
dspy-cli try copro --timeout 600
```

**Sample Output:**
```
🚀 Starting Bootstrap Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: GSM8K (Math Problems)
Examples: 10
LLM: gemini-pro

⏳ Optimizing prompts...

[1/10] Processing example: "Janet's ducks lay 16 eggs..."
  ✓ Generated demonstration
  ✓ Score: 1.0

[2/10] Processing example: "A robe takes 2 bolts..."
  ✓ Generated demonstration
  ✓ Score: 1.0

...

✨ Optimization Complete!

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Accuracy: 90.0%
  Examples: 10
  Duration: 45.2s
  Avg Time: 4.5s per example

Best Examples Selected: 5
Prompt Optimized: Yes
```

---

### `recommend` - Get Optimizer Recommendations

Get optimizer recommendations based on your requirements.

**Usage:**
```bash
dspy-cli recommend <profile> [flags]
```

**Arguments:**
| Argument | Description | Values |
|----------|-------------|--------|
| `<profile>` | Use case profile | `speed`, `balanced`, `quality`, `custom` |

**Flags (for custom profile):**
| Flag | Description | Range |
|------|-------------|-------|
| `--time-budget` | Time budget in seconds | `1-3600` |
| `--quality-target` | Target quality score | `0.0-1.0` |
| `--complexity` | Task complexity | `low`, `medium`, `high` |
| `--data-size` | Dataset size | `small`, `medium`, `large` |

**Examples:**

```bash
# Quick recommendation for speed
dspy-cli recommend speed

# Balanced recommendation
dspy-cli recommend balanced

# High-quality results
dspy-cli recommend quality

# Custom recommendation
dspy-cli recommend custom \
  --time-budget 300 \
  --quality-target 0.9 \
  --complexity high \
  --data-size medium
```

**Sample Output:**
```
🎯 Optimizer Recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Profile: Balanced
Time Budget: ~5 minutes
Quality Target: High

Recommended: MIPRO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Systematic prompt optimization
✅ Good balance of speed and quality
✅ Works well with moderate datasets
✅ TPE-based search is efficient

Command to try:
  dspy-cli try mipro --dataset gsm8k --max-examples 20

Alternative: Bootstrap (faster, slightly lower quality)
Alternative: GEPA (slower, higher quality)
```

---

### `compare` - Compare Optimizers

Compare multiple optimizers on the same dataset.

**Usage:**
```bash
dspy-cli compare [optimizers...] [flags]
```

**Flags:**
| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset to use | `gsm8k` |
| `--max-examples` | `-n` | Examples per optimizer | `10` |
| `--parallel` | `-p` | Run in parallel | `false` |
| `--output` | `-o` | Output file | stdout |
| `--format` | | Output format (table, json, csv) | `table` |

**Examples:**

```bash
# Compare all optimizers
dspy-cli compare bootstrap mipro gepa

# Compare with specific dataset
dspy-cli compare bootstrap mipro --dataset hotpotqa

# Run comparisons in parallel
dspy-cli compare bootstrap mipro simba --parallel

# Output as CSV
dspy-cli compare bootstrap gepa --format csv --output comparison.csv
```

**Sample Output:**
```
⚖️  Optimizer Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: GSM8K
Examples: 10

┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Optimizer ┃ Accuracy ┃ Time (s) ┃ Quality ┃
┣━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━┫
┃ Bootstrap ┃ 85.0%    ┃ 32.1     ┃ ⭐⭐⭐   ┃
┃ MIPRO     ┃ 90.0%    ┃ 58.4     ┃ ⭐⭐⭐⭐ ┃
┃ GEPA      ┃ 95.0%    ┃ 124.7    ┃ ⭐⭐⭐⭐⭐┃
┗━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━┛

🏆 Winner: GEPA (95.0% accuracy)
⚡ Fastest: Bootstrap (32.1s)
💎 Best Value: MIPRO (good balance)
```

---

### `benchmark` - Run Comprehensive Benchmarks

Run comprehensive benchmarks across datasets and optimizers.

**Usage:**
```bash
dspy-cli benchmark [flags]
```

**Flags:**
| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--optimizers` | | Comma-separated optimizer list | all |
| `--datasets` | | Comma-separated dataset list | all |
| `--samples` | `-n` | Samples per benchmark | `20` |
| `--iterations` | `-i` | Benchmark iterations | `3` |
| `--parallel` | `-p` | Parallel execution | `false` |
| `--output` | `-o` | Output directory | `./benchmarks` |

**Examples:**

```bash
# Run full benchmark suite
dspy-cli benchmark

# Benchmark specific optimizers
dspy-cli benchmark --optimizers bootstrap,mipro

# Benchmark with more samples
dspy-cli benchmark --samples 50 --iterations 5

# Parallel benchmarking
dspy-cli benchmark --parallel --output ./results
```

---

### `config` - Manage Configuration

View and manage CLI configuration.

**Usage:**
```bash
dspy-cli config <subcommand> [flags]
```

**Subcommands:**
| Subcommand | Description |
|------------|-------------|
| `show` | Show current configuration |
| `set` | Set configuration value |
| `get` | Get configuration value |
| `reset` | Reset to defaults |

**Examples:**

```bash
# Show configuration
dspy-cli config show

# Set default LLM
dspy-cli config set default_llm gemini-1.5-pro

# Get default dataset
dspy-cli config get default_dataset

# Reset configuration
dspy-cli config reset
```

---

## Global Flags

These flags work with all commands:

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--help` | `-h` | Show help | |
| `--version` | | Show version | |
| `--quiet` | `-q` | Quiet mode (minimal output) | `false` |
| `--debug` | | Enable debug logging | `false` |
| `--no-color` | | Disable colored output | `false` |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Configuration error |
| `4` | LLM error |
| `5` | Timeout |

---

## Environment Variables

The CLI respects these environment variables:

```bash
# LLM Configuration
GEMINI_API_KEY="..."
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."
OLLAMA_BASE_URL="..."

# CLI Defaults
DSPY_CLI_DEFAULT_DATASET="gsm8k"
DSPY_CLI_DEFAULT_OPTIMIZER="mipro"
DSPY_CLI_MAX_EXAMPLES="10"

# Output Options
DSPY_CLI_NO_COLOR="false"
DSPY_CLI_QUIET="false"
DSPY_CLI_OUTPUT_FORMAT="table"

# Performance
DSPY_CLI_TIMEOUT="300"
DSPY_CLI_PARALLEL="false"
```

---

## Configuration File

Create `~/.dspy-cli.yaml` for persistent settings:

```yaml
# Default LLM provider
default_llm: gemini-1.5-pro

# Default dataset
default_dataset: gsm8k

# Default optimizer
default_optimizer: mipro

# Example limits
max_examples: 20

# Output preferences
output_format: table
colored_output: true
verbose: false

# Performance
timeout: 300
parallel: false
```

---

## Advanced Usage

### Scripting with JSON Output

```bash
# Get optimizer recommendations as JSON
recommendations=$(dspy-cli recommend balanced --json)
echo "$recommendations" | jq '.recommended_optimizer'

# Run optimizer and parse results
results=$(dspy-cli try mipro --json --output -)
accuracy=$(echo "$results" | jq '.accuracy')
echo "Accuracy: $accuracy"
```

### Batch Processing

```bash
#!/bin/bash
# test_all_optimizers.sh

optimizers=("bootstrap" "mipro" "gepa" "simba")
dataset="gsm8k"

for opt in "${optimizers[@]}"; do
    echo "Testing $opt..."
    dspy-cli try "$opt" \
        --dataset "$dataset" \
        --max-examples 20 \
        --output "${opt}_results.json" \
        --json
done
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Test Optimizers

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build CLI
        run: |
          cd cmd/dspy-cli
          go build -o dspy-cli

      - name: Run optimizer tests
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          ./cmd/dspy-cli/dspy-cli try bootstrap --max-examples 5
          ./cmd/dspy-cli/dspy-cli try mipro --max-examples 5
```

---

## Troubleshooting

### Common Issues

#### "No API key found"

```bash
# Make sure you've set an API key
export GEMINI_API_KEY="your-key"

# Verify it's set
echo $GEMINI_API_KEY
```

#### "Command not found: dspy-cli"

```bash
# Add $GOPATH/bin to your PATH
export PATH=$PATH:$(go env GOPATH)/bin

# Or run from the build directory
cd cmd/dspy-cli
./dspy-cli list
```

#### Timeout Errors

```bash
# Increase timeout for long-running optimizations
dspy-cli try gepa --timeout 600  # 10 minutes
```

---

## Examples

### Quick Start

```bash
# 1. List available optimizers
dspy-cli list

# 2. Get a recommendation
dspy-cli recommend balanced

# 3. Try the recommended optimizer
dspy-cli try mipro --dataset gsm8k --max-examples 10

# 4. Compare with others
dspy-cli compare bootstrap mipro --max-examples 10
```

### Production Testing

```bash
# Full evaluation with 50 examples
dspy-cli try gepa \
  --dataset gsm8k \
  --max-examples 50 \
  --verbose \
  --output production_results.json \
  --json

# Parallel comparison of top optimizers
dspy-cli compare mipro gepa simba \
  --dataset hotpotqa \
  --max-examples 30 \
  --parallel \
  --output comparison.csv \
  --format csv
```

---

## Next Steps

- **[Configuration Reference →](configuration/)** - Configure the CLI
- **[Getting Started →](../../guides/getting-started/)** - Learn the basics
- **[Optimizers Guide →](../../guides/optimizers/)** - Understand each optimizer
