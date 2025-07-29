# GEPA (Generative Evolutionary Prompt Adaptation) Example

This example demonstrates the GEPA optimizer, a state-of-the-art evolutionary approach to prompt optimization that implements the key innovations from the research paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/pdf/2507.19457).

## Overview

GEPA represents a breakthrough in prompt optimization by combining:

- **Multi-objective Pareto optimization** across 7 dimensions
- **LLM-based self-reflection** for prompt analysis and critique  
- **Semantic diversity metrics** using language model similarity
- **Elite archive management** preserving diverse high-quality solutions
- **Real-time system monitoring** with context-aware performance tracking

## Key Features Demonstrated

### 🧬 **Evolutionary Multi-Objective Optimization**
- Population-based evolution with 7-dimensional fitness evaluation
- Success rate, quality, efficiency, robustness, generalization, diversity, innovation
- Pareto-based selection maintaining diverse trade-off solutions

### 🤔 **LLM-Based Self-Reflection** 
- Language models analyze and critique their own prompt performance
- Structured feedback with strengths, weaknesses, and improvement suggestions
- Reflection-driven evolution every few generations

### 🎯 **Pareto Archive of Elite Solutions**
- Maintains diverse high-quality solutions across generations
- Each solution optimized for different objective trade-offs
- Users can select solutions based on their specific needs

### 📊 **Real-Time System Monitoring**
- Context-aware performance tracking with actual system metrics
- CPU load, memory usage, and concurrency monitoring
- Adaptive parameter adjustment based on system state

## Usage

### Basic Usage

```bash
# Run with Gemini API key
go run main.go -api-key YOUR_GEMINI_API_KEY

# Use different dataset
go run main.go -api-key YOUR_KEY -dataset hotpotqa

# Customize evolution parameters
go run main.go -api-key YOUR_KEY -population 20 -generations 10

# Enable verbose logging to see evolution details
go run main.go -api-key YOUR_KEY -verbose
```

### Command Line Options

- `-api-key`: **Required** - Your Google Gemini API key
- `-dataset`: Dataset to use (`gsm8k` or `hotpotqa`, default: `gsm8k`)
- `-population`: Population size for evolution (default: 12)
- `-generations`: Maximum number of generations (default: 8)
- `-verbose`: Enable detailed logging of the evolutionary process

## Example Output

```
🧬 Starting GEPA (Generative Evolutionary Prompt Adaptation) Example
Dataset: gsm8k, Population: 12, Generations: 8
✅ Configured Gemini Flash model for GEPA optimization
📊 Loaded 50 training examples from gsm8k dataset

🚀 Starting GEPA optimization - this demonstrates the paper's key innovations:
   • Multi-objective Pareto optimization across 7 dimensions
   • LLM-based self-reflection and critique (every 2 generations)
   • Semantic diversity metrics using LLM similarity
   • Elite archive management with crowding distance
   • Real-time system monitoring and context awareness

🧬 GEPA Evolution: 2/8 (25.0%) - Generation 2
🧬 GEPA Evolution: 4/8 (50.0%) - Generation 4
🧬 GEPA Evolution: 6/8 (75.0%) - Generation 6
🧬 GEPA Evolution: 8/8 (100.0%) - Generation 8

✅ GEPA optimization completed in 2m15s

🏆 GEPA Optimization Results:
==================================================
Dataset: gsm8k
Total Generations: 8
Optimization Duration: 2m15s
Best Fitness Achieved: 0.8400

🥇 Best Evolved Prompt:
Generation: 6
Instruction: Solve the math problem step by step, showing detailed calculations and checking your work at each stage.

📊 Pareto Archive (Elite Solutions):
Elite solutions preserved: 5
These solutions represent different trade-offs:
• High accuracy vs fast execution
• Quality vs efficiency balance
• Robustness vs generalization optimization

🎯 GEPA Paper's Key Innovation - Pareto Archive Analysis:
=======================================================
The archive contains 5 elite solutions, each optimized for different objectives:

Elite Solution 1:
  Generation: 6
  Fitness: 0.8400
  Instruction: Solve the math problem step by step, showing detailed calculations and checking your work at each stage.
  Optimization Focus: High accuracy with detailed reasoning

Elite Solution 2:
  Generation: 4  
  Fitness: 0.7800
  Instruction: Solve this math problem efficiently with clear steps.
  Optimization Focus: Fast execution with good quality

Elite Solution 3:
  Generation: 5
  Fitness: 0.7600
  Instruction: Carefully analyze the problem, solve systematically, and verify the answer.
  Optimization Focus: Robust performance across diverse inputs

💡 This demonstrates GEPA's key advantage over single-objective optimizers:
   Instead of one 'best' solution, you get a diverse set of elite solutions
   optimized for different trade-offs, allowing you to choose based on your needs.
```

## Paper Implementation Details

This example faithfully implements the key contributions from the GEPA research paper:

### **Natural Language Reflection**
- Uses LLMs to analyze prompt performance in natural language
- Generates structured critique with strengths, weaknesses, suggestions
- Reflection-driven evolution every few generations

### **Multi-Objective Optimization**
- 7-dimensional fitness evaluation across multiple objectives
- Pareto-based selection maintaining solution diversity
- Crowding distance for diversity preservation

### **System-Level Trajectory Sampling**
- Captures complete execution traces for analysis
- Context-aware performance tracking
- Real-time system monitoring integration

### **Semantic Diversity Assessment**  
- LLM-based similarity for true semantic diversity
- Beyond simple string matching to meaningful prompt differences
- Enhanced with cosine similarity and Jaccard coefficients

## Datasets Supported

### GSM8K (Grade School Math 8K)
- **Task**: Mathematical reasoning and problem solving
- **Metric**: Exact answer matching with numerical extraction
- **Signature**: Question → Reasoning + Answer

### HotPotQA  
- **Task**: Multi-hop question answering
- **Metric**: String similarity with partial credit
- **Signature**: Question → Reasoning + Answer

## Configuration Options

The example demonstrates advanced GEPA configuration:

```go
config := &optimizers.GEPAConfig{
    PopulationSize:    12,              // Population for evolution
    MaxGenerations:    8,               // Maximum generations
    SelectionStrategy: "adaptive_pareto", // Multi-objective selection
    MutationRate:      0.3,             // Mutation probability
    CrossoverRate:     0.7,             // Crossover probability
    ReflectionFreq:    2,               // Reflection every 2 generations
    ReflectionDepth:   3,               // Depth of reflection analysis
    // ... additional parameters
}
```

## Performance Considerations

- **Population Size**: Larger populations explore more diverse solutions but require more computation
- **Generations**: More generations allow for better evolution but increase runtime
- **Reflection Frequency**: More frequent reflection improves quality but increases LLM usage
- **Dataset Size**: Larger training sets improve optimization but slow evaluation

## Comparison with Other Optimizers

| Aspect | GEPA | MIPRO | SIMBA | BootstrapFewShot |
|--------|------|-------|-------|------------------|
| **Approach** | Multi-objective evolution | TPE search | Introspective learning | Example selection |
| **Objectives** | 7-dimensional Pareto | Single objective | Self-analysis | Few-shot quality |
| **Reflection** | LLM-based critique | None | Self-introspection | None |
| **Diversity** | Semantic + archive | Limited | Batch-based | Example-based |
| **Complexity** | High (comprehensive) | Medium | Medium-High | Low |
| **Quality** | Highest | High | High | Medium |

## Requirements

- Go 1.19 or later
- Google Gemini API key
- Internet connection for dataset loading and LLM calls
- Sufficient computation time (GEPA is thorough but takes longer than simpler optimizers)

## Tips for Best Results

1. **Start Small**: Begin with smaller populations and fewer generations for testing
2. **Monitor Progress**: Use `-verbose` flag to understand the evolutionary process
3. **Dataset Quality**: Higher quality training examples lead to better optimization
4. **Patience**: GEPA's comprehensive approach takes time but delivers superior results
5. **Archive Analysis**: Examine the Pareto archive to understand different solution trade-offs

## Paper Reference

This implementation is based on:
- **Paper**: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- **URL**: https://arxiv.org/pdf/2507.19457
- **Key Innovation**: Natural language reflection for prompt optimization
- **Performance**: 10% average improvement over RL methods, 35x fewer rollouts

The example demonstrates all major paper contributions while providing a practical, usable implementation for real-world prompt optimization tasks.