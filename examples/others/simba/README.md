# SIMBA Optimizer Example

This example demonstrates the **SIMBA (Stochastic Introspective Mini-Batch Ascent)** optimizer, showcasing its unique introspective learning capabilities and stochastic optimization approach.

## What SIMBA Does

SIMBA is an advanced prompt optimizer that:

- **🧠 Introspective Learning**: Analyzes its own optimization progress and provides recommendations
- **🎯 Mini-batch Processing**: Uses stochastic mini-batches for efficient optimization
- **🌡️ Temperature-controlled Sampling**: Balances exploration vs exploitation dynamically
- **📊 Pattern Detection**: Identifies optimization trends and suggests adjustments
- **🔄 Adaptive Convergence**: Automatically detects when optimization should stop

## How to Run

1. **Set up your API key** for your LLM provider (e.g., Google Gemini):
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **Run the example**:
   ```bash
   go run main.go -api-key="your-api-key-here"
   ```

   Or use the environment variable:
   ```bash
   go run main.go -api-key="$GOOGLE_API_KEY"
   ```

## What You'll See

The example will show:

### 1. **Initial Setup**
- Program configuration with reasoning module
- SIMBA optimizer configuration with introspective learning enabled

### 2. **Optimization Process**
- Step-by-step optimization progress
- Mini-batch evaluation results
- Temperature decay over iterations
- Introspective analysis every 2 steps (configurable)

### 3. **Introspective Insights**
- **Performance Analysis**: Score improvements and optimization trends
- **Pattern Detection**: Identified optimization patterns (e.g., "Strong upward trend", "Convergence detected")
- **Recommendations**: Actionable suggestions (e.g., "Increase sampling temperature", "Try different instruction variations")
- **Convergence Analysis**: Automatic detection of optimization completion

### 4. **Results Comparison**
- Original vs optimized program performance
- Detailed accuracy metrics
- Improvement analysis

## Example Output

```
Starting SIMBA optimization with introspective learning...
SIMBA will:
  • Generate instruction variants using LLM-based perturbation
  • Evaluate candidates on mini-batches for efficiency  
  • Use temperature-controlled sampling for exploration
  • Perform introspective analysis every 2 steps
  • Detect convergence automatically

Starting SIMBA optimization with config: batch_size=4, max_steps=6, num_candidates=4
Initial program score: 0.4000
New best score: 0.6000 (improvement: +0.2000)
Optimization converged at step 4
SIMBA optimization completed: final_score=0.6000, steps=5, duration=15.2s

=== SIMBA INTROSPECTIVE ANALYSIS ===
Optimization Statistics:
  • Total steps executed: 5/6
  • Final best score: 0.6000
  • Total candidates evaluated: 20
  • Optimization duration: 15.2s

Optimization Progress:
  Step 0: Score=0.4000, Improvement=0.0000, Temperature=0.300, Batch=4
  Step 2: Score=0.5000, Improvement=0.1000, Temperature=0.271, Batch=4
    Introspection: Performance shows steady improvement with good exploration balance...
  Step 4: Score=0.6000, Improvement=0.1000, Temperature=0.245, Batch=4
    Introspection: Strong upward trend detected with high magnitude improvements...

Convergence Analysis:
  ✓ Converged: Average improvement (0.000500) below threshold (0.001000)

=== SIMBA OPTIMIZATION RESULTS ===
Original accuracy:  40.0%
Optimized accuracy: 60.0%
Improvement: +20.0% (SIMBA optimization successful!)
```

## Key Features Demonstrated

### 1. **Stochastic Mini-batch Optimization**
- Uses small batches (4 examples) for efficient evaluation
- Random sampling provides good optimization signal
- Faster than full dataset evaluation

### 2. **Introspective Analysis**
- **Pattern Recognition**: Identifies trends like "Strong upward trend", "Plateau detected"
- **Performance Metrics**: Tracks improvement magnitude and volatility
- **Convergence Detection**: Uses statistical analysis to determine stopping point

### 3. **Temperature-controlled Exploration**
- Starts with higher exploration (temperature=0.3)
- Gradually focuses on exploitation as optimization progresses
- 30% chance for exploration to escape local optima

### 4. **Adaptive Instruction Generation**
- Generates diverse instruction variants using LLM perturbation
- Selects best candidates using probabilistic sampling
- Maintains program structure while optimizing instructions

## Comparison with Other Optimizers

| Feature | SIMBA | MIPRO | BootstrapFewShot |
|---------|-------|-------|------------------|
| **Introspection** | ✅ Advanced | ❌ None | ❌ None |
| **Mini-batch** | ✅ Stochastic | ✅ Fixed | ❌ Full dataset |
| **Temperature Control** | ✅ Dynamic | ❌ None | ❌ None |
| **Pattern Detection** | ✅ Statistical | ❌ None | ❌ None |
| **Convergence Detection** | ✅ Automatic | ✅ Trial-based | ❌ Fixed iterations |

## Configuration Options

SIMBA can be configured with various options:

```go
optimizer := optimizers.NewSIMBA(
    optimizers.WithSIMBABatchSize(8),        // Mini-batch size
    optimizers.WithSIMBAMaxSteps(10),        // Maximum optimization steps
    optimizers.WithSIMBANumCandidates(6),    // Candidates per iteration
    optimizers.WithSamplingTemperature(0.2), // Exploration temperature
)
```

## Use Cases

SIMBA is particularly effective for:

- **Complex reasoning tasks** where instruction quality matters
- **Resource-constrained optimization** (mini-batch efficiency)
- **Iterative refinement** where multiple attempts improve results
- **Tasks requiring exploration** of instruction space
- **Scenarios where optimization insights** are valuable for debugging

## Next Steps

Try modifying the configuration to see how it affects optimization:

1. **Increase batch size** for more stable but slower optimization
2. **Adjust temperature** for different exploration/exploitation balance
3. **Change max steps** to see longer optimization trajectories
4. **Use different datasets** to see how SIMBA adapts to various tasks

The introspective analysis will provide insights into how these changes affect the optimization process!