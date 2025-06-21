# Refine Module Example

This example demonstrates the `Refine` module in dspy-go, which improves prediction quality through iterative refinement with reward-based selection.

## Overview

The `Refine` module runs a base module multiple times with varying temperatures and selects the best output based on a custom reward function. This is particularly useful for:

- **Quality Improvement**: Get better outputs by trying multiple approaches
- **Consistency**: Achieve more reliable results through multiple attempts  
- **Optimization**: Fine-tune outputs based on specific quality criteria

## Key Features

### 1. **Temperature Variation**
- Automatically generates diverse temperature settings
- Balances exploration (high temp) and exploitation (low temp)
- Configurable number of refinement attempts

### 2. **Custom Reward Functions**
- Define quality metrics specific to your use case
- Higher scores indicate better predictions
- Early termination when threshold is met

### 3. **Backwards Compatibility**
- API compatible with Python DSPy's Refine module
- Same constructor parameters and method signatures
- Easy migration from Python implementations

## Usage Examples

### Basic Usage

```go
// Create base module
signature := core.NewSignature(...)
predictor := modules.NewPredict(signature)

// Define reward function
rewardFn := func(inputs, outputs map[string]interface{}) float64 {
    // Return score 0.0-1.0 based on output quality
    return calculateQuality(outputs)
}

// Create refine module
config := modules.RefineConfig{
    N:         3,        // Try up to 3 attempts
    RewardFn:  rewardFn, // Quality evaluation function
    Threshold: 0.8,      // Stop early if score >= 0.8
}
refiner := modules.NewRefine(predictor, config)

// Use like any other module
outputs, err := refiner.Process(ctx, inputs)
```

### Python DSPy Compatibility

```python
# Python DSPy
refine = dspy.Refine(module=qa, N=3, reward_fn=check_fn, threshold=0.8)
result = refine(question="What is X?")
```

```go
// dspy-go (equivalent)
config := modules.RefineConfig{N: 3, RewardFn: checkFn, Threshold: 0.8}
refine := modules.NewRefine(qa, config)
result, _ := refine.Process(ctx, map[string]interface{}{"question": "What is X?"})
```

## Running the Examples

1. **Set up environment:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key"
   ```

2. **Run the example:**
   ```bash
   cd examples/refine
   go run main.go
   ```

## Example Scenarios

### 1. Math Problem Solving
- **Reward Function**: Checks for clear reasoning, mathematical operations, numerical answers
- **Goal**: Get accurate solutions with step-by-step explanations
- **Threshold**: High (0.8) for mathematical accuracy

### 2. Creative Writing  
- **Reward Function**: Evaluates creativity, descriptive language, narrative structure
- **Goal**: Generate engaging, well-structured stories
- **Threshold**: Moderate (0.7) to allow creative flexibility

### 3. Question Answering
- **Reward Function**: Assesses comprehensiveness, context usage, confidence indicators
- **Goal**: Provide accurate, well-supported answers
- **Threshold**: High (0.8) for factual accuracy

## Reward Function Design

Good reward functions should:

1. **Be Specific**: Target the exact quality you want to improve
2. **Be Scaled**: Return values between 0.0 and 1.0  
3. **Be Fast**: Avoid expensive computations during refinement
4. **Be Stable**: Consistent scoring for similar outputs

### Example Reward Functions

```go
// Length-based reward
lengthReward := func(inputs, outputs map[string]interface{}) float64 {
    text := outputs["answer"].(string)
    if len(text) > 100 {
        return 1.0
    }
    return float64(len(text)) / 100.0
}

// Keyword-based reward  
keywordReward := func(inputs, outputs map[string]interface{}) float64 {
    text := strings.ToLower(outputs["answer"].(string))
    score := 0.0
    keywords := []string{"because", "therefore", "however", "specifically"}
    for _, keyword := range keywords {
        if strings.Contains(text, keyword) {
            score += 0.25
        }
    }
    return score
}

// Multi-criteria reward
combinedReward := func(inputs, outputs map[string]interface{}) float64 {
    answer := outputs["answer"].(string)
    
    // Length component (0-0.3)
    lengthScore := math.Min(float64(len(answer))/200.0, 0.3)
    
    // Completeness component (0-0.4)  
    completeScore := 0.0
    if strings.Contains(answer, ".") { completeScore += 0.2 }
    if len(strings.Fields(answer)) > 10 { completeScore += 0.2 }
    
    // Relevance component (0-0.3)
    relevanceScore := calculateRelevance(inputs["question"], answer)
    
    return lengthScore + completeScore + relevanceScore
}
```

## Configuration Options

```go
type RefineConfig struct {
    N         int             // Number of refinement attempts
    RewardFn  RewardFunction  // Quality evaluation function  
    Threshold float64         // Early termination threshold
    FailCount *int           // Max failures before giving up (optional)
}
```

## Advanced Features

### OfferFeedback Module
Generate advice for improving module performance:

```go
feedback := modules.NewOfferFeedback()
advice, _ := feedback.Process(ctx, map[string]interface{}{
    "program_inputs":   inputs,
    "program_outputs":  outputs, 
    "reward_value":     "0.3",
    "target_threshold": "0.8",
})
```

### Dynamic Configuration
Update refinement settings at runtime:

```go
refiner.UpdateConfig(modules.RefineConfig{
    N:         5,           // Increase attempts
    Threshold: 0.9,         // Raise standards
    RewardFn:  newRewardFn, // Change criteria
})
```

## Best Practices

1. **Start Simple**: Begin with basic reward functions and tune iteratively
2. **Balance Attempts**: More attempts = better quality but higher cost
3. **Set Realistic Thresholds**: Too high may prevent early termination
4. **Monitor Performance**: Track reward scores to optimize your functions
5. **Cache Expensive Rewards**: Store results if reward calculation is costly

## Troubleshooting

### Common Issues

1. **No Early Termination**: Lower your threshold or improve reward function
2. **All Attempts Fail**: Check that base module works independently  
3. **Poor Quality**: Adjust reward function to better capture desired qualities
4. **Slow Performance**: Reduce number of attempts or optimize reward function

### Debugging Tips

```go
// Add logging to reward function
rewardFn := func(inputs, outputs map[string]interface{}) float64 {
    score := calculateScore(outputs)
    log.Printf("Reward: %.3f for output: %s", score, outputs["answer"])
    return score
}

// Check intermediate results
config := modules.RefineConfig{N: 1, ...} // Single attempt for testing
```

## Performance Considerations

- **Cost**: Each attempt uses LLM tokens - balance quality vs. expense
- **Latency**: More attempts = longer response time
- **Concurrency**: Refinement attempts are sequential, not parallel
- **Memory**: Large outputs consume memory during refinement

## Related Modules

- **[Predict](../predict/)**: Base prediction module
- **[ChainOfThought](../chain_of_thought/)**: Step-by-step reasoning  
- **[BestOfN](../best_of_n/)**: Alternative quality improvement approach
- **[Retry](../retry/)**: Error recovery and retry mechanisms