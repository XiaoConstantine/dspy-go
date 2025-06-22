# MultiChainComparison Module Example

This example demonstrates the usage of the `MultiChainComparison` module in DSPy-Go, which implements the same functionality as Python DSPy's `multi_chain_comparison.py`.

## Overview

The `MultiChainComparison` module allows you to:

1. **Compare Multiple Reasoning Attempts**: Take several different reasoning approaches for the same problem
2. **Generate Holistic Analysis**: Synthesize the different attempts into a comprehensive evaluation
3. **Improve Decision Making**: Leverage multiple perspectives to arrive at better conclusions

## How It Works

The module:

1. Takes an original signature (inputs → outputs)
2. Dynamically modifies it to accept M reasoning attempts as additional inputs
3. Prepends a "rationale" field to outputs for holistic reasoning
4. Processes multiple completions and formats them as reasoning attempts
5. Uses an internal LLM to compare and synthesize the attempts

## Key Features

- **Configurable Attempt Count (M)**: Specify how many reasoning attempts to compare
- **Temperature Control**: Adjust the creativity/randomness of the synthesis
- **Flexible Input Format**: Accept completions with either "rationale" or "reasoning" fields
- **Automatic Formatting**: Convert completions into structured reasoning attempts
- **Full DSPy Integration**: Works seamlessly with other DSPy-Go modules

## Running the Examples

```bash
# Run the multi-chain comparison examples
go run examples/multi_chain_comparison/main.go -api-key YOUR_API_KEY
```

## Example Scenarios

### 1. Mathematical Reasoning Comparison
Compares 3 different approaches to solving a quadratic equation:
- Algebraic manipulation
- Quadratic formula application  
- Factoring approach

### 2. Text Analysis with Multiple Perspectives
Analyzes sentiment using 4 different analytical perspectives:
- Keyword-based analysis
- Grammatical structure analysis
- Sarcasm/implicit meaning consideration
- Overall message evaluation

### 3. Problem Solving with Different Approaches
Evaluates business strategy using 3 different frameworks:
- Cost reduction focus
- Customer satisfaction focus
- Balanced approach

## Module Configuration

```go
// Create MultiChainComparison with custom settings
multiChain := modules.NewMultiChainComparison(
    signature,           // Your original signature
    3,                  // Number of attempts to compare (M)
    0.7,                // Temperature for synthesis
    // Additional options can be passed here
)
```

## Input Format

The module expects completions in this format:

```go
completions := []map[string]interface{}{
    {
        "rationale": "reasoning description",  // or "reasoning"
        "output1": "value1",
        "output2": "value2",
        // ... other output fields from original signature
    },
    // ... more completions
}
```

## Output Format

The module returns:

```go
{
    "rationale": "Holistic analysis comparing all attempts...",
    "output1": "Synthesized value1",
    "output2": "Synthesized value2",
    // ... other fields from original signature
}
```

## Implementation Details

This Go implementation provides full functionality parity with the Python version:

- ✅ Dynamic signature modification (adding reasoning attempt fields)
- ✅ Rationale field prepending to outputs
- ✅ Completion processing and formatting
- ✅ Configurable attempt count (M)
- ✅ Temperature control
- ✅ Error handling and validation
- ✅ Module interface compliance
- ✅ Comprehensive test coverage

## Use Cases

The `MultiChainComparison` module is particularly useful for:

- **Decision Making**: Compare different decision-making frameworks
- **Problem Solving**: Evaluate multiple solution approaches
- **Analysis Tasks**: Synthesize different analytical perspectives
- **Reasoning Tasks**: Compare various reasoning methodologies
- **Quality Assurance**: Cross-validate results from different approaches