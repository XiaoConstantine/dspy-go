# DSPy-Go Compatibility Testing Framework

This framework provides side-by-side comparison between the Python DSPy package and the Go dspy-go implementation to verify backwards compatibility from an optimizer's perspective.

## Overview

The compatibility testing framework consists of:

1. **Python DSPy Reference Implementation** (`dspy_comparison.py`)
2. **Go dspy-go Implementation** (`go_comparison.go`) 
3. **Results Comparison Tool** (`compare_results.py`)
4. **Automated Experiment Runner** (`run_experiment.sh`)

## Key Features

### Optimizer Testing
- **BootstrapFewShot**: Tests few-shot learning with bootstrapped demonstrations
- **MIPRO/MIPROv2**: Tests multi-stage instruction prompt optimization with Bayesian optimization

### Compatibility Verification
- API signature compatibility
- Parameter compatibility
- Behavioral consistency
- Performance comparison
- Results accuracy comparison

## Prerequisites

### Python Environment
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- Gemini API key

### Go Environment  
- Go 1.19+
- dspy-go dependencies

### Required Environment Variables
```bash
export GEMINI_API_KEY=your_api_key_here
```

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and navigate to the compatibility test directory:
```bash
cd compatibility_test
```

3. Ensure Go dependencies are available:
```bash
cd ..
go mod tidy
cd compatibility_test
```

That's it! The Python scripts use uv's inline script dependencies, so no separate virtual environment or requirements.txt is needed.

## Usage

### Quick Start
Run the complete compatibility experiment:
```bash
./run_experiment.sh
```

#### Test Specific Optimizers
Test only SIMBA optimizer:
```bash
./run_experiment.sh --optimizer simba
```

Test with custom dataset size:
```bash
./run_experiment.sh --optimizer bootstrap --dataset-size 50
```

Available optimizer options:
- `bootstrap`: BootstrapFewShot only
- `mipro`: MIPRO/MIPROv2 only  
- `simba`: SIMBA only
- `all`: All optimizers (default)

### Manual Execution

#### 1. Run Python DSPy Comparison
```bash
# Test all optimizers
python dspy_comparison.py

# Test specific optimizer
python dspy_comparison.py --optimizer bootstrap --dataset-size 30
```

#### 2. Run Go dspy-go Comparison
```bash
go build -o go_comparison go_comparison.go

# Test all optimizers
./go_comparison

# Test specific optimizer
./go_comparison --optimizer simba --dataset-size 30
```

#### 3. Compare Results
```bash
python compare_results.py
```

## Test Structure

### Dataset
- Simple Q&A pairs (20 examples)
- Split: 15 training, 5 validation
- Questions cover basic facts and calculations

### Metrics
- **Accuracy**: Simple substring matching
- **Compilation Time**: Time to optimize the program
- **Demonstrations**: Number of generated examples

### Optimizers Tested

#### BootstrapFewShot
- **Python**: `dspy.teleprompt.BootstrapFewShot`
- **Go**: `optimizers.BootstrapFewShot`
- **Parameters**: 
  - `max_bootstrapped_demos`: 4
  - `max_labeled_demos`: 4

#### MIPRO/MIPROv2
- **Python**: `dspy.teleprompt.MIPROv2`
- **Go**: `optimizers.MIPRO`
- **Parameters**:
  - `num_trials`: 5
  - `max_bootstrapped_demos`: 3
  - `max_labeled_demos`: 3

#### SIMBA
- **Python**: `dspy.teleprompt.SIMBA`
- **Go**: `optimizers.SIMBA`
- **Parameters**:
  - `batch_size`: 4
  - `max_steps`: 6
  - `num_candidates`: 4
  - `sampling_temperature`: 0.2

## Output Files

### `dspy_comparison_results.json`
Results from Python DSPy implementation:
```json
{
  "dataset_size": 20,
  "model": "gpt-3.5-turbo",
  "bootstrap_fewshot": {
    "optimizer": "BootstrapFewShot",
    "average_score": 0.85,
    "compilation_time": 12.34,
    "demonstrations": [...]
  },
  "mipro_v2": {
    "optimizer": "MIPROv2",
    "average_score": 0.92,
    "compilation_time": 25.67,
    "demonstrations": [...]
  }
}
```

### `go_comparison_results.json`
Results from Go dspy-go implementation:
```json
{
  "dataset_size": 20,
  "model": "gpt-3.5-turbo",
  "bootstrap_fewshot": {
    "optimizer": "BootstrapFewShot",
    "average_score": 0.83,
    "compilation_time": 11.89,
    "demonstrations": [...]
  },
  "mipro": {
    "optimizer": "MIPRO",
    "average_score": 0.90,
    "compilation_time": 24.12,
    "demonstrations": [...]
  }
}
```

### `compatibility_report.json`
Detailed compatibility analysis:
```json
{
  "compatibility_summary": {
    "bootstrap_fewshot_compatible": true,
    "mipro_compatible": true,
    "score_differences_acceptable": true,
    "api_signatures_match": true,
    "behavior_consistent": true
  },
  "recommendations": {
    "critical_issues": [],
    "improvements": [],
    "validation_needed": []
  }
}
```

## Compatibility Criteria

### ✅ Pass Criteria
- Score difference < 0.1 (10%)
- API signatures match
- Same parameter types and defaults
- Consistent behavior patterns

### ⚠️ Warning Criteria
- Score difference 0.1-0.2 (10-20%)
- Minor parameter differences
- Performance variations

### ❌ Fail Criteria
- Score difference > 0.2 (20%)
- API incompatibilities
- Behavioral inconsistencies
- Missing features

## Interpreting Results

### Compatibility Report Sections

1. **Compatibility Summary**: Overall compatibility status
2. **BootstrapFewShot Comparison**: Detailed comparison of few-shot optimizer
3. **MIPRO Comparison**: Detailed comparison of MIPRO optimizer
4. **Recommendations**: Action items for improvement

### Common Issues and Solutions

#### Score Differences
- **Cause**: Different random seeds, LLM variations, implementation differences
- **Solution**: Run multiple trials, use fixed seeds, verify algorithm implementation

#### Time Differences
- **Cause**: Language performance, concurrency differences, LLM call patterns
- **Solution**: Optimize critical paths, implement proper concurrency

#### Demonstration Count Differences
- **Cause**: Different filtering criteria, validation logic
- **Solution**: Align validation functions, verify example generation

## Extending the Framework

### Adding New Optimizers
1. Implement optimizer in both Python and Go
2. Add test cases in respective comparison files
3. Update results comparison logic
4. Add new compatibility criteria

### Adding New Metrics
1. Implement metric in both languages
2. Add to comparison functions
3. Update report generation
4. Add interpretation guidelines

### Adding New Datasets
1. Create dataset in both implementations
2. Ensure consistent format
3. Add dataset-specific metrics
4. Update compatibility criteria

## Troubleshooting

### Common Issues

#### OpenAI API Key
```bash
export OPENAI_API_KEY=your_api_key_here
```

#### Python Dependencies
```bash
pip install -r requirements.txt
```

#### Go Build Issues
```bash
go mod tidy
go build -o go_comparison go_comparison.go
```

#### Permission Issues
```bash
chmod +x run_experiment.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure compatibility tests pass
5. Submit a pull request

## License

This project is licensed under the same license as the main dspy-go project.