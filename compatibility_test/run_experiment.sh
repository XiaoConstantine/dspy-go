#!/bin/bash

set -e

# Parse command line arguments
OPTIMIZER="all"
DATASET_SIZE="20"

while [[ $# -gt 0 ]]; do
    case $1 in
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --dataset-size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--optimizer bootstrap|mipro|simba|copro|all] [--dataset-size N]"
            echo "  --optimizer: Which optimizer to test (default: all)"
            echo "  --dataset-size: Dataset size for testing (default: 20)"
            echo "  Note: CoPro is Go-specific and will only run Go tests"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting DSPy-Go Compatibility Experiment"
echo "🎯 Testing optimizer: $OPTIMIZER"
echo "📊 Dataset size: $DATASET_SIZE"
echo "=" * 50

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY environment variable is not set"
    echo "Please set your Gemini API key:"
    echo "export GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

echo "✅ Gemini API key is set"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv: https://docs.astral.sh/uv/"
    exit 1
fi

echo "✅ uv is available"

# Run Python DSPy comparison using uv
echo "🐍 Running Python DSPy comparison with uv..."
uv run dspy_comparison.py --optimizer "$OPTIMIZER" --dataset-size "$DATASET_SIZE"

# Check if Python comparison was successful
if [ $? -eq 0 ]; then
    echo "✅ Python DSPy comparison completed successfully"
else
    echo "❌ Python DSPy comparison failed"
    exit 1
fi

# Build and run Go comparison
echo "🚀 Building Go comparison..."
cd ..
go build -o compatibility_test/go_comparison compatibility_test/go_comparison.go

echo "🚀 Running Go dspy-go comparison..."
cd compatibility_test
./go_comparison --optimizer "$OPTIMIZER" --dataset-size "$DATASET_SIZE"

# Check if Go comparison was successful
if [ $? -eq 0 ]; then
    echo "✅ Go dspy-go comparison completed successfully"
else
    echo "❌ Go dspy-go comparison failed"
    exit 1
fi

# Run results comparison
echo "📊 Comparing results..."
uv run compare_results.py

# Check if results comparison was successful
if [ $? -eq 0 ]; then
    echo "✅ Results comparison completed successfully"
else
    echo "❌ Results comparison failed"
    exit 1
fi

echo ""
echo "🎉 Compatibility experiment completed successfully!"
echo "📁 Results files generated:"
echo "  - dspy_comparison_results.json (Python DSPy results)"
echo "  - go_comparison_results.json (Go dspy-go results)"
echo "  - compatibility_report.json (Compatibility analysis)"
echo ""
echo "📊 Check the compatibility report for detailed analysis"

# No need to deactivate - uv manages dependencies automatically