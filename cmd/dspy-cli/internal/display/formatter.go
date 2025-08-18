package display

import (
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
)

const (
	ColorReset  = "\033[0m"
	ColorBold   = "\033[1m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
)

func FormatOptimizerList() string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%s%sAvailable DSPy-Go Optimizers%s\n", ColorBold, ColorBlue, ColorReset))
	output.WriteString(strings.Repeat("=", 50) + "\n\n")

	optimizerNames := optimizers.ListAll()

	for _, name := range optimizerNames {
		info, _ := optimizers.GetOptimizer(name)
		complexity := getComplexityColor(info.Complexity)

		output.WriteString(fmt.Sprintf("%s%s%s%s\n", ColorBold, ColorGreen, info.Name, ColorReset))
		output.WriteString(fmt.Sprintf("  %s\n", info.Description))
		output.WriteString(fmt.Sprintf("  %sComplexity:%s %s%s%s | %sCost:%s %s\n",
			ColorCyan, ColorReset, complexity, info.Complexity, ColorReset,
			ColorCyan, ColorReset, info.ComputeCost))
		output.WriteString(fmt.Sprintf("  %sUse:%s %s\n", ColorYellow, ColorReset, info.UseCase))
		output.WriteString("\n")
	}

	output.WriteString(fmt.Sprintf("%sTip:%s Use 'dspy-cli describe <optimizer>' for detailed information\n",
		ColorPurple, ColorReset))

	return output.String()
}

func FormatOptimizerDetails(info optimizers.OptimizerInfo) string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%s%s%s%s\n", ColorBold, ColorBlue, info.Name, ColorReset))
	output.WriteString(strings.Repeat("=", len(info.Name)+10) + "\n\n")

	output.WriteString(fmt.Sprintf("%sDescription:%s\n%s\n\n", ColorBold, ColorReset, info.Description))

	output.WriteString(fmt.Sprintf("%sCharacteristics:%s\n", ColorBold, ColorReset))
	complexity := getComplexityColor(info.Complexity)
	output.WriteString(fmt.Sprintf("  • %sComplexity:%s %s%s%s\n", ColorCyan, ColorReset, complexity, info.Complexity, ColorReset))
	output.WriteString(fmt.Sprintf("  • %sCompute Cost:%s %s\n", ColorCyan, ColorReset, info.ComputeCost))
	output.WriteString(fmt.Sprintf("  • %sConvergence:%s %s\n", ColorCyan, ColorReset, info.Convergence))

	output.WriteString(fmt.Sprintf("\n%sBest For:%s\n", ColorBold, ColorReset))
	for _, useCase := range info.BestFor {
		output.WriteString(fmt.Sprintf("  • %s\n", useCase))
	}

	output.WriteString(fmt.Sprintf("\n%sExample Use Case:%s\n%s\n", ColorBold, ColorReset, info.Example))

	output.WriteString(fmt.Sprintf("\n%sTry it:%s\n", ColorBold, ColorReset))
	output.WriteString(fmt.Sprintf("  dspy-cli try %s\n", strings.ToLower(info.Name)))

	return output.String()
}

func FormatRecommendations(recommendations []string) string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%s%sRecommended Optimizers%s\n", ColorBold, ColorGreen, ColorReset))
	output.WriteString(strings.Repeat("=", 30) + "\n\n")

	for i, name := range recommendations {
		info, _ := optimizers.GetOptimizer(name)
		output.WriteString(fmt.Sprintf("%d. %s%s%s\n", i+1, ColorBold, info.Name, ColorReset))
		output.WriteString(fmt.Sprintf("   %s\n", info.Description))
		output.WriteString(fmt.Sprintf("   %sComplexity:%s %s | %sCost:%s %s\n\n",
			ColorCyan, ColorReset, info.Complexity,
			ColorCyan, ColorReset, info.ComputeCost))
	}

	return output.String()
}

func getComplexityColor(complexity string) string {
	switch strings.ToLower(complexity) {
	case "low":
		return ColorGreen
	case "medium", "medium-high":
		return ColorYellow
	case "high", "very high":
		return ColorRed
	default:
		return ColorReset
	}
}
