package viewer

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ViewLog displays the log with applied filters, auto-detecting format.
func ViewLog(data *LogData, cfg Config) {
	switch data.Format {
	case FormatRLM:
		viewRLMLog(data, cfg)
	case FormatDSPy:
		viewDSPyLog(data, cfg)
	default:
		fmt.Printf("%sUnknown log format%s\n", Red, Reset)
	}
}

// viewRLMLog displays RLM format logs.
func viewRLMLog(data *LogData, cfg Config) {
	iterations := FilterIterations(data.Iterations, cfg)

	if cfg.FinalOnly {
		PrintFinalOnly(data, iterations)
		return
	}

	if cfg.Stats {
		PrintDetailedStats(data)
		return
	}

	PrintHeader(data.Filename, data.Metadata, data.Format)

	var totalTokens TokenCount
	for _, iter := range iterations {
		tokens := PrintIteration(iter, cfg.Compact, cfg.Search)
		totalTokens.Prompt += tokens.Prompt
		totalTokens.Completion += tokens.Completion
	}

	PrintSummary(iterations, totalTokens.Prompt, totalTokens.Completion)
}

// viewDSPyLog displays native DSPy format logs.
func viewDSPyLog(data *LogData, cfg Config) {
	filtered := FilterEvents(data, cfg)

	if cfg.Stats {
		PrintDSPyDetailedStats(data)
		return
	}

	PrintDSPyHeader(data.Filename, data.Session, data.Format)

	// Print modules
	if len(filtered.Modules) > 0 {
		fmt.Printf("\n%s── Modules ──%s\n", Bold, Reset)
		for _, mod := range filtered.Modules {
			PrintModule(mod, cfg.Compact, cfg.Search)
		}
	}

	// Print LLM calls
	if len(filtered.LLMCalls) > 0 {
		fmt.Printf("\n%s── LLM Calls ──%s\n", Bold, Reset)
		for i, call := range filtered.LLMCalls {
			PrintLLMCall(call, i+1, cfg.Compact, cfg.Search)
		}
	}

	// Print code executions
	if len(filtered.CodeExecs) > 0 {
		fmt.Printf("\n%s── Code Executions ──%s\n", Bold, Reset)
		for _, exec := range filtered.CodeExecs {
			PrintCodeExec(exec, cfg.Compact, cfg.Search)
		}
	}

	// Print tool calls
	if len(filtered.ToolCalls) > 0 {
		fmt.Printf("\n%s── Tool Calls ──%s\n", Bold, Reset)
		for _, tool := range filtered.ToolCalls {
			PrintToolCall(tool, cfg.Compact)
		}
	}

	// Print errors
	if len(filtered.Errors) > 0 {
		fmt.Printf("\n%s── Errors ──%s\n", BoldRed, Reset)
		for _, err := range filtered.Errors {
			PrintError(err)
		}
	}

	PrintDSPySummary(data)
}

// PrintHeader displays session metadata for RLM format.
func PrintHeader(filename string, meta *Metadata, format LogFormat) {
	fmt.Printf("\n%s%s DSPy Log Viewer %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("%sFile:%s %s\n", Dim, Reset, filename)
	fmt.Printf("%sFormat:%s %s\n", Dim, Reset, format.String())

	if meta != nil {
		fmt.Printf("%sModel:%s %s\n", Dim, Reset, meta.RootModel)
		fmt.Printf("%sBackend:%s %s\n", Dim, Reset, meta.Backend)
		fmt.Printf("%sMax Iterations:%s %d\n", Dim, Reset, meta.MaxIterations)

		if meta.Query != "" {
			fmt.Printf("%sQuery:%s %s\n", Dim, Reset, Truncate(meta.Query, 100))
		}
		if meta.Context != "" {
			fmt.Printf("%sContext:%s %s\n", Dim, Reset, Truncate(meta.Context, 100))
		}

		if ts, err := time.Parse(time.RFC3339Nano, meta.Timestamp); err == nil {
			fmt.Printf("%sStarted:%s %s\n", Dim, Reset, ts.Format("2006-01-02 15:04:05"))
		}
	}
	fmt.Println()
}

// PrintDSPyHeader displays session metadata for native DSPy format.
func PrintDSPyHeader(filename string, session *SessionData, format LogFormat) {
	fmt.Printf("\n%s%s DSPy Log Viewer %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("%sFile:%s %s\n", Dim, Reset, filename)
	fmt.Printf("%sFormat:%s %s\n", Dim, Reset, format.String())

	if session != nil {
		fmt.Printf("%sTrace ID:%s %s\n", Dim, Reset, session.TraceID)
		fmt.Printf("%sStarted:%s %s\n", Dim, Reset, session.StartTime.Format("2006-01-02 15:04:05"))

		// Print any metadata from session data
		for k, v := range session.Metadata {
			if k != "start_time" {
				fmt.Printf("%s%s:%s %v\n", Dim, k, Reset, v)
			}
		}
	}
	fmt.Println()
}

// PrintIteration renders a single iteration with colors (RLM format).
func PrintIteration(iter Iteration, compact bool, searchQuery string) TokenCount {
	var tokens TokenCount

	// Iteration header
	fmt.Printf("%s┌─ Iteration %d %s", BoldYellow, iter.Iteration, Reset)
	if iter.IterationTime > 0 {
		fmt.Printf("%s(%.2fs)%s", Dim, iter.IterationTime, Reset)
	}

	// Final answer indicator
	if iter.FinalAnswer != nil {
		fmt.Printf(" %s[FINAL]%s", BoldGreen, Reset)
	}

	// Error indicator
	hasError := false
	for _, block := range iter.CodeBlocks {
		if block.Result.Stderr != "" {
			hasError = true
			break
		}
	}
	if hasError {
		fmt.Printf(" %s[ERROR]%s", BoldRed, Reset)
	}

	fmt.Println()

	// Response preview
	if !compact && iter.Response != "" {
		fmt.Printf("%s│%s %sResponse:%s\n", Yellow, Reset, Dim, Reset)
		response := iter.Response
		if searchQuery != "" {
			response = HighlightSearch(response, searchQuery)
		}
		PrintIndented(response, "│   ", 500)
	}

	// Code blocks
	for i, block := range iter.CodeBlocks {
		fmt.Printf("%s│%s\n", Yellow, Reset)
		fmt.Printf("%s├─ Code Block #%d%s", BoldBlue, i+1, Reset)
		if block.Result.ExecutionTime > 0 {
			fmt.Printf(" %s(%.2fs)%s", Dim, block.Result.ExecutionTime, Reset)
		}
		fmt.Println()

		// Code
		fmt.Printf("%s│%s  %s┌─ Code:%s\n", Yellow, Reset, Blue, Reset)
		code := block.Code
		if searchQuery != "" {
			code = HighlightSearch(code, searchQuery)
		}
		PrintCodeBlock(code, "│  │ ")

		// Output
		if block.Result.Stdout != "" {
			fmt.Printf("%s│%s  %s├─ Output:%s\n", Yellow, Reset, Green, Reset)
			stdout := block.Result.Stdout
			if searchQuery != "" {
				stdout = HighlightSearch(stdout, searchQuery)
			}
			PrintIndented(stdout, "│  │ ", 300)
		}
		if block.Result.Stderr != "" {
			fmt.Printf("%s│%s  %s├─ Stderr:%s\n", Yellow, Reset, Red, Reset)
			PrintIndented(block.Result.Stderr, "│  │ ", 300)
		}

		// Locals
		if len(block.Result.Locals) > 0 {
			fmt.Printf("%s│%s  %s├─ Locals:%s\n", Yellow, Reset, Magenta, Reset)
			for k, v := range block.Result.Locals {
				vStr := fmt.Sprintf("%v", v)
				fmt.Printf("%s│%s  │   %s%s%s = %s\n", Yellow, Reset, Cyan, k, Reset, Truncate(vStr, 80))
			}
		}

		// Sub-LLM calls
		if len(block.Result.RLMCalls) > 0 {
			fmt.Printf("%s│%s  %s└─ Sub-LLM Calls (%d):%s\n", Yellow, Reset, Magenta, len(block.Result.RLMCalls), Reset)
			for j, call := range block.Result.RLMCalls {
				tokens.Prompt += call.PromptTokens
				tokens.Completion += call.CompletionTokens

				fmt.Printf("%s│%s      %s[%d]%s ", Yellow, Reset, Dim, j+1, Reset)
				fmt.Printf("%s%.2fs%s", Dim, call.ExecutionTime, Reset)
				if call.PromptTokens > 0 || call.CompletionTokens > 0 {
					fmt.Printf(" %s(%d→%d tokens)%s", Dim, call.PromptTokens, call.CompletionTokens, Reset)
				}
				fmt.Println()

				if !compact {
					fmt.Printf("%s│%s        %sPrompt:%s %s\n", Yellow, Reset, Dim, Reset, Truncate(call.Prompt, 100))
					fmt.Printf("%s│%s        %sResponse:%s %s\n", Yellow, Reset, Dim, Reset, Truncate(call.Response, 100))
				}
			}
		}
	}

	// Final answer
	if iter.FinalAnswer != nil {
		fmt.Printf("%s│%s\n", Yellow, Reset)
		fmt.Printf("%s└─ Final Answer:%s\n", BoldGreen, Reset)
		switch v := iter.FinalAnswer.(type) {
		case string:
			PrintIndented(v, "   ", 500)
		case []any:
			if len(v) == 2 {
				fmt.Printf("   %s%v%s = %s\n", Cyan, v[0], Reset, Truncate(fmt.Sprintf("%v", v[1]), 200))
			}
		default:
			fmt.Printf("   %v\n", v)
		}
	}

	fmt.Println()
	return tokens
}

// PrintModule renders a module execution (DSPy format).
func PrintModule(mod ModuleData, compact bool, searchQuery string) {
	statusColor := Green
	statusIcon := "✓"
	if !mod.Success {
		statusColor = Red
		statusIcon = "✗"
	}

	fmt.Printf("%s┌─ %s:%s %s%s%s", BoldBlue, mod.ModuleType, Reset, Bold, mod.ModuleName, Reset)
	fmt.Printf(" %s%s%s", statusColor, statusIcon, Reset)
	if mod.DurationMs > 0 {
		fmt.Printf(" %s(%dms)%s", Dim, mod.DurationMs, Reset)
	}
	fmt.Println()

	if mod.Signature != "" {
		fmt.Printf("%s│%s %sSignature:%s %s\n", Blue, Reset, Dim, Reset, Truncate(mod.Signature, 80))
	}

	if mod.TotalTokens > 0 {
		fmt.Printf("%s│%s %sTokens:%s %d (%d LLM calls)\n", Blue, Reset, Dim, Reset, mod.TotalTokens, mod.LLMCalls)
	}

	if !compact {
		if len(mod.Inputs) > 0 {
			fmt.Printf("%s│%s %sInputs:%s\n", Blue, Reset, Dim, Reset)
			for k, v := range mod.Inputs {
				vStr := Truncate(fmt.Sprintf("%v", v), 80)
				if searchQuery != "" {
					vStr = HighlightSearch(vStr, searchQuery)
				}
				fmt.Printf("%s│%s   %s%s%s: %s\n", Blue, Reset, Cyan, k, Reset, vStr)
			}
		}
		if len(mod.Outputs) > 0 {
			fmt.Printf("%s│%s %sOutputs:%s\n", Blue, Reset, Dim, Reset)
			for k, v := range mod.Outputs {
				vStr := Truncate(fmt.Sprintf("%v", v), 80)
				if searchQuery != "" {
					vStr = HighlightSearch(vStr, searchQuery)
				}
				fmt.Printf("%s│%s   %s%s%s: %s\n", Blue, Reset, Cyan, k, Reset, vStr)
			}
		}
	}
	fmt.Println()
}

// PrintLLMCall renders an LLM call (DSPy format).
func PrintLLMCall(call LLMCallData, index int, compact bool, searchQuery string) {
	fmt.Printf("%s┌─ LLM Call #%d%s", BoldYellow, index, Reset)
	fmt.Printf(" %s%s/%s%s", Dim, call.Provider, call.Model, Reset)
	if call.LatencyMs > 0 {
		fmt.Printf(" %s(%dms)%s", Dim, call.LatencyMs, Reset)
	}
	fmt.Println()

	if call.PromptTokens > 0 || call.CompletionTokens > 0 {
		fmt.Printf("%s│%s %sTokens:%s %d prompt + %d completion", Yellow, Reset, Dim, Reset, call.PromptTokens, call.CompletionTokens)
		if call.Cost > 0 {
			fmt.Printf(" ($%.4f)", call.Cost)
		}
		fmt.Println()
	}

	if !compact {
		if call.Prompt != "" {
			fmt.Printf("%s│%s %sPrompt:%s\n", Yellow, Reset, Dim, Reset)
			prompt := call.Prompt
			if searchQuery != "" {
				prompt = HighlightSearch(prompt, searchQuery)
			}
			PrintIndented(prompt, "│   ", 300)
		}
		if call.Response != "" {
			fmt.Printf("%s│%s %sResponse:%s\n", Yellow, Reset, Dim, Reset)
			response := call.Response
			if searchQuery != "" {
				response = HighlightSearch(response, searchQuery)
			}
			PrintIndented(response, "│   ", 300)
		}
	}
	fmt.Println()
}

// PrintCodeExec renders a code execution (DSPy format).
func PrintCodeExec(exec CodeExecData, compact bool, searchQuery string) {
	fmt.Printf("%s┌─ Code Execution%s", BoldBlue, Reset)
	if exec.Iteration > 0 {
		fmt.Printf(" %s(iteration %d)%s", Dim, exec.Iteration, Reset)
	}
	if exec.DurationMs > 0 {
		fmt.Printf(" %s(%dms)%s", Dim, exec.DurationMs, Reset)
	}

	if exec.Stderr != "" {
		fmt.Printf(" %s[ERROR]%s", BoldRed, Reset)
	}
	fmt.Println()

	if !compact {
		fmt.Printf("%s│%s %sCode:%s\n", Blue, Reset, Dim, Reset)
		code := exec.Code
		if searchQuery != "" {
			code = HighlightSearch(code, searchQuery)
		}
		PrintCodeBlock(code, "│   ")

		if exec.Stdout != "" {
			fmt.Printf("%s│%s %sStdout:%s\n", Blue, Reset, Green, Reset)
			stdout := exec.Stdout
			if searchQuery != "" {
				stdout = HighlightSearch(stdout, searchQuery)
			}
			PrintIndented(stdout, "│   ", 300)
		}
		if exec.Stderr != "" {
			fmt.Printf("%s│%s %sStderr:%s\n", Blue, Reset, Red, Reset)
			PrintIndented(exec.Stderr, "│   ", 300)
		}
	}
	fmt.Println()
}

// PrintToolCall renders a tool call (DSPy format).
func PrintToolCall(tool ToolCallData, compact bool) {
	statusColor := Green
	statusIcon := "✓"
	if !tool.Success {
		statusColor = Red
		statusIcon = "✗"
	}

	fmt.Printf("%s┌─ Tool: %s%s %s%s%s", BoldMagenta, tool.ToolName, Reset, statusColor, statusIcon, Reset)
	if tool.DurationMs > 0 {
		fmt.Printf(" %s(%dms)%s", Dim, tool.DurationMs, Reset)
	}
	fmt.Println()

	if !compact {
		if tool.Input != nil {
			fmt.Printf("%s│%s %sInput:%s %s\n", Magenta, Reset, Dim, Reset, Truncate(fmt.Sprintf("%v", tool.Input), 100))
		}
		if tool.Output != nil {
			fmt.Printf("%s│%s %sOutput:%s %s\n", Magenta, Reset, Dim, Reset, Truncate(fmt.Sprintf("%v", tool.Output), 100))
		}
		if tool.Error != "" {
			fmt.Printf("%s│%s %sError:%s %s\n", Magenta, Reset, Red, Reset, tool.Error)
		}
	}
	fmt.Println()
}

// BoldMagenta is bold magenta color.
var BoldMagenta = "\033[1;35m"

// PrintError renders an error (DSPy format).
func PrintError(err ErrorData) {
	recoverable := "non-recoverable"
	if err.Recoverable {
		recoverable = "recoverable"
	}

	fmt.Printf("%s┌─ Error: %s%s %s(%s)%s\n", BoldRed, err.ErrorType, Reset, Dim, recoverable, Reset)
	fmt.Printf("%s│%s %s\n", Red, Reset, err.Message)
	fmt.Println()
}

// PrintFinalOnly displays only the final answer (RLM format).
func PrintFinalOnly(data *LogData, iterations []Iteration) {
	fmt.Printf("\n%s%s Final Answer %s\n", BoldGreen, "═══", Reset)

	if data.Metadata != nil && data.Metadata.Query != "" {
		fmt.Printf("%sQuery:%s %s\n\n", Dim, Reset, data.Metadata.Query)
	}

	for _, iter := range iterations {
		if iter.FinalAnswer != nil {
			switch v := iter.FinalAnswer.(type) {
			case string:
				fmt.Printf("%s\n", v)
			case []any:
				if len(v) == 2 {
					fmt.Printf("%s%v%s = %v\n", Cyan, v[0], Reset, v[1])
				}
			default:
				fmt.Printf("%v\n", v)
			}
			fmt.Printf("\n%s(Iteration %d, %.2fs)%s\n", Dim, iter.Iteration, iter.IterationTime, Reset)
			return
		}
	}

	fmt.Printf("%sNo final answer found%s\n", Dim, Reset)
}

// PrintDetailedStats displays comprehensive session statistics (RLM format).
func PrintDetailedStats(data *LogData) {
	fmt.Printf("\n%s%s Session Statistics %s\n\n", BoldCyan, "═══", Reset)
	fmt.Printf("%sFormat:%s %s\n", Dim, Reset, data.Format.String())

	if data.Metadata != nil {
		fmt.Printf("%sModel:%s %s\n", Dim, Reset, data.Metadata.RootModel)
		fmt.Printf("%sBackend:%s %s\n", Dim, Reset, data.Metadata.Backend)
		if data.Metadata.Query != "" {
			fmt.Printf("%sQuery:%s %s\n", Dim, Reset, Truncate(data.Metadata.Query, 80))
		}
		fmt.Println()
	}

	var totalTime float64
	var totalCodeBlocks, totalLLMCalls int
	var totalPromptTokens, totalCompletionTokens int
	var iterTimes []float64
	var llmCallTimes []float64

	for _, iter := range data.Iterations {
		totalTime += iter.IterationTime
		iterTimes = append(iterTimes, iter.IterationTime)
		totalCodeBlocks += len(iter.CodeBlocks)

		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				totalLLMCalls++
				totalPromptTokens += call.PromptTokens
				totalCompletionTokens += call.CompletionTokens
				llmCallTimes = append(llmCallTimes, call.ExecutionTime)
			}
		}
	}

	// Overview
	fmt.Printf("%s── Overview ──%s\n", Bold, Reset)
	fmt.Printf("  Iterations:     %d\n", len(data.Iterations))
	fmt.Printf("  Code Blocks:    %d\n", totalCodeBlocks)
	fmt.Printf("  Sub-LLM Calls:  %d\n", totalLLMCalls)
	fmt.Printf("  Total Time:     %.2fs\n", totalTime)
	fmt.Println()

	// Token usage
	if totalPromptTokens > 0 || totalCompletionTokens > 0 {
		fmt.Printf("%s── Token Usage ──%s\n", Bold, Reset)
		fmt.Printf("  Prompt Tokens:     %d\n", totalPromptTokens)
		fmt.Printf("  Completion Tokens: %d\n", totalCompletionTokens)
		fmt.Printf("  Total Tokens:      %d\n", totalPromptTokens+totalCompletionTokens)
		if totalLLMCalls > 0 {
			fmt.Printf("  Avg per Call:      %.0f\n", float64(totalPromptTokens+totalCompletionTokens)/float64(totalLLMCalls))
		}
		fmt.Println()
	}

	// Timing analysis
	fmt.Printf("%s── Timing Analysis ──%s\n", Bold, Reset)
	if len(iterTimes) > 0 {
		sort.Float64s(iterTimes)
		fmt.Printf("  Iteration Time:\n")
		fmt.Printf("    Min:    %.2fs\n", iterTimes[0])
		fmt.Printf("    Max:    %.2fs\n", iterTimes[len(iterTimes)-1])
		fmt.Printf("    Median: %.2fs\n", iterTimes[len(iterTimes)/2])
		fmt.Printf("    Avg:    %.2fs\n", totalTime/float64(len(iterTimes)))
	}

	if len(llmCallTimes) > 0 {
		sort.Float64s(llmCallTimes)
		var sum float64
		for _, t := range llmCallTimes {
			sum += t
		}
		fmt.Printf("  Sub-LLM Call Time:\n")
		fmt.Printf("    Min:    %.2fs\n", llmCallTimes[0])
		fmt.Printf("    Max:    %.2fs\n", llmCallTimes[len(llmCallTimes)-1])
		fmt.Printf("    Avg:    %.2fs\n", sum/float64(len(llmCallTimes)))
	}
	fmt.Println()

	// Per-iteration breakdown
	fmt.Printf("%s── Per-Iteration Breakdown ──%s\n", Bold, Reset)
	fmt.Printf("  %s%-5s %-8s %-6s %-6s %-12s%s\n", Dim, "Iter", "Time", "Code", "Calls", "Tokens", Reset)
	for _, iter := range data.Iterations {
		var iterPrompt, iterCompletion int
		var callCount int
		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				callCount++
				iterPrompt += call.PromptTokens
				iterCompletion += call.CompletionTokens
			}
		}

		marker := ""
		if iter.FinalAnswer != nil {
			marker = BoldGreen + " ✓" + Reset
		}

		fmt.Printf("  %-5d %-8.2fs %-6d %-6d %-12s%s\n",
			iter.Iteration,
			iter.IterationTime,
			len(iter.CodeBlocks),
			callCount,
			fmt.Sprintf("%d→%d", iterPrompt, iterCompletion),
			marker,
		)
	}
	fmt.Println()

	// Timeline visualization
	if len(data.Iterations) > 0 && len(data.Iterations) <= 20 {
		fmt.Printf("%s── Timeline ──%s\n", Bold, Reset)
		maxTime := iterTimes[len(iterTimes)-1]
		for _, iter := range data.Iterations {
			barLen := int((iter.IterationTime / maxTime) * 40)
			if barLen < 1 {
				barLen = 1
			}
			bar := strings.Repeat("█", barLen)
			color := Yellow
			if iter.FinalAnswer != nil {
				color = Green
			}
			fmt.Printf("  %2d %s%s%s %.2fs\n", iter.Iteration, color, bar, Reset, iter.IterationTime)
		}
		fmt.Println()
	}
}

// PrintDSPyDetailedStats displays comprehensive session statistics (DSPy format).
func PrintDSPyDetailedStats(data *LogData) {
	fmt.Printf("\n%s%s Session Statistics %s\n\n", BoldCyan, "═══", Reset)
	fmt.Printf("%sFormat:%s %s\n", Dim, Reset, data.Format.String())

	if data.Session != nil {
		fmt.Printf("%sTrace ID:%s %s\n", Dim, Reset, data.Session.TraceID)
		fmt.Printf("%sStarted:%s %s\n", Dim, Reset, data.Session.StartTime.Format("2006-01-02 15:04:05"))
	}
	fmt.Println()

	// Overview
	fmt.Printf("%s── Overview ──%s\n", Bold, Reset)
	fmt.Printf("  Total Events:   %d\n", len(data.Events))
	fmt.Printf("  Modules:        %d\n", len(data.Modules))
	fmt.Printf("  LLM Calls:      %d\n", len(data.LLMCalls))
	fmt.Printf("  Code Execs:     %d\n", len(data.CodeExecs))
	fmt.Printf("  Tool Calls:     %d\n", len(data.ToolCalls))
	fmt.Printf("  Errors:         %d\n", len(data.Errors))
	fmt.Println()

	// Token usage
	var totalPrompt, totalCompletion int
	var totalCost float64
	for _, call := range data.LLMCalls {
		totalPrompt += call.PromptTokens
		totalCompletion += call.CompletionTokens
		totalCost += call.Cost
	}

	if totalPrompt > 0 || totalCompletion > 0 {
		fmt.Printf("%s── Token Usage ──%s\n", Bold, Reset)
		fmt.Printf("  Prompt Tokens:     %d\n", totalPrompt)
		fmt.Printf("  Completion Tokens: %d\n", totalCompletion)
		fmt.Printf("  Total Tokens:      %d\n", totalPrompt+totalCompletion)
		if len(data.LLMCalls) > 0 {
			fmt.Printf("  Avg per Call:      %.0f\n", float64(totalPrompt+totalCompletion)/float64(len(data.LLMCalls)))
		}
		if totalCost > 0 {
			fmt.Printf("  Total Cost:        $%.4f\n", totalCost)
		}
		fmt.Println()
	}

	// Timing analysis
	if len(data.LLMCalls) > 0 {
		fmt.Printf("%s── LLM Call Timing ──%s\n", Bold, Reset)
		var latencies []int64
		for _, call := range data.LLMCalls {
			latencies = append(latencies, call.LatencyMs)
		}
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		var sum int64
		for _, l := range latencies {
			sum += l
		}
		fmt.Printf("  Min:    %dms\n", latencies[0])
		fmt.Printf("  Max:    %dms\n", latencies[len(latencies)-1])
		fmt.Printf("  Median: %dms\n", latencies[len(latencies)/2])
		fmt.Printf("  Avg:    %dms\n", sum/int64(len(latencies)))
		fmt.Println()
	}

	// Module breakdown
	if len(data.Modules) > 0 {
		fmt.Printf("%s── Module Breakdown ──%s\n", Bold, Reset)
		fmt.Printf("  %s%-20s %-10s %-8s %-8s%s\n", Dim, "Module", "Type", "Status", "Duration", Reset)
		for _, mod := range data.Modules {
			status := Green + "✓" + Reset
			if !mod.Success {
				status = Red + "✗" + Reset
			}
			fmt.Printf("  %-20s %-10s %s        %dms\n",
				Truncate(mod.ModuleName, 20),
				Truncate(mod.ModuleType, 10),
				status,
				mod.DurationMs,
			)
		}
		fmt.Println()
	}

	// Provider breakdown
	if len(data.LLMCalls) > 0 {
		fmt.Printf("%s── Provider Breakdown ──%s\n", Bold, Reset)
		providers := make(map[string]struct {
			calls, tokens int
			latency       int64
		})
		for _, call := range data.LLMCalls {
			key := call.Provider + "/" + call.Model
			p := providers[key]
			p.calls++
			p.tokens += call.PromptTokens + call.CompletionTokens
			p.latency += call.LatencyMs
			providers[key] = p
		}
		fmt.Printf("  %s%-30s %-6s %-10s %-10s%s\n", Dim, "Provider/Model", "Calls", "Tokens", "Avg Latency", Reset)
		for key, p := range providers {
			fmt.Printf("  %-30s %-6d %-10d %dms\n",
				Truncate(key, 30),
				p.calls,
				p.tokens,
				p.latency/int64(p.calls),
			)
		}
		fmt.Println()
	}
}

// PrintSummary displays a quick summary of the session (RLM format).
func PrintSummary(iterations []Iteration, promptTokens, completionTokens int) {
	var totalTime float64
	var totalCodeBlocks, totalLLMCalls int

	for _, iter := range iterations {
		totalTime += iter.IterationTime
		totalCodeBlocks += len(iter.CodeBlocks)
		for _, block := range iter.CodeBlocks {
			totalLLMCalls += len(block.Result.RLMCalls)
		}
	}

	fmt.Printf("%s%s Summary %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("  Iterations: %d\n", len(iterations))
	fmt.Printf("  Code Blocks: %d\n", totalCodeBlocks)
	fmt.Printf("  Sub-LLM Calls: %d\n", totalLLMCalls)
	if promptTokens > 0 || completionTokens > 0 {
		fmt.Printf("  Tokens: %d prompt + %d completion = %d total\n",
			promptTokens, completionTokens, promptTokens+completionTokens)
	}
	fmt.Printf("  Total Time: %.2fs\n", totalTime)
	fmt.Println()
}

// PrintDSPySummary displays a quick summary (DSPy format).
func PrintDSPySummary(data *LogData) {
	prompt, completion := data.GetTotalTokens()

	fmt.Printf("%s%s Summary %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("  Events: %d\n", len(data.Events))
	fmt.Printf("  LLM Calls: %d\n", len(data.LLMCalls))
	fmt.Printf("  Modules: %d\n", len(data.Modules))
	if prompt > 0 || completion > 0 {
		fmt.Printf("  Tokens: %d prompt + %d completion = %d total\n",
			prompt, completion, prompt+completion)
	}
	if len(data.Errors) > 0 {
		fmt.Printf("  %sErrors: %d%s\n", Red, len(data.Errors), Reset)
	}
	fmt.Println()
}

// HighlightSearch highlights search terms in text.
func HighlightSearch(text, query string) string {
	if query == "" {
		return text
	}
	re := regexp.MustCompile("(?i)" + regexp.QuoteMeta(query))
	return re.ReplaceAllStringFunc(text, func(match string) string {
		return BgGreen + Bold + match + Reset
	})
}

// PrintIndented prints text with a prefix and truncation.
func PrintIndented(text, prefix string, maxLen int) {
	text = TruncateContent(text, maxLen)
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+line)
	}
}

// PrintCodeBlock prints a code block with truncation.
func PrintCodeBlock(code, prefix string) {
	lines := strings.Split(strings.TrimSpace(code), "\n")
	maxLines := 15
	if len(lines) > maxLines {
		for i := 0; i < maxLines-1; i++ {
			fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+lines[i])
		}
		fmt.Printf("%s%s%s... (%d more lines)%s\n", Yellow, prefix, Dim, len(lines)-maxLines+1, Reset)
	} else {
		for _, line := range lines {
			fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+line)
		}
	}
}

// Truncate truncates a string to maxLen characters.
func Truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

// TruncateContent truncates content preserving newlines.
func TruncateContent(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "\n... (truncated)"
	}
	return s
}

// ClearScreen clears the terminal screen.
func ClearScreen() {
	fmt.Print("\033[2J\033[H")
}
