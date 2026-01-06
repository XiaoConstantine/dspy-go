package viewer

import (
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// ExportLog exports the log data to a markdown file.
func ExportLog(data *LogData, cfg Config) error {
	if !strings.HasSuffix(cfg.Export, ".md") {
		return fmt.Errorf("only markdown (.md) export is supported")
	}

	switch data.Format {
	case FormatRLM:
		return exportRLMLog(data, cfg)
	case FormatDSPy:
		return exportDSPyLog(data, cfg)
	default:
		return fmt.Errorf("unknown log format")
	}
}

// exportRLMLog exports RLM format logs to markdown.
func exportRLMLog(data *LogData, cfg Config) error {
	var sb strings.Builder

	// Header
	sb.WriteString("# DSPy Session Log (RLM Format)\n\n")

	if data.Metadata != nil {
		sb.WriteString("## Metadata\n\n")
		sb.WriteString(fmt.Sprintf("- **Model:** %s\n", data.Metadata.RootModel))
		sb.WriteString(fmt.Sprintf("- **Backend:** %s\n", data.Metadata.Backend))
		sb.WriteString(fmt.Sprintf("- **Max Iterations:** %d\n", data.Metadata.MaxIterations))
		if data.Metadata.Query != "" {
			sb.WriteString(fmt.Sprintf("- **Query:** %s\n", data.Metadata.Query))
		}
		if data.Metadata.Context != "" {
			sb.WriteString(fmt.Sprintf("- **Context:** %s\n", Truncate(data.Metadata.Context, 200)))
		}
		sb.WriteString("\n")
	}

	// Iterations
	for _, iter := range data.Iterations {
		finalMarker := ""
		if iter.FinalAnswer != nil {
			finalMarker = " ✅"
		}
		sb.WriteString(fmt.Sprintf("## Iteration %d%s\n\n", iter.Iteration, finalMarker))
		sb.WriteString(fmt.Sprintf("*Time: %.2fs*\n\n", iter.IterationTime))

		if iter.Response != "" {
			sb.WriteString("### Response\n\n")
			sb.WriteString(iter.Response)
			sb.WriteString("\n\n")
		}

		for i, block := range iter.CodeBlocks {
			sb.WriteString(fmt.Sprintf("### Code Block %d\n\n", i+1))
			sb.WriteString("```go\n")
			sb.WriteString(block.Code)
			sb.WriteString("\n```\n\n")

			if block.Result.Stdout != "" {
				sb.WriteString("**Output:**\n```\n")
				sb.WriteString(block.Result.Stdout)
				sb.WriteString("\n```\n\n")
			}

			if block.Result.Stderr != "" {
				sb.WriteString("**Errors:**\n```\n")
				sb.WriteString(block.Result.Stderr)
				sb.WriteString("\n```\n\n")
			}

			if len(block.Result.RLMCalls) > 0 {
				sb.WriteString(fmt.Sprintf("**Sub-LLM Calls (%d):**\n\n", len(block.Result.RLMCalls)))
				for j, call := range block.Result.RLMCalls {
					sb.WriteString(fmt.Sprintf("%d. *%.2fs, %d→%d tokens*\n", j+1, call.ExecutionTime, call.PromptTokens, call.CompletionTokens))
					sb.WriteString(fmt.Sprintf("   - Prompt: %s\n", Truncate(call.Prompt, 100)))
					sb.WriteString(fmt.Sprintf("   - Response: %s\n", Truncate(call.Response, 100)))
				}
				sb.WriteString("\n")
			}
		}

		if iter.FinalAnswer != nil {
			sb.WriteString("### Final Answer\n\n")
			switch v := iter.FinalAnswer.(type) {
			case string:
				sb.WriteString(v)
			case []any:
				if len(v) == 2 {
					sb.WriteString(fmt.Sprintf("`%v` = %v", v[0], v[1]))
				}
			default:
				sb.WriteString(fmt.Sprintf("%v", v))
			}
			sb.WriteString("\n\n")
		}
	}

	// Summary
	var totalTime float64
	var totalCodeBlocks, totalLLMCalls, totalPrompt, totalCompletion int
	for _, iter := range data.Iterations {
		totalTime += iter.IterationTime
		totalCodeBlocks += len(iter.CodeBlocks)
		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				totalLLMCalls++
				totalPrompt += call.PromptTokens
				totalCompletion += call.CompletionTokens
			}
		}
	}

	sb.WriteString("## Summary\n\n")
	sb.WriteString(fmt.Sprintf("- **Iterations:** %d\n", len(data.Iterations)))
	sb.WriteString(fmt.Sprintf("- **Code Blocks:** %d\n", totalCodeBlocks))
	sb.WriteString(fmt.Sprintf("- **Sub-LLM Calls:** %d\n", totalLLMCalls))
	sb.WriteString(fmt.Sprintf("- **Total Tokens:** %d (%d prompt + %d completion)\n", totalPrompt+totalCompletion, totalPrompt, totalCompletion))
	sb.WriteString(fmt.Sprintf("- **Total Time:** %.2fs\n", totalTime))

	return os.WriteFile(cfg.Export, []byte(sb.String()), 0644)
}

// exportDSPyLog exports native DSPy format logs to markdown.
func exportDSPyLog(data *LogData, cfg Config) error {
	var sb strings.Builder

	// Header
	sb.WriteString("# DSPy Session Log\n\n")

	if data.Session != nil {
		sb.WriteString("## Session\n\n")
		sb.WriteString(fmt.Sprintf("- **Trace ID:** %s\n", data.Session.TraceID))
		sb.WriteString(fmt.Sprintf("- **Start Time:** %s\n", data.Session.StartTime.Format(time.RFC3339)))
		sb.WriteString("\n")
	}

	// LLM Calls
	if len(data.LLMCalls) > 0 {
		sb.WriteString("## LLM Calls\n\n")
		for i, call := range data.LLMCalls {
			sb.WriteString(fmt.Sprintf("### Call %d\n\n", i+1))
			sb.WriteString(fmt.Sprintf("- **Provider:** %s\n", call.Provider))
			sb.WriteString(fmt.Sprintf("- **Model:** %s\n", call.Model))
			sb.WriteString(fmt.Sprintf("- **Latency:** %dms\n", call.LatencyMs))
			sb.WriteString(fmt.Sprintf("- **Tokens:** %d (%d prompt + %d completion)\n",
				call.TotalTokens, call.PromptTokens, call.CompletionTokens))
			if call.Cost > 0 {
				sb.WriteString(fmt.Sprintf("- **Cost:** $%.6f\n", call.Cost))
			}
			sb.WriteString("\n**Prompt:**\n```\n")
			sb.WriteString(Truncate(call.Prompt, 500))
			sb.WriteString("\n```\n\n**Response:**\n```\n")
			sb.WriteString(Truncate(call.Response, 500))
			sb.WriteString("\n```\n\n")
		}
	}

	// Modules
	if len(data.Modules) > 0 {
		sb.WriteString("## Modules\n\n")
		for i, mod := range data.Modules {
			status := "✅"
			if !mod.Success {
				status = "❌"
			}
			sb.WriteString(fmt.Sprintf("### %s %s\n\n", mod.ModuleName, status))
			sb.WriteString(fmt.Sprintf("- **Type:** %s\n", mod.ModuleType))
			if mod.Signature != "" {
				sb.WriteString(fmt.Sprintf("- **Signature:** %s\n", mod.Signature))
			}
			sb.WriteString(fmt.Sprintf("- **Duration:** %dms\n", mod.DurationMs))
			sb.WriteString(fmt.Sprintf("- **LLM Calls:** %d\n", mod.LLMCalls))
			sb.WriteString(fmt.Sprintf("- **Tokens:** %d\n", mod.TotalTokens))
			if i < len(data.Modules)-1 {
				sb.WriteString("\n")
			}
		}
		sb.WriteString("\n")
	}

	// Code Executions
	if len(data.CodeExecs) > 0 {
		sb.WriteString("## Code Executions\n\n")
		for i, exec := range data.CodeExecs {
			sb.WriteString(fmt.Sprintf("### Execution %d (Iteration %d)\n\n", i+1, exec.Iteration))
			sb.WriteString("```go\n")
			sb.WriteString(exec.Code)
			sb.WriteString("\n```\n\n")

			if exec.Stdout != "" {
				sb.WriteString("**Output:**\n```\n")
				sb.WriteString(exec.Stdout)
				sb.WriteString("\n```\n\n")
			}

			if exec.Stderr != "" {
				sb.WriteString("**Errors:**\n```\n")
				sb.WriteString(exec.Stderr)
				sb.WriteString("\n```\n\n")
			}
		}
	}

	// Tool Calls
	if len(data.ToolCalls) > 0 {
		sb.WriteString("## Tool Calls\n\n")
		for i, tool := range data.ToolCalls {
			status := "✅"
			if !tool.Success {
				status = "❌"
			}
			sb.WriteString(fmt.Sprintf("### %s %s\n\n", tool.ToolName, status))
			sb.WriteString(fmt.Sprintf("- **Duration:** %dms\n", tool.DurationMs))
			if tool.Error != "" {
				sb.WriteString(fmt.Sprintf("- **Error:** %s\n", tool.Error))
			}
			if i < len(data.ToolCalls)-1 {
				sb.WriteString("\n")
			}
		}
		sb.WriteString("\n")
	}

	// Errors
	if len(data.Errors) > 0 {
		sb.WriteString("## Errors\n\n")
		for _, errData := range data.Errors {
			recoverable := "No"
			if errData.Recoverable {
				recoverable = "Yes"
			}
			sb.WriteString(fmt.Sprintf("- **%s:** %s (Recoverable: %s)\n",
				errData.ErrorType, errData.Message, recoverable))
		}
		sb.WriteString("\n")
	}

	// Summary
	prompt, completion := data.GetTotalTokens()
	sb.WriteString("## Summary\n\n")
	sb.WriteString(fmt.Sprintf("- **LLM Calls:** %d\n", len(data.LLMCalls)))
	sb.WriteString(fmt.Sprintf("- **Modules:** %d\n", len(data.Modules)))
	sb.WriteString(fmt.Sprintf("- **Code Executions:** %d\n", len(data.CodeExecs)))
	sb.WriteString(fmt.Sprintf("- **Tool Calls:** %d\n", len(data.ToolCalls)))
	sb.WriteString(fmt.Sprintf("- **Errors:** %d\n", len(data.Errors)))
	sb.WriteString(fmt.Sprintf("- **Total Tokens:** %d (%d prompt + %d completion)\n", prompt+completion, prompt, completion))

	return os.WriteFile(cfg.Export, []byte(sb.String()), 0644)
}

// WatchLog watches a log file for changes and displays updates.
func WatchLog(filename string, cfg Config) error {
	fmt.Printf("%s%s Watching: %s %s\n", BoldCyan, "═══", filename, Reset)
	fmt.Printf("%sPress Ctrl+C to stop%s\n\n", Dim, Reset)

	lastSize := int64(0)
	lastIterCount := 0
	lastEventCount := 0
	headerPrinted := false

	// Handle Ctrl+C gracefully
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-sigCh:
			fmt.Printf("\n%sStopped watching%s\n", Dim, Reset)
			return nil
		case <-ticker.C:
			info, err := os.Stat(filename)
			if err != nil {
				continue
			}

			if info.Size() != lastSize {
				lastSize = info.Size()

				data, err := ParseLog(filename)
				if err != nil {
					continue
				}

				switch data.Format {
				case FormatRLM:
					// Print new iterations
					for i := lastIterCount; i < len(data.Iterations); i++ {
						if !headerPrinted && data.Metadata != nil {
							PrintHeader(filename, data.Metadata, FormatRLM)
							headerPrinted = true
						}
						PrintIteration(data.Iterations[i], cfg.Compact, "")
					}
					lastIterCount = len(data.Iterations)

				case FormatDSPy:
					// Print header on first event
					if !headerPrinted && data.Session != nil {
						PrintDSPyHeader(filename, data.Session, FormatDSPy)
						headerPrinted = true
					}
					// Print new events
					for i := lastEventCount; i < len(data.Events); i++ {
						printWatchEvent(data.Events[i], i)
					}
					lastEventCount = len(data.Events)
				}
			}
		}
	}
}

// printWatchEvent prints a single event for watch mode.
func printWatchEvent(event TraceEvent, index int) {
	switch event.Type {
	case TraceEventLLMCall:
		if call := parseLLMCallEvent(event); call != nil {
			PrintLLMCall(*call, index, false, "")
		}
	case TraceEventModule:
		if mod := parseModuleEvent(event); mod != nil {
			PrintModule(*mod, false, "")
		}
	case TraceEventCodeExec:
		if exec := parseCodeExecEvent(event); exec != nil {
			PrintCodeExec(*exec, false, "")
		}
	case TraceEventToolCall:
		if tool := parseToolCallEvent(event); tool != nil {
			PrintToolCall(*tool, false)
		}
	case TraceEventError:
		if errData := parseErrorEvent(event); errData != nil {
			PrintError(*errData)
		}
	}
}
