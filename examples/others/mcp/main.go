package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	dspyLogging "github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	mcpLogging "github.com/XiaoConstantine/mcp-go/pkg/logging"
)

// LoggerAdapter adapts dspy-go logger to mcp-go logger interface.
type LoggerAdapter struct {
	dspyLogger *dspyLogging.Logger
	ctx        context.Context
}

func NewLoggerAdapter(dspyLogger *dspyLogging.Logger) mcpLogging.Logger {
	return &LoggerAdapter{
		dspyLogger: dspyLogger,
		ctx:        context.Background(),
	}
}

// Debug implements mcp-go/pkg/logging.Logger interface.
func (a *LoggerAdapter) Debug(msg string, args ...interface{}) {
	a.dspyLogger.Debug(a.ctx, msg, args...)
}

// Info implements mcp-go/pkg/logging.Logger interface.
func (a *LoggerAdapter) Info(msg string, args ...interface{}) {
	a.dspyLogger.Info(a.ctx, msg, args...)
}

// Warn implements mcp-go/pkg/logging.Logger interface.
func (a *LoggerAdapter) Warn(msg string, args ...interface{}) {
	a.dspyLogger.Warn(a.ctx, msg, args...)
}

// Error implements mcp-go/pkg/logging.Logger interface.
func (a *LoggerAdapter) Error(msg string, args ...interface{}) {
	a.dspyLogger.Error(a.ctx, msg, args...)
}

func main() {
	// Setup logging
	ctx := core.WithExecutionState(context.Background())
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	loggerAdapter := NewLoggerAdapter(logger)
	// 1. Start MCP server as a subprocess (e.g., Git MCP server)
	cmd := exec.Command("./git-mcp-server")

	// Set up stdio for communication
	serverIn, err := cmd.StdinPipe()
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to create stdin pipe: %v", err))
	}

	serverOut, err := cmd.StdoutPipe()
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to create stdout pipe: %v", err))
	}

	cmd.Stderr = os.Stderr

	// Start the server
	if err := cmd.Start(); err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to start server: %v", err))
	}

	// Give the server a moment to initialize
	time.Sleep(1 * time.Second)

	// 2. Create MCP client
	mcpClient, err := tools.NewMCPClientFromStdio(
		serverOut,
		serverIn,
		tools.MCPClientOptions{
			ClientName:    "react-example",
			ClientVersion: "0.1.0",
			Logger:        loggerAdapter,
		},
	)
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to create MCP client: %v", err))
	}

	// 3. Create tool registry and register MCP tools
	registry := tools.NewRegistry()
	err = tools.RegisterMCPTools(registry, mcpClient)
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to register MCP tools: %v", err))
	}

	// Get the list of tools
	toolList := registry.List()
	fmt.Println("Registered tools:")
	for _, tool := range toolList {
		fmt.Printf("- %s: %s\n", tool.Name(), tool.Description())

		// Verify that tool satisfies core.Tool interface
		if _, ok := tool.(core.Tool); !ok {
			logger.Error(ctx, "Tool %s does not implement core.Tool interface", tool.Name())
			return
		}
	}

	// Convert tools.Tool list to core.Tool list
	var coreTools []core.Tool
	for _, tool := range toolList {
		if coreTool, ok := tool.(core.Tool); ok {
			coreTools = append(coreTools, coreTool)
		}
	}

	// 4. Create and configure ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "query"}}},
		[]core.OutputField{
			{Field: core.NewField("answer")},
		},
	).WithInstruction(`Answer the query about Git repositories by reasoning step by step and using the available Git tools.
Identify which Git tool is appropriate for the question and use it with the correct arguments.
Always use the git_ prefixed tools like git_blame, git_log, git_status, etc.
When you have a final answer, use the action 'Finish'.DO NOT wrap response in json format`)
	maxIters := 5
	reactModule := modules.NewReAct(signature, coreTools, maxIters)
	llms.EnsureFactory()

	// 5. Set up LLM
	// This assumes you've set the API_KEY environment variable
	// Configure the default LLM
	err = core.ConfigureDefaultLLM("", core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to configure LLM: %v", err))
	}

	// Set the LLM for the ReAct module
	reactModule.SetLLM(core.GetDefaultLLM())

	// 6. Execute query with ReAct
	result, err := reactModule.Process(ctx, map[string]interface{}{
		"query": "Show me the details of the most recent commit",
	})

	if err != nil {
		logger.Error(ctx, "Error executing ReAct: %v", err)
		return
	}

	// 7. Print the result
	logger.Info(ctx, "\nReAct Result: %v", result)
	logger.Info(ctx, "Thought: %v\n", result["thought"])
	logger.Info(ctx, "Action: %v\n", result["action"])
	logger.Info(ctx, "Observation: %v\n", result["observation"])
	logger.Info(ctx, "Answer: %v\n", result["answer"])

	// 8. Clean up
	logger.Info(ctx, "\nShutting down...")
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		logger.Error(ctx, "Failed to send interrupt signal: %v", err)
		if err := cmd.Process.Kill(); err != nil {
			logger.Error(ctx, "Failed to kill process: %v", err)
		}
	}

	if err := cmd.Wait(); err != nil {
		logger.Error(ctx, "Server exited with error: %v", err)
	}
}
