package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
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
	output := dspyLogging.NewConsoleOutput(true, dspyLogging.WithColor(true))

	logger := dspyLogging.NewLogger(dspyLogging.Config{
		Severity: dspyLogging.DEBUG,
		Outputs:  []dspyLogging.Output{output},
	})
	dspyLogging.SetLogger(logger)

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
	// signature := core.NewSignature(
	// 	[]core.InputField{{Field: core.Field{Name: "query"}}},
	// 	[]core.OutputField{
	// 		{Field: core.NewField("answer")},
	// 	},
	// ).WithInstruction(`Answer the query about Git repositories by reasoning step by step and using the available Git tools.
	// 	Identify which Git tool is appropriate for the question and use it with the correct arguments.
	// 	Always use the git_ prefixed tools like git_blame, git_log, git_status, etc.
	//
	// 	CRITICAL rules for the 'action' field's VALUE within output, DO NOT wrap response in markdown or json:
	// 	  1.  **Tool Call:** If calling a tool, the value MUST be a *string* containing the specific XML block below. Do NOT add any other
	// 	  text, prefixes, newlines, or markdown formatting around this XML string value.
	// 	      XML Format: <action><tool_name>TOOL_NAME</tool_name><arguments><arg key="ARG_KEY">ARG_VALUE</arg></arguments></action>
	// 	      Example XML String Value: "<action><tool_name>git_log</tool_name><arguments><arg key=\"n\">1</arg></arguments></action>"
	//
	//                 2.  **Final Answer:** When you have the final answer, generate 'thought' summarizing it, and set 'action' to "Finish".
	//
	//          DO NOT omit the 'action' field in the final step.
	//
	// 	 `)
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "query"}}},
		[]core.OutputField{
			{Field: core.NewField("answer")},
		},
	).WithInstruction(`Answer the query about Git repositories by reasoning step by step and using the available Git tools.
    Identify which Git tool is appropriate for the question and use it with the correct arguments.
    Always use the git_ prefixed tools like git_blame, git_log, git_status, etc.

    CRITICAL FORMATTING RULES:
    1. Format your response with these EXACT field headers:
       thought: [your reasoning]
       action: [your action]
       observation: [result from previous action, if any]
       answer: [your final answer when complete]

    2. For the action field, ONLY use one of these two formats:
       - XML tool call: action: <action><tool_name>git_log</tool_name><arguments><arg key="n">5</arg></arguments></action>
       - Final answer: action: Finish

    3. ALWAYS include both 'thought' and 'action' fields in EVERY response.

    Example correct format:
    thought: I need to check recent commits.
    action: <action><tool_name>git_log</tool_name><arguments><arg key="n">5</arg></arguments></action>

    When giving your final answer:
    thought: Based on the git_log output, I can see the 5 most recent commits.
    action: Finish
    answer: Here are the details of the 5 most recent commits: [summary of commits]
    `)

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
		"query": "Show me the details of latest 5 commit",
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
