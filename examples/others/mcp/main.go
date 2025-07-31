// MCP Configuration Example
//
// This example demonstrates how to use the DSPy-Go configuration system
// to set up MCP (Model Context Protocol) integration with comprehensive
// configuration management including:
//
// - LLM provider configuration from config file
// - Logging setup from configuration
// - MCP server configuration with environment variables
// - ReAct module configuration
// - Execution timeouts and resource limits
//
// The configuration is loaded from config.yaml and demonstrates how
// to replace hardcoded values with flexible configuration management.

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
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
	// Parse command line flags
	configPath := flag.String("config", "", "Path to configuration file")
	flag.Parse()

	// Load configuration using specified path or automatic discovery
	var manager *config.Manager
	var err error

	if *configPath != "" {
		manager, err = config.NewManager(config.WithConfigPath(*configPath))
	} else {
		manager, err = config.NewManager()
	}

	if err != nil {
		fmt.Printf("Failed to create config manager: %v\n", err)
		os.Exit(1)
	}

	err = manager.Load()
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		fmt.Printf("Working directory: %s\n", func() string {
			if wd, err := os.Getwd(); err == nil {
				return wd
			}
			return "unknown"
		}())
		os.Exit(1)
	}

	cfg := manager.Get()

	// Setup logging from config
	ctx := core.WithExecutionState(context.Background())

	// Create logger manually from config (since NewFromConfig doesn't exist yet)
	var outputs []dspyLogging.Output
	for _, outputCfg := range cfg.Logging.Outputs {
		if outputCfg.Type == "console" {
			output := dspyLogging.NewConsoleOutput(true, dspyLogging.WithColor(outputCfg.Colors))
			outputs = append(outputs, output)
		}
	}

	// Convert log level string to severity using utility function
	severity := dspyLogging.ParseSeverity(cfg.Logging.Level)

	logger := dspyLogging.NewLogger(dspyLogging.Config{
		Severity: severity,
		Outputs:  outputs,
	})
	dspyLogging.SetLogger(logger)

	logger.Info(ctx, "Starting MCP example with configuration")
	logger.Info(ctx, "LLM Provider: %s, Model: %s", cfg.LLM.Default.Provider, cfg.LLM.Default.ModelID)
	logger.Info(ctx, "MCP Servers configured: %d", len(cfg.Tools.MCP.Servers))

	loggerAdapter := NewLoggerAdapter(logger)
	// 1. Start MCP servers from configuration
	var mcpClients []interface{} // Use interface{} until we fix the type issue
	for _, serverCfg := range cfg.Tools.MCP.Servers {
		logger.Info(ctx, "Starting MCP server: %s", serverCfg.Name)

		cmd := exec.Command(serverCfg.Command, serverCfg.Args...)

		// Set environment variables from config
		if len(serverCfg.Env) > 0 {
			cmd.Env = os.Environ()
			for k, v := range serverCfg.Env {
				cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
			}
		}

		// Set working directory if specified
		if serverCfg.WorkingDir != "" {
			cmd.Dir = serverCfg.WorkingDir
		}

		// Set up stdio for communication
		serverIn, err := cmd.StdinPipe()
		if err != nil {
			logger.Fatal(ctx, fmt.Sprintf("Failed to create stdin pipe for %s: %v", serverCfg.Name, err))
		}

		serverOut, err := cmd.StdoutPipe()
		if err != nil {
			logger.Fatal(ctx, fmt.Sprintf("Failed to create stdout pipe for %s: %v", serverCfg.Name, err))
		}

		cmd.Stderr = os.Stderr

		// Start the server
		if err := cmd.Start(); err != nil {
			logger.Warn(ctx, "Failed to start server %s: %v (continuing with other servers)", serverCfg.Name, err)
			continue
		}

		// Give the server a moment to initialize
		time.Sleep(1 * time.Second)

		// 2. Create MCP client
		mcpClient, err := tools.NewMCPClientFromStdio(
			serverOut,
			serverIn,
			tools.MCPClientOptions{
				ClientName:    "mcp-config-example",
				ClientVersion: "0.1.0",
				Logger:        loggerAdapter,
			},
		)
		if err != nil {
			logger.Warn(ctx, "Failed to create MCP client for %s: %v", serverCfg.Name, err)
			continue
		}

		mcpClients = append(mcpClients, mcpClient)
		logger.Info(ctx, "Successfully connected to MCP server: %s", serverCfg.Name)
	}

	if len(mcpClients) == 0 {
		logger.Fatal(ctx, "No MCP clients available")
	}

	// Use the first available client (in a real app, you might want to use all)
	mcpClientInterface := mcpClients[0].(tools.MCPClientInterface)

	// 3. Create tool registry and register MCP tools
	registry := tools.NewInMemoryToolRegistry()
	err = tools.RegisterMCPTools(registry, mcpClientInterface)
	if err != nil {
		logger.Fatal(ctx, fmt.Sprintf("Failed to register MCP tools: %v", err))
	}

	// 4. Create and configure ReAct module from config
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "query"}}},
		[]core.OutputField{
			{Field: core.NewField("answer")},
		},
	).WithInstruction(`Answer the query about Git repositories by using the available Git tools.
    Identify which Git tool is appropriate for the question and use it with the correct arguments.
    Always use the git_ prefixed tools like git_blame, git_log, git_status, etc.
    `)

	// Use max_cycles from config
	maxIters := cfg.Modules.ReAct.MaxCycles
	reactModule := modules.NewReAct(signature, registry, maxIters)
	llms.EnsureFactory()

	// 5. Set up LLM from configuration
	logger.Info(ctx, "Configuring LLM from config: %s/%s", cfg.LLM.Default.Provider, cfg.LLM.Default.ModelID)
	llms.EnsureFactory()

	// For now, use the existing configuration method until we implement NewFromConfig
	var modelName core.ModelID
	switch cfg.LLM.Default.ModelID {
	case "gemini-2.0-flash":
		modelName = core.ModelGoogleGeminiFlash
	default:
		modelName = core.ModelGoogleGeminiFlash // fallback to gemini
	}

	err = core.ConfigureDefaultLLM("", modelName)
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
	// Note: In a real implementation, we'd keep track of all started processes
	// and clean them up properly. For now, this is just demonstrating config usage.
}
