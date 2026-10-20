// Look up local code through the MCP registry without configuring an LLM.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/XiaoConstantine/mcp-go/pkg/transport"
)

// mcp-go v0.3.1 cannot decode nullable schema types from tools/list. Register
// only the non-null parameters this example uses, rather than rewriting the
// server's schema or exposing its full tool set.
func lookupTool(c *client.Client) *tools.MCPTool {
	return tools.NewMCPTool("sem_find", "Find code context in the selected local checkout",
		models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{
			"query":        {Type: "string", Required: true},
			"mode":         {Type: "string"},
			"token_budget": {Type: "integer"},
		}}, c, "sem_find")
}

func main() {
	repo := flag.String("repo", "", "Local checkout to search (required)")
	query := flag.String("query", "", "Symbol name or code query (required)")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if err := run(ctx, *repo, *query, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, repo, query string, output io.Writer) error {
	if strings.TrimSpace(repo) == "" || strings.TrimSpace(query) == "" {
		return fmt.Errorf("-repo and -query are required")
	}
	checkout, err := filepath.Abs(repo)
	if err != nil {
		return err
	}
	info, err := os.Stat(checkout)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("repo must be a directory")
	}
	cmd := exec.CommandContext(ctx, "sem", "mcp")
	cmd.Dir = checkout
	cmd.Env = append(os.Environ(), "SEM_REPO="+checkout)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	defer stdin.Close()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	defer stdout.Close()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start sem (install it on PATH first): %w", err)
	}
	mcpClient := client.NewClient(transport.NewStdioTransport(stdout, stdin, nil),
		client.WithClientInfo("dspy-go-sem-example", "1.0.0"))
	defer func() {
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		_ = stdout.Close()
		_ = mcpClient.Shutdown()
	}()
	if _, err := mcpClient.Initialize(ctx); err != nil {
		return err
	}
	registry := tools.NewInMemoryToolRegistry()
	if err := registry.Register(lookupTool(mcpClient)); err != nil {
		return err
	}
	tool, err := registry.Get("sem_find")
	if err != nil {
		return err
	}
	result, err := tool.Execute(ctx, map[string]any{"query": query, "mode": "context", "token_budget": 1200})
	if err != nil {
		return err
	}
	if result.Metadata["isError"] == true {
		return fmt.Errorf("sem_find failed: %v", result.Data)
	}
	_, err = fmt.Fprintln(output, result.Data)
	return err
}
