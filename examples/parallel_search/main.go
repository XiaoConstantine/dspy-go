// Parallel Search MCP example: discover remote tools and execute them through
// the dspy-go registry without configuring an LLM.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/XiaoConstantine/mcp-go/pkg/transport"
)

const endpoint = "https://search.parallel.ai/mcp"
const userAgent = "dspy-go/parallel-search-example"

// RegisterMCPTools supplies its own discovery timeout. Also honor the example's
// overall context, including cancellation while discovery is in flight.
type boundedClient struct {
	*client.Client
	ctx context.Context
}

func (c boundedClient) ListTools(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error) {
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(c.ctx, cancel)
	defer stop()
	defer cancel()
	return c.Client.ListTools(ctx, cursor)
}

func main() {
	query := flag.String("query", "Go context cancellation documentation", "Web search query")
	url := flag.String("url", "", "Fetch this URL instead of searching")
	timeout := flag.Duration("timeout", time.Minute, "Overall execution timeout")
	flag.Parse()
	if *timeout <= 0 {
		fmt.Fprintln(os.Stderr, "timeout must be positive")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	if err := run(ctx, endpoint, *query, *url, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, serverURL, query, url string, output io.Writer) error {
	// Isolate any saved bridge credentials so this example always uses the
	// anonymous endpoint, even on machines with authenticated MCP connections.
	configDir, err := os.MkdirTemp("", "dspy-go-parallel-search-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(configDir)
	// Use the documented HTTP-to-stdio bridge with the existing MCP client.
	// No Authorization header or credential configuration is passed.
	cmd := exec.CommandContext(ctx, "mcp-remote", serverURL,
		"--transport", "http-only", "--enable-proxy", "--header", "User-Agent: "+userAgent)
	cmd.Env = append(os.Environ(), "MCP_REMOTE_CONFIG_DIR="+configDir)
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
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start mcp-remote (see README for installation): %w", err)
	}

	mcpClient := client.NewClient(transport.NewStdioTransport(stdout, stdin, nil),
		client.WithClientInfo("dspy-go-parallel-search", "1.0.0"))
	defer func() {
		// Close pipes and reap the bridge before shutting down the client, so its
		// stdio reader cannot remain blocked waiting for the next message.
		_ = stdin.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		_ = stdout.Close()
		_ = mcpClient.Shutdown()
	}()
	if _, err := mcpClient.Initialize(ctx); err != nil {
		return fmt.Errorf("initialize MCP: %w", err)
	}
	registry := tools.NewInMemoryToolRegistry()
	if err := tools.RegisterMCPTools(registry, boundedClient{Client: mcpClient, ctx: ctx}); err != nil {
		return err
	}

	// Reuse this identifier for related calls in a conversation.
	sessionBytes := make([]byte, 16)
	if _, err := rand.Read(sessionBytes); err != nil {
		return err
	}
	args := map[string]any{"session_id": hex.EncodeToString(sessionBytes)}
	toolName := "web_search"
	if url != "" {
		toolName = "web_fetch"
		args["urls"] = []string{url}
	} else {
		args["objective"] = query
		args["search_queries"] = []string{query}
	}
	tool, err := registry.Get(toolName)
	if err != nil {
		return err
	}
	result, err := tool.Execute(ctx, args)
	if err != nil {
		return err
	}
	if result.Annotations["mcp_error"] == true {
		return fmt.Errorf("%s failed: %v", toolName, result.Data)
	}
	text, ok := result.Data.(string)
	if !ok || strings.TrimSpace(text) == "" {
		return fmt.Errorf("%s returned no text", toolName)
	}
	_, err = fmt.Fprintln(output, text)
	return err
}
