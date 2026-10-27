// deed.page MCP example: publish an HTML page through the dspy-go tool registry,
// check that it is live, then delete it. No LLM or API key is needed.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/XiaoConstantine/mcp-go/pkg/transport"
)

const endpoint = "https://deed.page/mcp"
const userAgent = "dspy-go/deed-page-example"

const samplePage = `<!doctype html><meta charset="utf-8"><title>dspy-go</title>
<h1>Published from dspy-go</h1>
<p>This page was published with the deed.page MCP server through the dspy-go tool registry.</p>`

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

type options struct {
	html string       // Page to publish.
	ttl  int          // Seconds until the page expires; 0 keeps it.
	keep bool         // Leave the page up instead of deleting it.
	http *http.Client // Used to check the published URL.
}

func main() {
	file := flag.String("file", "", "HTML file to publish (default: a small sample page)")
	ttl := flag.Int("ttl", 600, "Seconds until the page expires (0 keeps it until deleted)")
	keep := flag.Bool("keep", false, "Leave the page up instead of deleting it")
	timeout := flag.Duration("timeout", time.Minute, "Overall execution timeout")
	flag.Parse()
	if *timeout <= 0 {
		fmt.Fprintln(os.Stderr, "timeout must be positive")
		os.Exit(1)
	}
	html := samplePage
	if *file != "" {
		b, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		html = string(b)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	opts := options{html: html, ttl: *ttl, keep: *keep, http: &http.Client{Timeout: 30 * time.Second}}
	if err := run(ctx, endpoint, opts, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, serverURL string, opts options, output io.Writer) error {
	// Isolate any saved bridge credentials so the example never picks up an
	// unrelated MCP login.
	configDir, err := os.MkdirTemp("", "dspy-go-deed-page-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(configDir)
	// Use the documented HTTP-to-stdio bridge with the existing MCP client.
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
		client.WithClientInfo("dspy-go-deed-page", "1.0.0"))
	defer func() {
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

	args := map[string]any{"files": map[string]any{"index.html": opts.html}}
	if opts.ttl > 0 {
		args["ttl_seconds"] = opts.ttl
	}
	var site struct {
		OK    bool   `json:"ok"`
		URL   string `json:"url"`
		Slug  string `json:"slug"`
		Token string `json:"token"`
	}
	if err := call(ctx, registry, "publish_site", args, &site); err != nil {
		return err
	}
	if !site.OK || site.URL == "" || site.Slug == "" {
		return errors.New("publish_site returned no URL")
	}
	fmt.Fprintln(output, "published:", site.URL)

	status, err := check(ctx, opts.http, site.URL)
	if err != nil {
		return err
	}
	fmt.Fprintln(output, "GET", site.URL, "->", status)
	if status != http.StatusOK {
		return fmt.Errorf("published page returned HTTP %d", status)
	}
	if opts.keep {
		// The token is the only way to update or delete this page later.
		// It is printed once here on purpose; store it somewhere private.
		fmt.Fprintln(output, "kept; owner token:", site.Token)
		return nil
	}

	var deleted struct {
		OK bool `json:"ok"`
	}
	if err := call(ctx, registry, "delete_site", map[string]any{"slug": site.Slug, "token": site.Token}, &deleted); err != nil {
		return err
	}
	if !deleted.OK {
		return errors.New("delete_site did not confirm the deletion")
	}
	status, err = check(ctx, opts.http, site.URL)
	if err != nil {
		return err
	}
	fmt.Fprintln(output, "deleted; GET", site.URL, "->", status)
	return nil
}

// call selects a discovered tool from the registry, executes it and decodes
// the JSON text it returns.
func call(ctx context.Context, registry *tools.InMemoryToolRegistry, name string, args map[string]any, out any) error {
	tool, err := registry.Get(name)
	if err != nil {
		return err
	}
	result, err := tool.Execute(ctx, args)
	if err != nil {
		return err
	}
	return decode(name, result, out)
}

func decode(name string, result core.ToolResult, out any) error {
	text, ok := result.Data.(string)
	if result.Annotations["mcp_error"] == true {
		return fmt.Errorf("%s failed: %v", name, result.Data)
	}
	if !ok || strings.TrimSpace(text) == "" {
		return fmt.Errorf("%s returned no text", name)
	}
	if err := json.Unmarshal([]byte(text), out); err != nil {
		return fmt.Errorf("%s: unexpected result: %w", name, err)
	}
	return nil
}

func check(ctx context.Context, client *http.Client, url string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode, nil
}
