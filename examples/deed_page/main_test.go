package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// Exercise the actual HTTP-to-stdio bridge, MCP registration, selection and
// execution against a local fixture. These tests need the same bridge
// installation as the example and make no live requests.
func TestRun(t *testing.T) {
	if _, err := exec.LookPath("mcp-remote"); err != nil {
		t.Skip("install mcp-remote@0.14.3 to run the bridge integration tests")
	}
	t.Setenv("NO_PROXY", "127.0.0.1,localhost")
	t.Setenv("no_proxy", "127.0.0.1,localhost")
	for _, mode := range []string{"publish_delete", "keep", "no_ttl", "tool_error", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()

			var mu sync.Mutex
			live := false
			var calls []string
			page := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				if !live {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				_, _ = w.Write([]byte("<h1>ok</h1>"))
			}))
			defer page.Close()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					ID     any            `json:"id"`
					Method string         `json:"method"`
					Params map[string]any `json:"params"`
				}
				if r.Method != http.MethodPost {
					w.WriteHeader(http.StatusMethodNotAllowed)
					return
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if r.Header.Get("User-Agent") != userAgent {
					t.Errorf("%s User-Agent = %q", req.Method, r.Header.Get("User-Agent"))
				}
				if req.ID == nil {
					w.WriteHeader(http.StatusAccepted)
					return
				}
				text := func(v any) map[string]any {
					b, _ := json.Marshal(v)
					return map[string]any{"content": []any{map[string]any{"type": "text", "text": string(b)}}}
				}
				var result any
				switch req.Method {
				case "initialize":
					result = map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": "fixture", "version": "1"}}
				case "tools/list":
					result = map[string]any{"tools": []any{
						map[string]any{"name": "publish_site", "inputSchema": map[string]any{"type": "object", "properties": map[string]any{"files": map[string]any{"type": "object"}, "ttl_seconds": map[string]any{"type": "integer"}}, "required": []string{"files"}}},
						map[string]any{"name": "delete_site", "inputSchema": map[string]any{"type": "object", "properties": map[string]any{"slug": map[string]any{"type": "string"}, "token": map[string]any{"type": "string"}}, "required": []string{"slug", "token"}}},
					}}
				case "tools/call":
					name, _ := req.Params["name"].(string)
					args, _ := req.Params["arguments"].(map[string]any)
					mu.Lock()
					calls = append(calls, name)
					mu.Unlock()
					if mode == "cancel" {
						cancel()
						<-r.Context().Done()
						return
					}
					switch name {
					case "publish_site":
						if mode == "tool_error" {
							result = map[string]any{"isError": true, "content": []any{map[string]any{"type": "text", "text": "too large"}}}
							break
						}
						files, _ := args["files"].(map[string]any)
						if html, _ := files["index.html"].(string); !strings.Contains(html, "dspy-go") {
							t.Errorf("index.html = %q", html)
						}
						ttl, hasTTL := args["ttl_seconds"]
						if mode == "no_ttl" && hasTTL {
							t.Errorf("ttl_seconds sent with -ttl 0: %v", ttl)
						}
						if mode != "no_ttl" && ttl != float64(600) {
							t.Errorf("ttl_seconds = %v, want 600", ttl)
						}
						mu.Lock()
						live = true
						mu.Unlock()
						result = text(map[string]any{"ok": true, "url": page.URL + "/", "slug": "quiet-fern-abcd", "token": "secret-token"})
					case "delete_site":
						if args["slug"] != "quiet-fern-abcd" || args["token"] != "secret-token" {
							t.Errorf("delete_site args = %v", args)
						}
						mu.Lock()
						live = false
						mu.Unlock()
						result = text(map[string]any{"ok": true})
					default:
						t.Errorf("unexpected tool %q", name)
					}
				default:
					result = map[string]any{}
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
			}))
			defer server.Close()

			ttl := 600
			if mode == "no_ttl" {
				ttl = 0
			}
			var out bytes.Buffer
			err := run(ctx, server.URL, options{html: samplePage, ttl: ttl, keep: mode == "keep", http: page.Client()}, &out)
			mu.Lock()
			defer mu.Unlock()
			switch mode {
			case "tool_error":
				if err == nil || !strings.Contains(err.Error(), "publish_site failed") {
					t.Fatalf("err = %v, want publish_site failure", err)
				}
			case "cancel":
				if err == nil {
					t.Fatal("expected cancellation error")
				}
			case "keep":
				if err != nil {
					t.Fatal(err)
				}
				if strings.Join(calls, ",") != "publish_site" || !live {
					t.Fatalf("calls = %v, live = %v", calls, live)
				}
			default:
				if err != nil {
					t.Fatal(err)
				}
				if strings.Join(calls, ",") != "publish_site,delete_site" || live {
					t.Fatalf("calls = %v, live = %v", calls, live)
				}
				if !strings.Contains(out.String(), "-> 200") || !strings.Contains(out.String(), "deleted; GET") || !strings.Contains(out.String(), "-> 404") {
					t.Fatalf("output = %q", out.String())
				}
				if strings.Contains(out.String(), "secret-token") {
					t.Fatal("token printed without -keep")
				}
			}
		})
	}
}
