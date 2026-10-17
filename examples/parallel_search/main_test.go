package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// Exercise the actual HTTP-to-stdio bridge, MCP registration, selection and
// execution. These tests need the same bridge installation as the example.
func TestRun(t *testing.T) {
	if _, err := exec.LookPath("mcp-remote"); err != nil {
		t.Skip("install mcp-remote@0.14.3 to run the bridge integration tests")
	}
	t.Setenv("NO_PROXY", "127.0.0.1,localhost")
	t.Setenv("no_proxy", "127.0.0.1,localhost")
	for _, mode := range []string{"search", "fetch", "tool_error", "rpc_error", "cancel", "cancel_discovery", "cancel_initialize"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			var mu sync.Mutex
			seen := map[string]bool{}
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
					t.Errorf("decode request: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if r.Header.Get("User-Agent") != userAgent {
					t.Errorf("%s User-Agent = %q", req.Method, r.Header.Get("User-Agent"))
				}
				if r.Header.Get("Authorization") != "" {
					t.Errorf("%s must be anonymous", req.Method)
				}
				mu.Lock()
				seen[req.Method] = true
				mu.Unlock()
				if req.ID == nil {
					w.WriteHeader(http.StatusAccepted)
					return
				}
				if (mode == "cancel_initialize" && req.Method == "initialize") || (mode == "cancel_discovery" && req.Method == "tools/list") {
					cancel()
					<-r.Context().Done()
					return
				}
				var result any
				switch req.Method {
				case "initialize":
					result = map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": "fixture", "version": "1"}}
				case "tools/list":
					result = map[string]any{"tools": []any{
						map[string]any{"name": "web_search", "inputSchema": map[string]any{"type": "object", "properties": map[string]any{"objective": map[string]any{"type": "string"}, "search_queries": map[string]any{"type": "array"}}, "required": []string{"objective", "search_queries"}}},
						map[string]any{"name": "web_fetch", "inputSchema": map[string]any{"type": "object", "properties": map[string]any{"urls": map[string]any{"type": "array"}}, "required": []string{"urls"}}},
					}}
				case "tools/call":
					wantName := "web_search"
					if mode == "fetch" {
						wantName = "web_fetch"
					}
					if req.Params["name"] != wantName {
						t.Errorf("selected tool = %v, want %s", req.Params["name"], wantName)
					}
					args, ok := req.Params["arguments"].(map[string]any)
					if !ok {
						t.Error("missing tool arguments")
					}
					if session, ok := args["session_id"].(string); !ok || len(session) != 32 {
						t.Error("missing random conversation identifier")
					}
					if mode == "fetch" {
						if fmt.Sprint(args["urls"]) != "[https://go.dev/blog/context]" {
							t.Errorf("fetch arguments: %v", args)
						}
					} else if args["objective"] != "Go context" || fmt.Sprint(args["search_queries"]) != "[Go context]" {
						t.Errorf("search arguments: %v", args)
					}
					if mode == "cancel" {
						cancel()
						<-r.Context().Done()
						return
					}
					result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "Go context documentation: https://go.dev/blog/context"}}, "isError": mode == "tool_error"}
				default:
					t.Errorf("unexpected method: %s", req.Method)
				}
				response := map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result}
				if mode == "rpc_error" && req.Method == "tools/call" {
					delete(response, "result")
					response["error"] = map[string]any{"code": -32602, "message": "invalid arguments"}
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(response)
			}))
			defer server.Close()
			var output bytes.Buffer
			url := ""
			if mode == "fetch" {
				url = "https://go.dev/blog/context"
			}
			start := time.Now()
			err := run(ctx, server.URL, "Go context", url, &output)
			if mode == "search" || mode == "fetch" {
				if err != nil || !strings.Contains(output.String(), "https://go.dev/blog/context") {
					t.Fatalf("run: %v, output: %q", err, output.String())
				}
			} else if err == nil || output.Len() != 0 {
				t.Fatalf("failed tool must return an error without output: %v, %q", err, output.String())
			}
			if strings.HasPrefix(mode, "cancel") && time.Since(start) > 5*time.Second {
				t.Error("cancellation did not stop the bridge promptly")
			}
			mu.Lock()
			defer mu.Unlock()
			for _, method := range []string{"initialize", "tools/list", "tools/call"} {
				if (mode == "cancel_initialize" && method != "initialize") || (mode == "cancel_discovery" && method == "tools/call") {
					continue
				}
				if !seen[method] {
					t.Errorf("missing actual HTTP request for %s", method)
				}
			}
		})
	}
}
