package files

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools/internal/localfs"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

const (
	DefaultModelOutputLimit   = 1600
	DefaultDisplayOutputLimit = 6000
)

type Config struct {
	Root               string
	ModelOutputLimit   int
	DisplayOutputLimit int
}

type Toolset struct {
	resolver           *localfs.Resolver
	root               string
	modelOutputLimit   int
	displayOutputLimit int
}

func NewToolset(cfg Config) (*Toolset, error) {
	root := strings.TrimSpace(cfg.Root)
	resolver, err := localfs.NewResolver(root)
	if err != nil {
		return nil, err
	}
	if cfg.ModelOutputLimit <= 0 {
		cfg.ModelOutputLimit = DefaultModelOutputLimit
	}
	if cfg.DisplayOutputLimit <= 0 {
		cfg.DisplayOutputLimit = DefaultDisplayOutputLimit
	}

	return &Toolset{
		resolver:           resolver,
		root:               resolver.Root(),
		modelOutputLimit:   cfg.ModelOutputLimit,
		displayOutputLimit: cfg.DisplayOutputLimit,
	}, nil
}

func (t *Toolset) Root() string {
	if t == nil {
		return ""
	}
	return t.root
}

func (t *Toolset) Tools() []core.Tool {
	if t == nil {
		return nil
	}
	return []core.Tool{
		t.newTool(
			"ls",
			"List files and directories within the workspace.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative path to list. Defaults to the workspace root.",
					},
					"recursive": {
						Type:        "boolean",
						Description: "Whether to recursively list descendants.",
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := t.resolver.ResolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				entries, err := listEntries(targetPath, t.root, boolValue(params["recursive"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				text := strings.Join(entries, "\n")
				if text == "" {
					text = "(empty)"
				}
				return successToolResult(
					truncateRunes(text, t.modelOutputLimit),
					truncateRunes(text, t.displayOutputLimit),
					map[string]any{
						"path":        t.resolver.DisplayPath(targetPath),
						"entry_count": len(entries),
					},
				), nil
			},
		),
		t.newTool(
			"read",
			"Read a UTF-8 text file inside the workspace.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative file path to read.",
						Required:    true,
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := t.resolver.ResolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				data, err := os.ReadFile(targetPath)
				if err != nil {
					return errorToolResult(fmt.Sprintf("read file: %v", err)), nil
				}
				content := string(data)
				return successToolResult(
					truncateRunes(content, t.modelOutputLimit),
					truncateRunes(content, t.displayOutputLimit),
					map[string]any{
						"path":  t.resolver.DisplayPath(targetPath),
						"bytes": len(data),
					},
				), nil
			},
		),
		t.newTool(
			"write",
			"Write a UTF-8 text file inside the workspace, creating parent directories when needed.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative file path to write.",
						Required:    true,
					},
					"content": {
						Type:        "string",
						Description: "New file contents.",
						Required:    true,
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := t.resolver.ResolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				content := stringValue(params["content"])
				if content == "" {
					return errorToolResult("content is required"), nil
				}
				if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
					return errorToolResult(fmt.Sprintf("create parent directories: %v", err)), nil
				}
				if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
					return errorToolResult(fmt.Sprintf("write file: %v", err)), nil
				}
				message := fmt.Sprintf("wrote %d bytes to %s", len(content), t.resolver.DisplayPath(targetPath))
				return successToolResult(message, message, map[string]any{
					"path":  t.resolver.DisplayPath(targetPath),
					"bytes": len(content),
				}), nil
			},
		),
		t.newTool(
			"edit",
			"Edit an existing UTF-8 text file by replacing exact text.",
			models.InputSchema{
				Type: "object",
				Properties: map[string]models.ParameterSchema{
					"path": {
						Type:        "string",
						Description: "Relative file path to edit.",
						Required:    true,
					},
					"old_text": {
						Type:        "string",
						Description: "Exact existing text to replace.",
						Required:    true,
					},
					"new_text": {
						Type:        "string",
						Description: "Replacement text.",
						Required:    true,
					},
					"replace_all": {
						Type:        "boolean",
						Description: "Replace every occurrence instead of only the first match.",
					},
				},
			},
			func(_ context.Context, params map[string]any) (core.ToolResult, error) {
				targetPath, err := t.resolver.ResolveSecurePath(stringValue(params["path"]))
				if err != nil {
					return errorToolResult(err.Error()), nil
				}
				oldText := stringValue(params["old_text"])
				if oldText == "" {
					return errorToolResult("old_text is required"), nil
				}
				newText := stringValue(params["new_text"])
				data, err := os.ReadFile(targetPath)
				if err != nil {
					return errorToolResult(fmt.Sprintf("read file: %v", err)), nil
				}
				content := string(data)
				if !strings.Contains(content, oldText) {
					return errorToolResult("old_text was not found in the target file"), nil
				}

				replacements := 1
				replaceAll := boolValue(params["replace_all"])
				updated := strings.Replace(content, oldText, newText, 1)
				if replaceAll {
					replacements = strings.Count(content, oldText)
					updated = strings.ReplaceAll(content, oldText, newText)
				}

				if err := os.WriteFile(targetPath, []byte(updated), 0o644); err != nil {
					return errorToolResult(fmt.Sprintf("write file: %v", err)), nil
				}

				message := fmt.Sprintf("edited %s with %d replacement(s)", t.resolver.DisplayPath(targetPath), replacements)
				return successToolResult(message, message, map[string]any{
					"path":          t.resolver.DisplayPath(targetPath),
					"replacements":  replacements,
					"replace_all":   replaceAll,
					"old_text_size": len(oldText),
					"new_text_size": len(newText),
				}), nil
			},
		),
	}
}

type tool struct {
	name        string
	description string
	schema      models.InputSchema
	run         func(context.Context, map[string]any) (core.ToolResult, error)
}

func (t *Toolset) newTool(name, description string, schema models.InputSchema, run func(context.Context, map[string]any) (core.ToolResult, error)) *tool {
	return &tool{
		name:        name,
		description: description,
		schema:      schema,
		run:         run,
	}
}

func (t *tool) Name() string {
	return t.name
}

func (t *tool) Description() string {
	return t.description
}

func (t *tool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         t.name,
		Description:  t.description,
		InputSchema:  t.schema,
		Capabilities: []string{"workspace", "files"},
		Version:      "1.0.0",
	}
}

func (t *tool) CanHandle(_ context.Context, intent string) bool {
	intent = strings.ToLower(intent)
	return strings.Contains(intent, "file") ||
		strings.Contains(intent, "workspace") ||
		strings.Contains(intent, "edit") ||
		strings.Contains(intent, "read") ||
		strings.Contains(intent, "write") ||
		strings.Contains(intent, "list")
}

func (t *tool) Execute(ctx context.Context, params map[string]any) (core.ToolResult, error) {
	return t.run(ctx, params)
}

func (t *tool) Validate(params map[string]any) error {
	for name, property := range t.schema.Properties {
		if property.Required && strings.TrimSpace(stringValue(params[name])) == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	return nil
}

func (t *tool) InputSchema() models.InputSchema {
	return t.schema
}

func (t *tool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	return &cloned
}

func successToolResult(modelText, displayText string, details map[string]any) core.ToolResult {
	if strings.TrimSpace(displayText) == "" {
		displayText = modelText
	}
	return core.ToolResult{
		Data: displayText,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   modelText,
			core.ToolResultDisplayTextMeta: displayText,
			core.ToolResultIsErrorMeta:     false,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: details,
		},
	}
}

func errorToolResult(message string) core.ToolResult {
	return core.ToolResult{
		Data: message,
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   message,
			core.ToolResultDisplayTextMeta: message,
			core.ToolResultIsErrorMeta:     true,
		},
		Annotations: map[string]any{
			core.ToolResultDetailsAnnotation: map[string]any{"error": message},
		},
	}
}

func listEntries(targetPath, workspaceRoot string, recursive bool) ([]string, error) {
	if recursive {
		entries := make([]string, 0, 16)
		err := filepath.WalkDir(targetPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, relErr := filepath.Rel(workspaceRoot, path)
			if relErr != nil {
				return relErr
			}
			if rel == "." {
				entries = append(entries, ".")
				return nil
			}
			label := filepath.ToSlash(rel)
			if d.IsDir() {
				label += "/"
			}
			entries = append(entries, label)
			return nil
		})
		return entries, err
	}

	dirEntries, err := os.ReadDir(targetPath)
	if err != nil {
		return nil, err
	}
	entries := make([]string, 0, len(dirEntries))
	for _, entry := range dirEntries {
		childPath := filepath.Join(targetPath, entry.Name())
		rel, relErr := filepath.Rel(workspaceRoot, childPath)
		if relErr != nil {
			return nil, relErr
		}
		label := filepath.ToSlash(rel)
		if entry.IsDir() {
			label += "/"
		}
		entries = append(entries, label)
	}
	return entries, nil
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(strings.TrimSpace(text))
	if len(runes) <= limit {
		return string(runes)
	}
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		return ""
	}
}

func boolValue(value any) bool {
	typed, ok := value.(bool)
	return ok && typed
}
