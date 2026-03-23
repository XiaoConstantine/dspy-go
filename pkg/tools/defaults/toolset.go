package defaults

import (
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools/bash"
	"github.com/XiaoConstantine/dspy-go/pkg/tools/files"
)

type Config struct {
	Root               string
	ShellPath          string
	CommandTimeout     time.Duration
	ModelOutputLimit   int
	DisplayOutputLimit int
}

type Toolset struct {
	root  string
	tools []core.Tool
}

func NewToolset(cfg Config) (*Toolset, error) {
	fileTools, err := files.NewToolset(files.Config{
		Root:               cfg.Root,
		ModelOutputLimit:   cfg.ModelOutputLimit,
		DisplayOutputLimit: cfg.DisplayOutputLimit,
	})
	if err != nil {
		return nil, err
	}
	bashTool, err := bash.NewTool(bash.Config{
		Root:               fileTools.Root(),
		ShellPath:          cfg.ShellPath,
		Timeout:            cfg.CommandTimeout,
		ModelOutputLimit:   cfg.ModelOutputLimit,
		DisplayOutputLimit: cfg.DisplayOutputLimit,
	})
	if err != nil {
		return nil, err
	}
	tools := append([]core.Tool{}, fileTools.Tools()...)
	tools = append(tools, bashTool)
	return &Toolset{
		root:  fileTools.Root(),
		tools: tools,
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
	result := make([]core.Tool, len(t.tools))
	copy(result, t.tools)
	return result
}
