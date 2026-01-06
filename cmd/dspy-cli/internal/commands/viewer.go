package commands

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/viewer"
)

// NewViewerCommand creates a new viewer command for viewing dspy-go JSONL session logs.
func NewViewerCommand() *cobra.Command {
	cfg := viewer.Config{Iteration: -1}

	cmd := &cobra.Command{
		Use:   "view <file.jsonl>",
		Short: "View dspy-go JSONL session logs",
		Long: `Enhanced CLI viewer for dspy-go JSONL session logs.

The viewer auto-detects and supports both:
- RLM format (iteration-based logs)
- Native DSPy format (event-based logs)

Features:
- Log parsing and display with colored output
- Interactive navigation mode
- Search functionality with highlighting
- Statistics and metrics visualization
- Watch mode (live tail)
- Markdown export

Examples:
  dspy-cli view session.jsonl              # View log
  dspy-cli view -i session.jsonl           # Interactive mode
  dspy-cli view -w session.jsonl           # Watch live
  dspy-cli view --stats session.jsonl      # Statistics only
  dspy-cli view --iter 3 session.jsonl     # Show iteration 3 (RLM)
  dspy-cli view -s "error" session.jsonl   # Search for "error"`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			filename := args[0]

			if cfg.NoColor {
				viewer.DisableColors()
			}

			if cfg.Watch {
				return viewer.WatchLog(filename, cfg)
			}

			data, err := viewer.ParseLog(filename)
			if err != nil {
				return fmt.Errorf("error parsing log: %w", err)
			}

			if cfg.Export != "" {
				if err := viewer.ExportLog(data, cfg); err != nil {
					return fmt.Errorf("error exporting: %w", err)
				}
				fmt.Printf("Exported to: %s\n", cfg.Export)
				return nil
			}

			if cfg.Interactive {
				viewer.RunInteractive(data, cfg)
				return nil
			}

			viewer.ViewLog(data, cfg)
			return nil
		},
	}

	// Flags
	cmd.Flags().BoolVarP(&cfg.Compact, "compact", "c", false, "Compact output (hide full responses)")
	cmd.Flags().BoolVarP(&cfg.Interactive, "interactive", "i", false, "Interactive navigation mode")
	cmd.Flags().BoolVarP(&cfg.Watch, "watch", "w", false, "Watch file for changes (live tail)")
	cmd.Flags().IntVar(&cfg.Iteration, "iter", -1, "Show only specific iteration (1-indexed)")
	cmd.Flags().BoolVar(&cfg.ErrorsOnly, "errors", false, "Show only iterations with errors")
	cmd.Flags().BoolVar(&cfg.FinalOnly, "final", false, "Show only the final answer")
	cmd.Flags().StringVarP(&cfg.Search, "search", "s", "", "Search for text in responses/code")
	cmd.Flags().BoolVar(&cfg.Stats, "stats", false, "Show detailed statistics only")
	cmd.Flags().StringVar(&cfg.Export, "export", "", "Export to markdown file (.md)")
	cmd.Flags().BoolVar(&cfg.NoColor, "no-color", false, "Disable colored output")

	return cmd
}
