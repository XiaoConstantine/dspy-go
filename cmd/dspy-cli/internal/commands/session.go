package commands

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/spf13/cobra"
)

type sessionCommandConfig struct {
	DBPath string
}

type sessionState struct {
	Session      *sessionevent.Session
	Branches     []sessionevent.SessionBranch
	ActiveBranch *sessionevent.SessionBranch
	HeadEntry    *sessionevent.SessionEntry
}

// NewSessionCommand exposes minimal session inspection and branch control commands.
func NewSessionCommand() *cobra.Command {
	cfg := &sessionCommandConfig{}

	cmd := &cobra.Command{
		Use:   "session",
		Short: "Inspect and control persisted native-agent sessions",
		Long: `Inspect session state stored in the SQLite session event store and perform
basic branch control operations such as switching the active branch or forking
from the current active head.`,
	}
	cmd.PersistentFlags().StringVar(&cfg.DBPath, "db", "", "Path to the SQLite session event store")
	_ = cmd.MarkPersistentFlagRequired("db")

	cmd.AddCommand(newSessionShowCommand(cfg))
	cmd.AddCommand(newSessionSwitchCommand(cfg))
	cmd.AddCommand(newSessionForkCommand(cfg))
	return cmd
}

func newSessionShowCommand(cfg *sessionCommandConfig) *cobra.Command {
	return &cobra.Command{
		Use:   "show <session-id>",
		Short: "Show session state and branches",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := openSessionStore(cfg)
			if err != nil {
				return err
			}
			defer store.Close()

			state, err := loadSessionState(cmd.Context(), store, args[0])
			if err != nil {
				return err
			}
			renderSessionState(cmd.OutOrStdout(), state)
			return nil
		},
	}
}

func newSessionSwitchCommand(cfg *sessionCommandConfig) *cobra.Command {
	return &cobra.Command{
		Use:   "switch <session-id> <branch-id>",
		Short: "Switch the active branch for a session",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := openSessionStore(cfg)
			if err != nil {
				return err
			}
			defer store.Close()

			head, err := switchSessionBranch(cmd.Context(), store, args[0], args[1])
			if err != nil {
				return err
			}

			if head == nil {
				_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Active branch for session %s set to %s (empty branch)\n", args[0], args[1])
				return nil
			}
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Active branch for session %s set to %s (head: %s)\n", args[0], args[1], head.ID)
			return nil
		},
	}
}

func newSessionForkCommand(cfg *sessionCommandConfig) *cobra.Command {
	var (
		name     string
		activate bool
	)

	cmd := &cobra.Command{
		Use:   "fork <session-id>",
		Short: "Fork the active branch from its current head",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := openSessionStore(cfg)
			if err != nil {
				return err
			}
			defer store.Close()

			state, err := loadSessionState(cmd.Context(), store, args[0])
			if err != nil {
				return err
			}
			if state.HeadEntry == nil {
				return fmt.Errorf("session %s has no active head entry to fork", args[0])
			}

			branch, err := store.ForkBranch(cmd.Context(), args[0], state.HeadEntry.ID, name, nil)
			if err != nil {
				return err
			}
			if activate {
				if _, err := switchSessionBranch(cmd.Context(), store, args[0], branch.ID); err != nil {
					return err
				}
			}

			activeSuffix := ""
			if activate {
				activeSuffix = " and activated"
			}
			_, _ = fmt.Fprintf(cmd.OutOrStdout(), "Forked branch %s from %s%s\n", branch.ID, state.HeadEntry.ID, activeSuffix)
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "Optional name for the new branch")
	cmd.Flags().BoolVar(&activate, "activate", false, "Make the new branch active immediately")
	return cmd
}

func openSessionStore(cfg *sessionCommandConfig) (*sessionevent.SQLiteStore, error) {
	if cfg == nil || strings.TrimSpace(cfg.DBPath) == "" {
		return nil, fmt.Errorf("session database path is required")
	}
	path := strings.TrimSpace(cfg.DBPath)
	if path != ":memory:" {
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				return nil, fmt.Errorf("session database %q not found", path)
			}
			return nil, fmt.Errorf("stat session database %q: %w", path, err)
		}
	}
	return sessionevent.NewSQLiteStore(path)
}

func loadSessionState(ctx context.Context, store sessionevent.SessionEventStore, sessionID string) (*sessionState, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, fmt.Errorf("session id is required")
	}

	session, err := store.GetSession(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	branches, err := store.ListBranches(ctx, sessionID)
	if err != nil {
		return nil, err
	}

	state := &sessionState{
		Session:  session,
		Branches: branches,
	}

	activeBranchID := strings.TrimSpace(session.ActiveBranchID)
	if activeBranchID == "" {
		return state, nil
	}
	for i := range branches {
		if branches[i].ID == activeBranchID {
			state.ActiveBranch = &branches[i]
			break
		}
	}
	if state.ActiveBranch == nil {
		return nil, fmt.Errorf("active branch %q not found for session %q", activeBranchID, sessionID)
	}

	head, err := store.GetBranchHead(ctx, sessionID, activeBranchID)
	if err != nil {
		return nil, err
	}
	state.HeadEntry = head
	return state, nil
}

func switchSessionBranch(ctx context.Context, store sessionevent.SessionEventStore, sessionID, branchID string) (*sessionevent.SessionEntry, error) {
	head, err := store.GetBranchHead(ctx, sessionID, branchID)
	if err != nil {
		return nil, err
	}
	if err := store.SetActiveBranch(ctx, sessionID, branchID); err != nil {
		return nil, err
	}
	return head, nil
}

func renderSessionState(w io.Writer, state *sessionState) {
	if state == nil || state.Session == nil {
		return
	}

	_, _ = fmt.Fprintf(w, "Session: %s\n", state.Session.ID)
	if strings.TrimSpace(state.Session.Title) != "" {
		_, _ = fmt.Fprintf(w, "Title: %s\n", state.Session.Title)
	}
	_, _ = fmt.Fprintf(w, "Status: %s\n", state.Session.Status)
	if state.ActiveBranch != nil {
		_, _ = fmt.Fprintf(w, "Active branch: %s (%s)\n", state.ActiveBranch.ID, state.ActiveBranch.Name)
	} else {
		_, _ = fmt.Fprintln(w, "Active branch: <none>")
	}
	if state.HeadEntry != nil {
		_, _ = fmt.Fprintf(w, "Head entry: %s [%s] %s\n", state.HeadEntry.ID, state.HeadEntry.Kind, sessionEntrySummary(state.HeadEntry))
	} else {
		_, _ = fmt.Fprintln(w, "Head entry: <none>")
	}
	_, _ = fmt.Fprintln(w, "Branches:")
	for _, branch := range state.Branches {
		prefix := " "
		if state.ActiveBranch != nil && branch.ID == state.ActiveBranch.ID {
			prefix = "*"
		}
		_, _ = fmt.Fprintf(w, "%s %s  %s", prefix, branch.ID, branch.Name)
		if origin := strings.TrimSpace(branch.OriginEntryID); origin != "" {
			_, _ = fmt.Fprintf(w, "  origin=%s", origin)
		}
		if head := strings.TrimSpace(branch.HeadEntryID); head != "" {
			_, _ = fmt.Fprintf(w, "  head=%s", head)
		}
		_, _ = fmt.Fprintln(w)
	}
}

func sessionEntrySummary(entry *sessionevent.SessionEntry) string {
	if entry == nil {
		return ""
	}
	for _, key := range []string{"text", "final_answer", "event", "observation_display", "observation"} {
		value := strings.TrimSpace(fmt.Sprint(entry.Payload[key]))
		if value != "" && value != "<nil>" {
			return value
		}
	}
	return strings.TrimSpace(entry.SearchText)
}
