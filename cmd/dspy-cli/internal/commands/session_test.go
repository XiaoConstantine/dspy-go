package commands

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionShowCommand_PrintsSessionState(t *testing.T) {
	store, path := newTestSessionEventStore(t)
	session, branch, head := seedSessionForCommands(t, store, "session-show")

	cmd := NewSessionCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--db", path, "show", session.ID})

	require.NoError(t, cmd.Execute())
	output := out.String()
	assert.Contains(t, output, "Session: "+session.ID)
	assert.Contains(t, output, "Active branch: "+branch.ID)
	assert.Contains(t, output, "Head entry: "+head.ID)
	assert.Contains(t, output, "seed answer")
}

func TestSessionSwitchCommand_UpdatesActiveBranch(t *testing.T) {
	store, path := newTestSessionEventStore(t)
	session, _, head := seedSessionForCommands(t, store, "session-switch")

	forked, err := store.ForkBranch(context.Background(), session.ID, head.ID, "alt-path", nil)
	require.NoError(t, err)

	cmd := NewSessionCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--db", path, "switch", session.ID, forked.ID})

	require.NoError(t, cmd.Execute())
	assert.Contains(t, out.String(), "set to "+forked.ID)

	updated, err := store.GetSession(context.Background(), session.ID)
	require.NoError(t, err)
	assert.Equal(t, forked.ID, updated.ActiveBranchID)
}

func TestSessionForkCommand_KeepsActiveBranchByDefault(t *testing.T) {
	store, path := newTestSessionEventStore(t)
	session, branch, _ := seedSessionForCommands(t, store, "session-fork")

	cmd := NewSessionCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--db", path, "fork", session.ID, "--name", "alt-branch"})

	require.NoError(t, cmd.Execute())
	assert.Contains(t, out.String(), "Forked branch")

	updated, err := store.GetSession(context.Background(), session.ID)
	require.NoError(t, err)
	assert.Equal(t, branch.ID, updated.ActiveBranchID)

	branches, err := store.ListBranches(context.Background(), session.ID)
	require.NoError(t, err)
	require.Len(t, branches, 2)
	assert.Contains(t, out.String(), branchIDByName(branches, "alt-branch"))
}

func TestSessionForkCommand_ActivatesWhenRequested(t *testing.T) {
	store, path := newTestSessionEventStore(t)
	session, _, _ := seedSessionForCommands(t, store, "session-fork-activate")

	cmd := NewSessionCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--db", path, "fork", session.ID, "--name", "active-branch", "--activate"})

	require.NoError(t, cmd.Execute())
	assert.Contains(t, out.String(), "and activated")

	updated, err := store.GetSession(context.Background(), session.ID)
	require.NoError(t, err)

	branches, err := store.ListBranches(context.Background(), session.ID)
	require.NoError(t, err)
	require.Len(t, branches, 2)
	assert.Equal(t, branchIDByName(branches, "active-branch"), updated.ActiveBranchID)
}

func TestSessionShowCommand_MissingDatabaseFailsWithoutCreatingFile(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "missing-session-events.db")

	cmd := NewSessionCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs([]string{"--db", missingPath, "show", "session-missing"})

	err := cmd.Execute()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
	_, statErr := os.Stat(missingPath)
	assert.Error(t, statErr)
	assert.True(t, os.IsNotExist(statErr))
}

func newTestSessionEventStore(t *testing.T) (*sessionevent.SQLiteStore, string) {
	t.Helper()

	path := filepath.Join(t.TempDir(), "session-events.db")
	store, err := sessionevent.NewSQLiteStore(path)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, store.Close())
	})
	return store, path
}

func seedSessionForCommands(t *testing.T, store *sessionevent.SQLiteStore, sessionID string) (*sessionevent.Session, *sessionevent.SessionBranch, *sessionevent.SessionEntry) {
	t.Helper()

	ctx := context.Background()
	session, branch, err := store.CreateSession(ctx, sessionevent.CreateSessionParams{
		ID:    sessionID,
		Title: "Seed session",
	})
	require.NoError(t, err)

	inserted, err := store.AppendEntries(ctx, []sessionevent.SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       sessionevent.EntryKindUserMessage,
			Role:       "user",
			SearchText: "seed task",
			Payload:    map[string]any{"text": "seed task"},
		},
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "seed answer",
			Payload:    map[string]any{"text": "seed answer"},
		},
	})
	require.NoError(t, err)
	require.Len(t, inserted, 2)
	return session, branch, &inserted[1]
}

func branchIDByName(branches []sessionevent.SessionBranch, name string) string {
	for _, branch := range branches {
		if branch.Name == name {
			return branch.ID
		}
	}
	return ""
}
