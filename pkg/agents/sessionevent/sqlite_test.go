package sessionevent

import (
	"context"
	goerrors "errors"
	"path/filepath"
	"testing"
	"time"

	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLiteStoreCreateSessionCreatesDefaultBranch(t *testing.T) {
	t.Parallel()

	store := newTestSQLiteStore(t)
	defer store.Close()

	ctx := context.Background()
	session, branch, err := store.CreateSession(ctx, CreateSessionParams{
		Title:    "Plan quarterly review",
		Metadata: map[string]any{"team": "platform"},
	})
	require.NoError(t, err)

	require.NotEmpty(t, session.ID)
	require.NotEmpty(t, branch.ID)
	assert.Equal(t, session.ID, branch.SessionID)
	assert.Equal(t, branch.ID, session.ActiveBranchID)
	assert.Equal(t, SessionStatusActive, session.Status)
	assert.Equal(t, BranchStatusActive, branch.Status)
	assert.Equal(t, defaultBranchName, branch.Name)
	assert.Equal(t, "platform", session.Metadata["team"])

	loadedSession, err := store.GetSession(ctx, session.ID)
	require.NoError(t, err)
	assert.Equal(t, session.ID, loadedSession.ID)
	assert.Equal(t, branch.ID, loadedSession.ActiveBranchID)

	branches, err := store.ListBranches(ctx, session.ID)
	require.NoError(t, err)
	require.Len(t, branches, 1)
	assert.Equal(t, branch.ID, branches[0].ID)
	assert.Empty(t, branches[0].HeadEntryID)
}

func TestSQLiteStoreAppendEntriesAndLoadLineage(t *testing.T) {
	t.Parallel()

	store := newTestSQLiteStore(t)
	defer store.Close()

	ctx := context.Background()
	session, branch, err := store.CreateSession(ctx, CreateSessionParams{Title: "Investigate flaky tests"})
	require.NoError(t, err)

	entries := []SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       EntryKindUserMessage,
			Role:       "user",
			Payload:    map[string]any{"text": "Investigate the failures"},
			SearchText: "Investigate the failures",
		},
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       EntryKindAssistantMessage,
			Role:       "assistant",
			Payload:    map[string]any{"text": "Checking the failing suite"},
			SearchText: "Checking the failing suite",
		},
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       EntryKindToolResult,
			ToolName:   "exec_command",
			Payload:    map[string]any{"stdout": "2 tests failing"},
			SearchText: "2 tests failing",
		},
	}

	inserted, err := store.AppendEntries(ctx, entries)
	require.NoError(t, err)
	require.Len(t, inserted, 3)
	assert.NotEmpty(t, inserted[0].ID)
	assert.Equal(t, inserted[0].ID, inserted[1].ParentID)
	assert.Equal(t, inserted[1].ID, inserted[2].ParentID)

	head, err := store.GetBranchHead(ctx, session.ID, branch.ID)
	require.NoError(t, err)
	require.NotNil(t, head)
	assert.Equal(t, EntryKindToolResult, head.Kind)
	assert.Equal(t, inserted[2].ID, head.ID)

	lineage, err := store.LoadLineage(ctx, session.ID, head.ID, LoadOptions{})
	require.NoError(t, err)
	require.Len(t, lineage, 3)
	assert.Equal(t, EntryKindUserMessage, lineage[0].Kind)
	assert.Equal(t, EntryKindAssistantMessage, lineage[1].Kind)
	assert.Equal(t, EntryKindToolResult, lineage[2].Kind)
	assert.Equal(t, lineage[0].ID, lineage[1].ParentID)
	assert.Equal(t, lineage[1].ID, lineage[2].ParentID)
}

func TestSQLiteStoreLoadLineageRespectsStopAtAndMaxEntries(t *testing.T) {
	t.Parallel()

	store := newTestSQLiteStore(t)
	defer store.Close()

	ctx := context.Background()
	session, branch, err := store.CreateSession(ctx, CreateSessionParams{Title: "Lineage controls"})
	require.NoError(t, err)

	entries := []SessionEntry{
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindUserMessage, Payload: map[string]any{"text": "a"}},
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindAssistantMessage, Payload: map[string]any{"text": "b"}},
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindToolCall, Payload: map[string]any{"tool": "read"}},
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindToolResult, Payload: map[string]any{"stdout": "done"}},
	}
	inserted, err := store.AppendEntries(ctx, entries)
	require.NoError(t, err)
	require.Len(t, inserted, 4)

	head, err := store.GetBranchHead(ctx, session.ID, branch.ID)
	require.NoError(t, err)
	require.NotNil(t, head)

	limited, err := store.LoadLineage(ctx, session.ID, head.ID, LoadOptions{MaxEntries: 2})
	require.NoError(t, err)
	require.Len(t, limited, 2)
	assert.Equal(t, EntryKindToolCall, limited[0].Kind)
	assert.Equal(t, EntryKindToolResult, limited[1].Kind)

	stopAtID := limited[0].ID
	stopped, err := store.LoadLineage(ctx, session.ID, head.ID, LoadOptions{StopAtEntryID: stopAtID})
	require.NoError(t, err)
	require.Len(t, stopped, 1)
	assert.Equal(t, head.ID, stopped[0].ID)
}

func TestSQLiteStoreAppendSummaryAndForkBranch(t *testing.T) {
	t.Parallel()

	store := newTestSQLiteStore(t)
	defer store.Close()

	ctx := context.Background()
	session, branch, err := store.CreateSession(ctx, CreateSessionParams{Title: "Forkable session"})
	require.NoError(t, err)

	inserted, err := store.AppendEntries(ctx, []SessionEntry{
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindUserMessage, Payload: map[string]any{"text": "start"}},
		{SessionID: session.ID, BranchID: branch.ID, Kind: EntryKindAssistantMessage, Payload: map[string]any{"text": "working"}},
	})
	require.NoError(t, err)
	require.Len(t, inserted, 2)

	head, err := store.GetBranchHead(ctx, session.ID, branch.ID)
	require.NoError(t, err)
	require.NotNil(t, head)

	lineage, err := store.LoadLineage(ctx, session.ID, head.ID, LoadOptions{})
	require.NoError(t, err)
	require.Len(t, lineage, 2)

	summaryTime := time.Now().UTC().Add(time.Second)
	require.NoError(t, store.AppendSummary(ctx, SessionSummary{
		SessionID:    session.ID,
		BranchID:     branch.ID,
		StartEntryID: lineage[0].ID,
		EndEntryID:   lineage[1].ID,
		CreatedAt:    summaryTime,
		SummaryText:  "Initial branch summary",
		Metadata:     map[string]any{"stage": "initial"},
	}))

	summaries, err := store.LoadSummaries(ctx, session.ID, branch.ID, 10)
	require.NoError(t, err)
	require.Len(t, summaries, 1)
	assert.Equal(t, "Initial branch summary", summaries[0].SummaryText)
	assert.Equal(t, "initial", summaries[0].Metadata["stage"])

	forked, err := store.ForkBranch(ctx, session.ID, head.ID, "alt-path", map[string]any{"owner": "tests"})
	require.NoError(t, err)
	assert.Equal(t, session.ID, forked.SessionID)
	assert.Equal(t, head.ID, forked.OriginEntryID)
	assert.Equal(t, head.ID, forked.HeadEntryID)
	assert.Equal(t, "tests", forked.Metadata["owner"])

	require.NoError(t, store.SetActiveBranch(ctx, session.ID, forked.ID))

	updatedSession, err := store.GetSession(ctx, session.ID)
	require.NoError(t, err)
	assert.Equal(t, forked.ID, updatedSession.ActiveBranchID)
}

func TestSQLiteStoreReturnsInvalidInputForUnmarshalableMaps(t *testing.T) {
	t.Parallel()

	store := newTestSQLiteStore(t)
	defer store.Close()

	ctx := context.Background()

	_, _, err := store.CreateSession(ctx, CreateSessionParams{
		Title:    "bad metadata",
		Metadata: map[string]any{"bad": make(chan int)},
	})
	require.Error(t, err)
	assertErrorCode(t, err, dspyerrors.InvalidInput)

	session, branch, err := store.CreateSession(ctx, CreateSessionParams{Title: "good"})
	require.NoError(t, err)

	_, err = store.AppendEntries(ctx, []SessionEntry{{
		SessionID: session.ID,
		BranchID:  branch.ID,
		Kind:      EntryKindToolResult,
		Payload:   map[string]any{"bad": make(chan int)},
	}})
	require.Error(t, err)
	assertErrorCode(t, err, dspyerrors.InvalidInput)
}

func newTestSQLiteStore(t *testing.T) *SQLiteStore {
	t.Helper()

	path := filepath.Join(t.TempDir(), "session-events.db")
	store, err := NewSQLiteStore(path)
	require.NoError(t, err)
	return store
}

func assertErrorCode(t *testing.T, err error, code dspyerrors.ErrorCode) {
	t.Helper()

	var dspyErr *dspyerrors.Error
	require.True(t, goerrors.As(err, &dspyErr), "expected dspy error, got %T", err)
	assert.Equal(t, code, dspyErr.Code())
}
