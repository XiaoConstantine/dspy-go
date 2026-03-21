package agents

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionStoreAppendAndRecent(t *testing.T) {
	store := NewSessionStore(NewInMemoryStore())
	now := time.Date(2026, time.March, 21, 10, 0, 0, 0, time.UTC)

	require.NoError(t, store.Append(SessionRecord{
		ID:          "run-1",
		SessionID:   "session-1",
		AgentType:   "native",
		Task:        "first task",
		StartedAt:   now,
		CompletedAt: now.Add(time.Minute),
		Completed:   true,
		FinalAnswer: "first answer",
		Highlights:  []string{"write_file: created note"},
		Metadata:    map[string]any{"turns": 2},
	}))
	require.NoError(t, store.Append(SessionRecord{
		ID:          "run-2",
		SessionID:   "session-1",
		AgentType:   "native",
		Task:        "second task",
		StartedAt:   now.Add(2 * time.Minute),
		CompletedAt: now.Add(3 * time.Minute),
		Completed:   false,
		Error:       "validation failed",
	}))
	require.NoError(t, store.Append(SessionRecord{
		ID:        "run-3",
		SessionID: "session-2",
		AgentType: "native",
		Task:      "other session",
		StartedAt: now.Add(4 * time.Minute),
	}))

	records, err := store.Recent("session-1", 5)
	require.NoError(t, err)
	require.Len(t, records, 2)
	assert.Equal(t, "run-1", records[0].ID)
	assert.Equal(t, "run-2", records[1].ID)
	assert.Equal(t, "first answer", records[0].FinalAnswer)
	assert.Equal(t, "validation failed", records[1].Error)
	assert.Equal(t, []string{"write_file: created note"}, records[0].Highlights)
	assert.Equal(t, map[string]any{"turns": float64(2)}, records[0].Metadata)

	limited, err := store.Recent("session-1", 1)
	require.NoError(t, err)
	require.Len(t, limited, 1)
	assert.Equal(t, "run-2", limited[0].ID)

	empty, err := store.Recent("missing", 3)
	require.NoError(t, err)
	assert.Empty(t, empty)
}

func TestSessionStoreAppendRejectsInvalidRecords(t *testing.T) {
	store := NewSessionStore(NewInMemoryStore())

	err := store.Append(SessionRecord{ID: "run-1", Task: "missing session"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "session_id")

	err = store.Append(SessionRecord{ID: "run-1", SessionID: "session-1"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "task")
}
