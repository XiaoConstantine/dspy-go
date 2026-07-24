package native

import (
	"context"
	"time"
)

// SessionEventPayload is a native-only typed session lifecycle notification.
type SessionEventPayload interface {
	sessionEventPayload()
}

// SessionEvent wraps one native-only session lifecycle notification.
type SessionEvent struct {
	Timestamp time.Time
	Payload   SessionEventPayload
}

// SessionEventSink consumes typed native-only session lifecycle notifications.
type SessionEventSink interface {
	EmitSessionEvent(context.Context, SessionEvent)
}

// SessionEventSinkFunc adapts a function to SessionEventSink.
type SessionEventSinkFunc func(context.Context, SessionEvent)

// EmitSessionEvent implements SessionEventSink.
func (f SessionEventSinkFunc) EmitSessionEvent(ctx context.Context, event SessionEvent) {
	if f != nil {
		f(ctx, event)
	}
}

// SessionLoadedEvent reports session recall/loading metadata before a run.
type SessionLoadedEvent struct {
	TaskID            string
	SessionID         string
	Source            string
	RecordCount       int
	EntryCount        int
	SummaryCount      int
	RecallChars       int
	BranchID          string
	HeadEntryID       string
	ForkedFromEntryID string
	Err               error
}

func (SessionLoadedEvent) sessionEventPayload() {}

// SessionPersistedEvent reports session persistence outcomes after a run.
type SessionPersistedEvent struct {
	SessionID         string
	RecordID          string
	TaskID            string
	Success           bool
	Completed         bool
	Err               error
	EventStoreEnabled bool
	EventStoreSuccess bool
	EventEntryCount   int
	EventBranchID     string
	EventHeadEntryID  string
	EventStoreErr     error
}

func (SessionPersistedEvent) sessionEventPayload() {}
