package sessionevent

import "time"

type SessionStatus string

const (
	SessionStatusActive   SessionStatus = "active"
	SessionStatusArchived SessionStatus = "archived"
)

type BranchStatus string

const (
	BranchStatusActive   BranchStatus = "active"
	BranchStatusArchived BranchStatus = "archived"
)

type EntryKind string

const (
	EntryKindUserMessage      EntryKind = "user_message"
	EntryKindAssistantMessage EntryKind = "assistant_message"
	EntryKindToolCall         EntryKind = "tool_call"
	EntryKindToolResult       EntryKind = "tool_result"
	EntryKindLabel            EntryKind = "label"
	EntryKindSystemEvent      EntryKind = "system_event"
)

type SummaryKind string

const (
	SummaryKindRange  SummaryKind = "range"
	SummaryKindBranch SummaryKind = "branch"
)

type Session struct {
	ID             string
	Title          string
	Status         SessionStatus
	ActiveBranchID string
	CreatedAt      time.Time
	UpdatedAt      time.Time
	Metadata       map[string]any
}

type SessionBranch struct {
	ID            string
	SessionID     string
	Name          string
	OriginEntryID string
	HeadEntryID   string
	Status        BranchStatus
	CreatedAt     time.Time
	UpdatedAt     time.Time
	Metadata      map[string]any
}

type SessionEntry struct {
	ID               string
	SessionID        string
	BranchID         string
	ParentID         string
	Kind             EntryKind
	CreatedAt        time.Time
	Role             string
	ToolName         string
	IsError          bool
	Synthetic        bool
	Redacted         bool
	Truncated        bool
	Payload          map[string]any
	SearchText       string
	Metadata         map[string]any
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
}

type SessionSummary struct {
	ID              string
	SessionID       string
	BranchID        string
	StartEntryID    string
	EndEntryID      string
	ParentSummaryID string
	Kind            SummaryKind
	CreatedAt       time.Time
	SummaryText     string
	Metadata        map[string]any
}

// LoadOptions controls how lineage is loaded from a head entry backward.
type LoadOptions struct {
	// MaxEntries limits the number of lineage entries returned. Zero means no explicit limit.
	MaxEntries int
	// StopAtEntryID stops lineage traversal before the matching ancestor. The stop entry itself is excluded.
	StopAtEntryID string
	// PreferSummary is reserved for future summary-aware lineage loading.
	PreferSummary bool
}

type CreateSessionParams struct {
	ID         string
	Title      string
	BranchName string
	Metadata   map[string]any
}
