package sessionevent

import "context"

type SessionWriter interface {
	CreateSession(ctx context.Context, params CreateSessionParams) (*Session, *SessionBranch, error)
	AppendEntries(ctx context.Context, entries []SessionEntry) ([]SessionEntry, error)
	AppendSummary(ctx context.Context, summary SessionSummary) error
	SetActiveBranch(ctx context.Context, sessionID, branchID string) error
	ForkBranch(ctx context.Context, sessionID, fromEntryID, name string, metadata map[string]any) (*SessionBranch, error)
}

type SessionReader interface {
	GetSession(ctx context.Context, sessionID string) (*Session, error)
	ListBranches(ctx context.Context, sessionID string) ([]SessionBranch, error)
	GetEntry(ctx context.Context, sessionID, entryID string) (*SessionEntry, error)
	GetBranchHead(ctx context.Context, sessionID, branchID string) (*SessionEntry, error)
	LoadLineage(ctx context.Context, sessionID, headEntryID string, opts LoadOptions) ([]SessionEntry, error)
	LoadSummaries(ctx context.Context, sessionID, branchID string, limit int) ([]SessionSummary, error)
}

type SessionEventStore interface {
	SessionWriter
	SessionReader
}
