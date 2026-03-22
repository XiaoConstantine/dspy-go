package sessionevent

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
)

const defaultBranchName = "main"

type SQLiteStore struct {
	db *sql.DB
}

var _ SessionEventStore = (*SQLiteStore)(nil)

func NewSQLiteStore(path string) (*SQLiteStore, error) {
	if strings.TrimSpace(path) == "" {
		path = ":memory:"
	}
	if path != ":memory:" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, dspyerrors.WithFields(
				dspyerrors.Wrap(err, dspyerrors.Unknown, "failed to create sqlite store directory"),
				dspyerrors.Fields{"path": path},
			)
		}
	}

	db, err := sql.Open("sqlite3", sqliteDSN(path))
	if err != nil {
		return nil, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.Unknown, "failed to open session event sqlite database"),
			dspyerrors.Fields{"path": path},
		)
	}

	store := &SQLiteStore{
		db: db,
	}
	if _, err := db.Exec(sessionSchema); err != nil {
		_ = db.Close()
		return nil, wrapSQLError(err, "failed to initialize session event schema", nil)
	}
	return store, nil
}

func (s *SQLiteStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *SQLiteStore) CreateSession(ctx context.Context, params CreateSessionParams) (*Session, *SessionBranch, error) {
	sessionID := strings.TrimSpace(params.ID)
	if sessionID == "" {
		sessionID = uuid.NewString()
	}
	branchID := uuid.NewString()
	now := time.Now().UTC()
	branchName := strings.TrimSpace(params.BranchName)
	if branchName == "" {
		branchName = defaultBranchName
	}

	session := &Session{
		ID:             sessionID,
		Title:          strings.TrimSpace(params.Title),
		Status:         SessionStatusActive,
		ActiveBranchID: branchID,
		CreatedAt:      now,
		UpdatedAt:      now,
		Metadata:       core.ShallowCopyMap(params.Metadata),
	}
	branch := &SessionBranch{
		ID:        branchID,
		SessionID: sessionID,
		Name:      branchName,
		Status:    BranchStatusActive,
		CreatedAt: now,
		UpdatedAt: now,
		Metadata:  map[string]any{},
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, wrapSQLError(err, "failed to begin session create transaction", nil)
	}
	defer rollbackTx(ctx, tx)

	sessionMetadataJSON, err := encodeJSONMap(session.Metadata)
	if err != nil {
		return nil, nil, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode session metadata"),
			dspyerrors.Fields{"session_id": session.ID},
		)
	}
	branchMetadataJSON, err := encodeJSONMap(branch.Metadata)
	if err != nil {
		return nil, nil, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode branch metadata"),
			dspyerrors.Fields{"session_id": session.ID, "branch_id": branch.ID},
		)
	}

	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO sessions (id, title, status, created_at, updated_at, active_branch_id, metadata_json)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		session.ID,
		session.Title,
		string(session.Status),
		formatTime(session.CreatedAt),
		formatTime(session.UpdatedAt),
		session.ActiveBranchID,
		sessionMetadataJSON,
	); err != nil {
		return nil, nil, wrapSQLError(err, "failed to insert session", dspyerrors.Fields{"session_id": session.ID})
	}

	if _, err := tx.ExecContext(
		ctx,
		`INSERT INTO session_branches (id, session_id, name, status, created_at, updated_at, origin_entry_id, head_entry_id, metadata_json)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		branch.ID,
		branch.SessionID,
		branch.Name,
		string(branch.Status),
		formatTime(branch.CreatedAt),
		formatTime(branch.UpdatedAt),
		nil,
		nil,
		branchMetadataJSON,
	); err != nil {
		return nil, nil, wrapSQLError(err, "failed to insert default branch", dspyerrors.Fields{"session_id": session.ID, "branch_id": branch.ID})
	}

	if err := tx.Commit(); err != nil {
		return nil, nil, wrapSQLError(err, "failed to commit session create transaction", dspyerrors.Fields{"session_id": session.ID})
	}
	return session, branch, nil
}

func (s *SQLiteStore) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id is required")
	}

	row := s.db.QueryRowContext(ctx,
		`SELECT id, title, status, created_at, updated_at, active_branch_id, metadata_json
		   FROM sessions
		  WHERE id = ?`,
		sessionID,
	)

	session, err := scanSession(row)
	if err != nil {
		return nil, err
	}
	return session, nil
}

func (s *SQLiteStore) ListBranches(ctx context.Context, sessionID string) ([]SessionBranch, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id is required")
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT id, session_id, name, status, created_at, updated_at, origin_entry_id, head_entry_id, metadata_json
		   FROM session_branches
		  WHERE session_id = ?
		  ORDER BY created_at ASC`,
		sessionID,
	)
	if err != nil {
		return nil, wrapSQLError(err, "failed to list session branches", dspyerrors.Fields{"session_id": sessionID})
	}
	defer rows.Close()

	var branches []SessionBranch
	for rows.Next() {
		branch, err := scanBranch(rows)
		if err != nil {
			return nil, err
		}
		branches = append(branches, *branch)
	}
	if err := rows.Err(); err != nil {
		return nil, wrapSQLError(err, "failed to iterate session branches", dspyerrors.Fields{"session_id": sessionID})
	}
	return branches, nil
}

func (s *SQLiteStore) GetEntry(ctx context.Context, sessionID, entryID string) (*SessionEntry, error) {
	sessionID = strings.TrimSpace(sessionID)
	entryID = strings.TrimSpace(entryID)
	if sessionID == "" || entryID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id and entry id are required")
	}

	row := s.db.QueryRowContext(ctx,
		selectEntrySQL+` WHERE session_id = ? AND id = ?`,
		sessionID,
		entryID,
	)
	entry, err := scanEntry(row)
	if err != nil {
		return nil, err
	}
	return entry, nil
}

func (s *SQLiteStore) GetBranchHead(ctx context.Context, sessionID, branchID string) (*SessionEntry, error) {
	sessionID = strings.TrimSpace(sessionID)
	branchID = strings.TrimSpace(branchID)
	if sessionID == "" || branchID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id and branch id are required")
	}

	var headID sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT head_entry_id
		   FROM session_branches
		  WHERE session_id = ? AND id = ?`,
		sessionID,
		branchID,
	).Scan(&headID)
	if err == sql.ErrNoRows {
		return nil, notFound("branch not found", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	if err != nil {
		return nil, wrapSQLError(err, "failed to load branch head", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	if !headID.Valid || strings.TrimSpace(headID.String) == "" {
		return nil, nil
	}

	row := s.db.QueryRowContext(ctx,
		selectEntrySQL+` WHERE session_id = ? AND id = ?`,
		sessionID,
		headID.String,
	)
	return scanEntry(row)
}

func (s *SQLiteStore) LoadLineage(ctx context.Context, sessionID, headEntryID string, opts LoadOptions) ([]SessionEntry, error) {
	sessionID = strings.TrimSpace(sessionID)
	headEntryID = strings.TrimSpace(headEntryID)
	if sessionID == "" || headEntryID == "" {
		// Empty lineage inputs are treated as "nothing to load" so callers can safely chain
		// GetBranchHead -> LoadLineage without special casing empty branches.
		return nil, nil
	}

	stopAt := strings.TrimSpace(opts.StopAtEntryID)
	maxEntries := opts.MaxEntries

	rows, err := s.db.QueryContext(ctx, `
WITH RECURSIVE lineage (
	id, session_id, branch_id, parent_id, kind, created_at, role, tool_name,
	is_error, synthetic, redacted, truncated, payload_json, search_text,
	metadata_json, prompt_tokens, completion_tokens, total_tokens, depth
) AS (
	SELECT
		id, session_id, branch_id, parent_id, kind, created_at, role, tool_name,
		is_error, synthetic, redacted, truncated, payload_json, search_text,
		metadata_json, prompt_tokens, completion_tokens, total_tokens, 1
	FROM session_entries
	WHERE session_id = ? AND id = ?
	  AND (? = '' OR id != ?)

	UNION ALL

	SELECT
		e.id, e.session_id, e.branch_id, e.parent_id, e.kind, e.created_at, e.role, e.tool_name,
		e.is_error, e.synthetic, e.redacted, e.truncated, e.payload_json, e.search_text,
		e.metadata_json, e.prompt_tokens, e.completion_tokens, e.total_tokens, l.depth + 1
	FROM session_entries e
	JOIN lineage l ON e.id = l.parent_id
	WHERE l.parent_id IS NOT NULL
	  AND (? = '' OR e.id != ?)
	  AND (? <= 0 OR l.depth < ?)
)
SELECT
	id, session_id, branch_id, parent_id, kind, created_at, role, tool_name,
	is_error, synthetic, redacted, truncated, payload_json, search_text,
	metadata_json, prompt_tokens, completion_tokens, total_tokens
FROM lineage
ORDER BY depth DESC
`,
		sessionID,
		headEntryID,
		stopAt,
		stopAt,
		stopAt,
		stopAt,
		maxEntries,
		maxEntries,
	)
	if err != nil {
		return nil, wrapSQLError(err, "failed to load lineage", dspyerrors.Fields{"session_id": sessionID, "head_entry_id": headEntryID})
	}
	defer rows.Close()

	var entries []SessionEntry
	for rows.Next() {
		entry, err := scanEntry(rows)
		if err != nil {
			return nil, err
		}
		entries = append(entries, *entry)
	}
	if err := rows.Err(); err != nil {
		return nil, wrapSQLError(err, "failed to iterate lineage rows", dspyerrors.Fields{"session_id": sessionID, "head_entry_id": headEntryID})
	}
	if len(entries) == 0 {
		if stopAt != "" && stopAt == headEntryID {
			exists, err := s.entryExists(ctx, sessionID, headEntryID)
			if err != nil {
				return nil, err
			}
			if exists {
				return []SessionEntry{}, nil
			}
		}
		return nil, notFound("entry not found", dspyerrors.Fields{"session_id": sessionID, "entry_id": headEntryID})
	}
	return entries, nil
}

func (s *SQLiteStore) AppendEntries(ctx context.Context, entries []SessionEntry) ([]SessionEntry, error) {
	if len(entries) == 0 {
		return nil, nil
	}

	baseSessionID := strings.TrimSpace(entries[0].SessionID)
	baseBranchID := strings.TrimSpace(entries[0].BranchID)
	if baseSessionID == "" || baseBranchID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "entries require session_id and branch_id")
	}

	now := time.Now().UTC()

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, wrapSQLError(err, "failed to begin append entries transaction", dspyerrors.Fields{"session_id": baseSessionID, "branch_id": baseBranchID})
	}
	defer rollbackTx(ctx, tx)

	currentHeadID, err := s.lookupBranchHeadTx(ctx, tx, baseSessionID, baseBranchID)
	if err != nil {
		return nil, err
	}

	inserted := make([]SessionEntry, len(entries))
	prevID := currentHeadID
	for i := range entries {
		entry := cloneEntry(entries[i])
		if strings.TrimSpace(entry.SessionID) != baseSessionID || strings.TrimSpace(entry.BranchID) != baseBranchID {
			return nil, dspyerrors.New(dspyerrors.InvalidInput, "all appended entries must share the same session_id and branch_id")
		}
		if entry.ID == "" {
			entry.ID = uuid.NewString()
		}
		if entry.CreatedAt.IsZero() {
			entry.CreatedAt = now
		} else {
			entry.CreatedAt = entry.CreatedAt.UTC()
		}
		if strings.TrimSpace(entry.ParentID) == "" {
			entry.ParentID = prevID
		}
		if entry.Kind == "" {
			return nil, dspyerrors.New(dspyerrors.InvalidInput, "entry kind is required")
		}
		if strings.TrimSpace(entry.ParentID) != "" {
			exists, err := s.entryExistsTx(ctx, tx, entry.SessionID, entry.ParentID)
			if err != nil {
				return nil, err
			}
			if !exists {
				return nil, notFound("parent entry not found", dspyerrors.Fields{
					"session_id": entry.SessionID,
					"branch_id":  entry.BranchID,
					"parent_id":  entry.ParentID,
				})
			}
		}
		payloadJSON, err := encodeJSONMap(entry.Payload)
		if err != nil {
			return nil, dspyerrors.WithFields(
				dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode session entry payload"),
				dspyerrors.Fields{"session_id": entry.SessionID, "branch_id": entry.BranchID, "entry_id": entry.ID},
			)
		}
		metadataJSON, err := encodeJSONMap(entry.Metadata)
		if err != nil {
			return nil, dspyerrors.WithFields(
				dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode session entry metadata"),
				dspyerrors.Fields{"session_id": entry.SessionID, "branch_id": entry.BranchID, "entry_id": entry.ID},
			)
		}

		if _, err := tx.ExecContext(ctx,
			`INSERT INTO session_entries (
				id, session_id, branch_id, parent_id, kind, created_at, role, tool_name,
				is_error, synthetic, redacted, truncated, payload_json, search_text,
				metadata_json, prompt_tokens, completion_tokens, total_tokens
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			entry.ID,
			entry.SessionID,
			entry.BranchID,
			nullIfEmpty(entry.ParentID),
			string(entry.Kind),
			formatTime(entry.CreatedAt),
			entry.Role,
			entry.ToolName,
			boolToInt(entry.IsError),
			boolToInt(entry.Synthetic),
			boolToInt(entry.Redacted),
			boolToInt(entry.Truncated),
			payloadJSON,
			entry.SearchText,
			metadataJSON,
			entry.PromptTokens,
			entry.CompletionTokens,
			entry.TotalTokens,
		); err != nil {
			return nil, wrapSQLError(err, "failed to insert session entry", dspyerrors.Fields{"session_id": entry.SessionID, "branch_id": entry.BranchID, "entry_id": entry.ID})
		}

		prevID = entry.ID
		inserted[i] = entry
	}

	if _, err := tx.ExecContext(ctx,
		`UPDATE session_branches
		    SET head_entry_id = ?, updated_at = ?
		  WHERE session_id = ? AND id = ?`,
		prevID,
		formatTime(now),
		baseSessionID,
		baseBranchID,
	); err != nil {
		return nil, wrapSQLError(err, "failed to update branch head", dspyerrors.Fields{"session_id": baseSessionID, "branch_id": baseBranchID})
	}

	if _, err := tx.ExecContext(ctx,
		`UPDATE sessions
		    SET updated_at = ?
		  WHERE id = ?`,
		formatTime(now),
		baseSessionID,
	); err != nil {
		return nil, wrapSQLError(err, "failed to update session timestamp", dspyerrors.Fields{"session_id": baseSessionID})
	}

	if err := tx.Commit(); err != nil {
		return nil, wrapSQLError(err, "failed to commit append entries transaction", dspyerrors.Fields{"session_id": baseSessionID, "branch_id": baseBranchID})
	}
	return inserted, nil
}

func (s *SQLiteStore) AppendSummary(ctx context.Context, summary SessionSummary) error {
	summary = cloneSummary(summary)
	if strings.TrimSpace(summary.SessionID) == "" || strings.TrimSpace(summary.BranchID) == "" {
		return dspyerrors.New(dspyerrors.InvalidInput, "summary requires session_id and branch_id")
	}
	if strings.TrimSpace(summary.StartEntryID) == "" || strings.TrimSpace(summary.EndEntryID) == "" {
		return dspyerrors.New(dspyerrors.InvalidInput, "summary requires start_entry_id and end_entry_id")
	}
	if strings.TrimSpace(summary.SummaryText) == "" {
		return dspyerrors.New(dspyerrors.InvalidInput, "summary text is required")
	}
	if summary.ID == "" {
		summary.ID = uuid.NewString()
	}
	if summary.CreatedAt.IsZero() {
		summary.CreatedAt = time.Now().UTC()
	} else {
		summary.CreatedAt = summary.CreatedAt.UTC()
	}
	if summary.Kind == "" {
		summary.Kind = SummaryKindRange
	}
	metadataJSON, err := encodeJSONMap(summary.Metadata)
	if err != nil {
		return dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode session summary metadata"),
			dspyerrors.Fields{"session_id": summary.SessionID, "branch_id": summary.BranchID, "summary_id": summary.ID},
		)
	}

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO session_summaries (
			id, session_id, branch_id, start_entry_id, end_entry_id, parent_summary_id, kind, created_at, summary_text, metadata_json
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		summary.ID,
		summary.SessionID,
		summary.BranchID,
		summary.StartEntryID,
		summary.EndEntryID,
		nullIfEmpty(summary.ParentSummaryID),
		string(summary.Kind),
		formatTime(summary.CreatedAt),
		summary.SummaryText,
		metadataJSON,
	)
	if err != nil {
		return wrapSQLError(err, "failed to insert session summary", dspyerrors.Fields{"session_id": summary.SessionID, "branch_id": summary.BranchID, "summary_id": summary.ID})
	}
	return nil
}

func (s *SQLiteStore) LoadSummaries(ctx context.Context, sessionID, branchID string, limit int) ([]SessionSummary, error) {
	sessionID = strings.TrimSpace(sessionID)
	branchID = strings.TrimSpace(branchID)
	if sessionID == "" || branchID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id and branch id are required")
	}
	if limit <= 0 {
		limit = 50
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT id, session_id, branch_id, start_entry_id, end_entry_id, parent_summary_id, kind, created_at, summary_text, metadata_json
		   FROM session_summaries
		  WHERE session_id = ? AND branch_id = ?
		  ORDER BY created_at DESC
		  LIMIT ?`,
		sessionID,
		branchID,
		limit,
	)
	if err != nil {
		return nil, wrapSQLError(err, "failed to list session summaries", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	defer rows.Close()

	var summaries []SessionSummary
	for rows.Next() {
		summary, err := scanSummary(rows)
		if err != nil {
			return nil, err
		}
		summaries = append(summaries, *summary)
	}
	if err := rows.Err(); err != nil {
		return nil, wrapSQLError(err, "failed to iterate session summaries", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	return summaries, nil
}

func (s *SQLiteStore) SetActiveBranch(ctx context.Context, sessionID, branchID string) error {
	sessionID = strings.TrimSpace(sessionID)
	branchID = strings.TrimSpace(branchID)
	if sessionID == "" || branchID == "" {
		return dspyerrors.New(dspyerrors.InvalidInput, "session id and branch id are required")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return wrapSQLError(err, "failed to begin set active branch transaction", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	defer rollbackTx(ctx, tx)

	var exists int
	if err := tx.QueryRowContext(ctx,
		`SELECT 1
		   FROM session_branches
		  WHERE session_id = ? AND id = ?`,
		sessionID,
		branchID,
	).Scan(&exists); err == sql.ErrNoRows {
		return notFound("branch not found", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	} else if err != nil {
		return wrapSQLError(err, "failed to validate active branch", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}

	result, err := tx.ExecContext(ctx,
		`UPDATE sessions
		    SET active_branch_id = ?, updated_at = ?
		  WHERE id = ?`,
		branchID,
		formatTime(time.Now().UTC()),
		sessionID,
	)
	if err != nil {
		return wrapSQLError(err, "failed to set active branch", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	rowsAffected, err := result.RowsAffected()
	if err == nil && rowsAffected == 0 {
		return notFound("session not found", dspyerrors.Fields{"session_id": sessionID})
	}
	if err := tx.Commit(); err != nil {
		return wrapSQLError(err, "failed to commit set active branch transaction", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	return nil
}

func (s *SQLiteStore) ForkBranch(ctx context.Context, sessionID, fromEntryID, name string, metadata map[string]any) (*SessionBranch, error) {
	sessionID = strings.TrimSpace(sessionID)
	fromEntryID = strings.TrimSpace(fromEntryID)
	if sessionID == "" || fromEntryID == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "session id and from entry id are required")
	}

	now := time.Now().UTC()
	branch := &SessionBranch{
		ID:            uuid.NewString(),
		SessionID:     sessionID,
		Name:          strings.TrimSpace(name),
		OriginEntryID: fromEntryID,
		HeadEntryID:   fromEntryID,
		Status:        BranchStatusActive,
		CreatedAt:     now,
		UpdatedAt:     now,
		Metadata:      core.ShallowCopyMap(metadata),
	}
	if branch.Name == "" {
		branch.Name = "branch-" + branch.ID[:8]
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, wrapSQLError(err, "failed to begin fork branch transaction", dspyerrors.Fields{"session_id": sessionID, "from_entry_id": fromEntryID})
	}
	defer rollbackTx(ctx, tx)

	var exists int
	if err := tx.QueryRowContext(ctx,
		`SELECT 1 FROM session_entries WHERE session_id = ? AND id = ?`,
		sessionID,
		fromEntryID,
	).Scan(&exists); err == sql.ErrNoRows {
		return nil, notFound("entry not found", dspyerrors.Fields{"session_id": sessionID, "entry_id": fromEntryID})
	} else if err != nil {
		return nil, wrapSQLError(err, "failed to validate fork origin entry", dspyerrors.Fields{"session_id": sessionID, "entry_id": fromEntryID})
	}
	metadataJSON, err := encodeJSONMap(branch.Metadata)
	if err != nil {
		return nil, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidInput, "failed to encode fork branch metadata"),
			dspyerrors.Fields{"session_id": sessionID, "branch_id": branch.ID},
		)
	}

	if _, err := tx.ExecContext(ctx,
		`INSERT INTO session_branches (id, session_id, name, status, created_at, updated_at, origin_entry_id, head_entry_id, metadata_json)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		branch.ID,
		branch.SessionID,
		branch.Name,
		string(branch.Status),
		formatTime(branch.CreatedAt),
		formatTime(branch.UpdatedAt),
		branch.OriginEntryID,
		branch.HeadEntryID,
		metadataJSON,
	); err != nil {
		return nil, wrapSQLError(err, "failed to insert forked branch", dspyerrors.Fields{"session_id": sessionID, "branch_id": branch.ID})
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE sessions
		    SET updated_at = ?
		  WHERE id = ?`,
		formatTime(now),
		sessionID,
	); err != nil {
		return nil, wrapSQLError(err, "failed to update session timestamp after fork", dspyerrors.Fields{"session_id": sessionID, "branch_id": branch.ID})
	}

	if err := tx.Commit(); err != nil {
		return nil, wrapSQLError(err, "failed to commit fork branch transaction", dspyerrors.Fields{"session_id": sessionID, "branch_id": branch.ID})
	}
	return branch, nil
}

func (s *SQLiteStore) lookupBranchHeadTx(ctx context.Context, tx *sql.Tx, sessionID, branchID string) (string, error) {
	var headID sql.NullString
	err := tx.QueryRowContext(ctx,
		`SELECT head_entry_id
		   FROM session_branches
		  WHERE session_id = ? AND id = ?`,
		sessionID,
		branchID,
	).Scan(&headID)
	if err == sql.ErrNoRows {
		return "", notFound("branch not found", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	if err != nil {
		return "", wrapSQLError(err, "failed to lookup branch head", dspyerrors.Fields{"session_id": sessionID, "branch_id": branchID})
	}
	if !headID.Valid {
		return "", nil
	}
	return strings.TrimSpace(headID.String), nil
}

func (s *SQLiteStore) entryExistsTx(ctx context.Context, tx *sql.Tx, sessionID, entryID string) (bool, error) {
	var exists int
	if err := tx.QueryRowContext(ctx,
		`SELECT 1
		   FROM session_entries
		  WHERE session_id = ? AND id = ?`,
		sessionID,
		entryID,
	).Scan(&exists); err == sql.ErrNoRows {
		return false, nil
	} else if err != nil {
		return false, wrapSQLError(err, "failed to validate parent entry", dspyerrors.Fields{"session_id": sessionID, "entry_id": entryID})
	}
	return true, nil
}

func (s *SQLiteStore) entryExists(ctx context.Context, sessionID, entryID string) (bool, error) {
	var exists int
	if err := s.db.QueryRowContext(ctx,
		`SELECT 1
		   FROM session_entries
		  WHERE session_id = ? AND id = ?`,
		sessionID,
		entryID,
	).Scan(&exists); err == sql.ErrNoRows {
		return false, nil
	} else if err != nil {
		return false, wrapSQLError(err, "failed to validate entry", dspyerrors.Fields{"session_id": sessionID, "entry_id": entryID})
	}
	return true, nil
}

func scanSession(scanner interface{ Scan(dest ...any) error }) (*Session, error) {
	var (
		session      Session
		createdAtRaw string
		updatedAtRaw string
		metadataJSON string
	)
	if err := scanner.Scan(
		&session.ID,
		&session.Title,
		&session.Status,
		&createdAtRaw,
		&updatedAtRaw,
		&session.ActiveBranchID,
		&metadataJSON,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, notFound("session not found", nil)
		}
		return nil, wrapSQLError(err, "failed to scan session", nil)
	}
	var err error
	session.CreatedAt, err = parseTime(createdAtRaw)
	if err != nil {
		return nil, err
	}
	session.UpdatedAt, err = parseTime(updatedAtRaw)
	if err != nil {
		return nil, err
	}
	session.Metadata, err = decodeJSONMap(metadataJSON)
	if err != nil {
		return nil, err
	}
	return &session, nil
}

func scanBranch(scanner interface{ Scan(dest ...any) error }) (*SessionBranch, error) {
	var (
		branch        SessionBranch
		createdAtRaw  string
		updatedAtRaw  string
		originEntryID sql.NullString
		headEntryID   sql.NullString
		metadataJSON  string
	)
	if err := scanner.Scan(
		&branch.ID,
		&branch.SessionID,
		&branch.Name,
		&branch.Status,
		&createdAtRaw,
		&updatedAtRaw,
		&originEntryID,
		&headEntryID,
		&metadataJSON,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, notFound("branch not found", nil)
		}
		return nil, wrapSQLError(err, "failed to scan session branch", nil)
	}
	var err error
	branch.CreatedAt, err = parseTime(createdAtRaw)
	if err != nil {
		return nil, err
	}
	branch.UpdatedAt, err = parseTime(updatedAtRaw)
	if err != nil {
		return nil, err
	}
	branch.OriginEntryID = nullString(originEntryID)
	branch.HeadEntryID = nullString(headEntryID)
	branch.Metadata, err = decodeJSONMap(metadataJSON)
	if err != nil {
		return nil, err
	}
	return &branch, nil
}

func scanEntry(scanner interface{ Scan(dest ...any) error }) (*SessionEntry, error) {
	var (
		entry        SessionEntry
		parentID     sql.NullString
		createdAtRaw string
		payloadJSON  string
		metadataJSON string
		isError      int
		synthetic    int
		redacted     int
		truncated    int
	)
	if err := scanner.Scan(
		&entry.ID,
		&entry.SessionID,
		&entry.BranchID,
		&parentID,
		&entry.Kind,
		&createdAtRaw,
		&entry.Role,
		&entry.ToolName,
		&isError,
		&synthetic,
		&redacted,
		&truncated,
		&payloadJSON,
		&entry.SearchText,
		&metadataJSON,
		&entry.PromptTokens,
		&entry.CompletionTokens,
		&entry.TotalTokens,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, notFound("entry not found", nil)
		}
		return nil, wrapSQLError(err, "failed to scan session entry", nil)
	}
	var err error
	entry.CreatedAt, err = parseTime(createdAtRaw)
	if err != nil {
		return nil, err
	}
	entry.ParentID = nullString(parentID)
	entry.IsError = intToBool(isError)
	entry.Synthetic = intToBool(synthetic)
	entry.Redacted = intToBool(redacted)
	entry.Truncated = intToBool(truncated)
	entry.Payload, err = decodeJSONMap(payloadJSON)
	if err != nil {
		return nil, err
	}
	entry.Metadata, err = decodeJSONMap(metadataJSON)
	if err != nil {
		return nil, err
	}
	return &entry, nil
}

func scanSummary(scanner interface{ Scan(dest ...any) error }) (*SessionSummary, error) {
	var (
		summary         SessionSummary
		parentSummaryID sql.NullString
		createdAtRaw    string
		metadataJSON    string
	)
	if err := scanner.Scan(
		&summary.ID,
		&summary.SessionID,
		&summary.BranchID,
		&summary.StartEntryID,
		&summary.EndEntryID,
		&parentSummaryID,
		&summary.Kind,
		&createdAtRaw,
		&summary.SummaryText,
		&metadataJSON,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, notFound("summary not found", nil)
		}
		return nil, wrapSQLError(err, "failed to scan session summary", nil)
	}
	var err error
	summary.CreatedAt, err = parseTime(createdAtRaw)
	if err != nil {
		return nil, err
	}
	summary.ParentSummaryID = nullString(parentSummaryID)
	summary.Metadata, err = decodeJSONMap(metadataJSON)
	if err != nil {
		return nil, err
	}
	return &summary, nil
}

func cloneEntry(entry SessionEntry) SessionEntry {
	entry.Payload = core.ShallowCopyMap(entry.Payload)
	entry.Metadata = core.ShallowCopyMap(entry.Metadata)
	return entry
}

func cloneSummary(summary SessionSummary) SessionSummary {
	summary.Metadata = core.ShallowCopyMap(summary.Metadata)
	return summary
}

func parseTime(value string) (time.Time, error) {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidResponse, "failed to parse sqlite timestamp"),
			dspyerrors.Fields{"value": value},
		)
	}
	return parsed.UTC(), nil
}

func formatTime(value time.Time) string {
	return value.UTC().Format("2006-01-02T15:04:05.000000000Z07:00")
}

func decodeJSONMap(value string) (map[string]any, error) {
	if strings.TrimSpace(value) == "" {
		return map[string]any{}, nil
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(value), &decoded); err != nil {
		return nil, dspyerrors.WithFields(
			dspyerrors.Wrap(err, dspyerrors.InvalidResponse, "failed to decode sqlite json map"),
			dspyerrors.Fields{"value": value},
		)
	}
	if decoded == nil {
		return map[string]any{}, nil
	}
	return decoded, nil
}

func encodeJSONMap(value map[string]any) (string, error) {
	if len(value) == 0 {
		return "{}", nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func intToBool(value int) bool {
	return value != 0
}

func nullIfEmpty(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func nullString(value sql.NullString) string {
	if !value.Valid {
		return ""
	}
	return value.String
}

func sqliteDSN(path string) string {
	if path == ":memory:" {
		return "file::memory:?cache=shared&_foreign_keys=on&_busy_timeout=5000&_txlock=immediate"
	}
	return path + "?_foreign_keys=on&_busy_timeout=5000&_journal_mode=WAL&_txlock=immediate"
}

func rollbackTx(ctx context.Context, tx *sql.Tx) {
	if tx == nil {
		return
	}
	if err := tx.Rollback(); err != nil && err != sql.ErrTxDone {
		logging.GetLogger().Error(ctx, "failed to rollback session event transaction: %v", err)
	}
}

func wrapSQLError(err error, message string, fields dspyerrors.Fields) error {
	if err == nil {
		return nil
	}
	if fields == nil {
		fields = dspyerrors.Fields{}
	}
	return dspyerrors.WithFields(dspyerrors.Wrap(err, dspyerrors.Unknown, message), fields)
}

func notFound(message string, fields dspyerrors.Fields) error {
	if fields == nil {
		fields = dspyerrors.Fields{}
	}
	return dspyerrors.WithFields(dspyerrors.New(dspyerrors.ResourceNotFound, message), fields)
}

const selectEntrySQL = `
SELECT
	id, session_id, branch_id, parent_id, kind, created_at, role, tool_name,
	is_error, synthetic, redacted, truncated, payload_json, search_text,
	metadata_json, prompt_tokens, completion_tokens, total_tokens
FROM session_entries
`

const sessionSchema = `
CREATE TABLE IF NOT EXISTS sessions (
	id TEXT PRIMARY KEY,
	title TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'active',
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL,
	active_branch_id TEXT,
	metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_branches (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	name TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'active',
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL,
	origin_entry_id TEXT,
	head_entry_id TEXT,
	metadata_json TEXT NOT NULL DEFAULT '{}',
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_entries (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	branch_id TEXT NOT NULL,
	parent_id TEXT,
	kind TEXT NOT NULL,
	created_at TEXT NOT NULL,
	role TEXT NOT NULL DEFAULT '',
	tool_name TEXT NOT NULL DEFAULT '',
	is_error INTEGER NOT NULL DEFAULT 0,
	synthetic INTEGER NOT NULL DEFAULT 0,
	redacted INTEGER NOT NULL DEFAULT 0,
	truncated INTEGER NOT NULL DEFAULT 0,
	payload_json TEXT NOT NULL DEFAULT '{}',
	search_text TEXT NOT NULL DEFAULT '',
	metadata_json TEXT NOT NULL DEFAULT '{}',
	prompt_tokens INTEGER NOT NULL DEFAULT 0,
	completion_tokens INTEGER NOT NULL DEFAULT 0,
	total_tokens INTEGER NOT NULL DEFAULT 0,
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
	FOREIGN KEY(branch_id) REFERENCES session_branches(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_summaries (
	id TEXT PRIMARY KEY,
	session_id TEXT NOT NULL,
	branch_id TEXT NOT NULL,
	start_entry_id TEXT NOT NULL,
	end_entry_id TEXT NOT NULL,
	parent_summary_id TEXT,
	kind TEXT NOT NULL,
	created_at TEXT NOT NULL,
	summary_text TEXT NOT NULL,
	metadata_json TEXT NOT NULL DEFAULT '{}',
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
	FOREIGN KEY(branch_id) REFERENCES session_branches(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_branches_session_updated
ON session_branches(session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_session_entries_session_branch_created
ON session_entries(session_id, branch_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_session_entries_parent
ON session_entries(parent_id);

CREATE INDEX IF NOT EXISTS idx_session_summaries_session_branch_created
ON session_summaries(session_id, branch_id, created_at DESC);
`
