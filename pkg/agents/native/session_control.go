package native

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
)

var (
	// ErrSessionEventStoreUnavailable indicates that session branch controls require a configured event store.
	ErrSessionEventStoreUnavailable = errors.New("session event store is not configured")
	// ErrSessionIDRequired indicates that a session control call needs either an explicit session ID or Config.SessionID.
	ErrSessionIDRequired = errors.New("session id is required")
	// ErrSessionBranchIDRequired indicates that a branch switch was requested without a target branch ID.
	ErrSessionBranchIDRequired = errors.New("session branch id is required")
)

// SessionState captures the current persisted state of a session in the event store.
type SessionState struct {
	Session      *sessionevent.Session
	Branches     []sessionevent.SessionBranch
	ActiveBranch *sessionevent.SessionBranch
	HeadEntry    *sessionevent.SessionEntry
}

// GetSessionState returns the active branch and branch list for a stored session.
func (a *Agent) GetSessionState(ctx context.Context, sessionID string) (*SessionState, error) {
	store, resolvedSessionID, err := a.sessionControlStore(sessionID)
	if err != nil {
		return nil, err
	}

	session, err := store.GetSession(ctx, resolvedSessionID)
	if err != nil {
		return nil, err
	}
	branches, err := store.ListBranches(ctx, resolvedSessionID)
	if err != nil {
		return nil, err
	}

	state := &SessionState{
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
		return nil, fmt.Errorf("active branch %q not found for session %q", activeBranchID, resolvedSessionID)
	}

	head, err := store.GetBranchHead(ctx, resolvedSessionID, activeBranchID)
	if err != nil {
		return nil, err
	}
	state.HeadEntry = head
	return state, nil
}

// SwitchSessionBranch updates the active branch for the configured or provided session.
func (a *Agent) SwitchSessionBranch(ctx context.Context, sessionID, branchID string) error {
	store, resolvedSessionID, err := a.sessionControlStore(sessionID)
	if err != nil {
		return err
	}
	if strings.TrimSpace(branchID) == "" {
		return ErrSessionBranchIDRequired
	}

	_, err = switchSessionEventBranch(ctx, store, resolvedSessionID, branchID)
	return err
}

// ForkActiveSession creates a branch from the current active head. When setActive is true,
// the new branch becomes the session's active branch immediately.
func (a *Agent) ForkActiveSession(ctx context.Context, sessionID, name string, setActive bool) (*sessionevent.SessionBranch, error) {
	store, resolvedSessionID, err := a.sessionControlStore(sessionID)
	if err != nil {
		return nil, err
	}

	state, err := a.GetSessionState(ctx, resolvedSessionID)
	if err != nil {
		return nil, err
	}
	if state.HeadEntry == nil {
		return nil, fmt.Errorf("session %q has no active head entry to fork", resolvedSessionID)
	}

	branch, err := store.ForkBranch(ctx, resolvedSessionID, state.HeadEntry.ID, name, nil)
	if err != nil {
		return nil, err
	}
	if setActive {
		if err := store.SetActiveBranch(ctx, resolvedSessionID, branch.ID); err != nil {
			return nil, err
		}
	}
	return branch, nil
}

func (a *Agent) sessionControlStore(sessionID string) (sessionevent.SessionEventStore, string, error) {
	if a == nil {
		return nil, "", fmt.Errorf("tool-calling agent is nil")
	}
	store := a.sessionEventStore()
	if store == nil {
		return nil, "", ErrSessionEventStoreUnavailable
	}

	resolvedSessionID := strings.TrimSpace(sessionID)
	if resolvedSessionID == "" {
		resolvedSessionID = strings.TrimSpace(a.config.SessionID)
	}
	if resolvedSessionID == "" {
		return nil, "", ErrSessionIDRequired
	}
	return store, resolvedSessionID, nil
}

func switchSessionEventBranch(ctx context.Context, store sessionevent.SessionEventStore, sessionID, branchID string) (*sessionevent.SessionEntry, error) {
	head, err := store.GetBranchHead(ctx, sessionID, branchID)
	if err != nil {
		return nil, err
	}
	if err := store.SetActiveBranch(ctx, sessionID, branchID); err != nil {
		return nil, err
	}
	return head, nil
}
