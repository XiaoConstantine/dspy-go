package agents

import (
	"encoding/json"
	goerrors "errors"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
)

const defaultSessionStorePrefix = "agent_sessions"

// SessionRecord captures one persisted agent run inside a logical session.
type SessionRecord struct {
	ID          string         `json:"id"`
	SessionID   string         `json:"session_id"`
	AgentID     string         `json:"agent_id,omitempty"`
	AgentType   string         `json:"agent_type,omitempty"`
	TaskID      string         `json:"task_id,omitempty"`
	Task        string         `json:"task"`
	StartedAt   time.Time      `json:"started_at"`
	CompletedAt time.Time      `json:"completed_at"`
	Completed   bool           `json:"completed"`
	FinalAnswer string         `json:"final_answer,omitempty"`
	Error       string         `json:"error,omitempty"`
	Highlights  []string       `json:"highlights,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// Clone returns a deep copy of the session record.
func (r SessionRecord) Clone() SessionRecord {
	return SessionRecord{
		ID:          r.ID,
		SessionID:   r.SessionID,
		AgentID:     r.AgentID,
		AgentType:   r.AgentType,
		TaskID:      r.TaskID,
		Task:        r.Task,
		StartedAt:   r.StartedAt,
		CompletedAt: r.CompletedAt,
		Completed:   r.Completed,
		FinalAnswer: r.FinalAnswer,
		Error:       r.Error,
		Highlights:  append([]string(nil), r.Highlights...),
		Metadata:    core.ShallowCopyMap(r.Metadata),
	}
}

type sessionIndex struct {
	IDs []string `json:"ids"`
}

// SessionStore persists session records on top of the generic Memory interface.
type SessionStore struct {
	memory Memory
	prefix string
}

// NewSessionStore returns a session store using the default key prefix.
func NewSessionStore(memory Memory) *SessionStore {
	return NewSessionStoreWithPrefix(memory, defaultSessionStorePrefix)
}

// NewSessionStoreWithPrefix returns a session store with a custom key prefix.
func NewSessionStoreWithPrefix(memory Memory, prefix string) *SessionStore {
	if memory == nil {
		memory = NewInMemoryStore()
	}
	if strings.TrimSpace(prefix) == "" {
		prefix = defaultSessionStorePrefix
	}
	return &SessionStore{
		memory: memory,
		prefix: prefix,
	}
}

// Append writes a session record and updates the ordered session index.
func (s *SessionStore) Append(record SessionRecord) error {
	if s == nil {
		return fmt.Errorf("session store is nil")
	}
	record = record.Clone()
	record.ID = strings.TrimSpace(record.ID)
	record.SessionID = strings.TrimSpace(record.SessionID)
	record.Task = strings.TrimSpace(record.Task)
	if record.ID == "" {
		return fmt.Errorf("session record id is required")
	}
	if record.SessionID == "" {
		return fmt.Errorf("session record session_id is required")
	}
	if record.Task == "" {
		return fmt.Errorf("session record task is required")
	}

	encodedRecord, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal session record: %w", err)
	}
	if err := s.memory.Store(s.recordKey(record.SessionID, record.ID), string(encodedRecord)); err != nil {
		return err
	}

	index, err := s.loadIndex(record.SessionID)
	if err != nil {
		return err
	}
	if !containsString(index.IDs, record.ID) {
		index.IDs = append(index.IDs, record.ID)
	}

	encodedIndex, err := json.Marshal(index)
	if err != nil {
		return fmt.Errorf("marshal session index: %w", err)
	}
	return s.memory.Store(s.indexKey(record.SessionID), string(encodedIndex))
}

// Recent loads up to limit recent session records in chronological order.
func (s *SessionStore) Recent(sessionID string, limit int) ([]SessionRecord, error) {
	if s == nil {
		return nil, fmt.Errorf("session store is nil")
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" || limit <= 0 {
		return nil, nil
	}

	index, err := s.loadIndex(sessionID)
	if err != nil {
		return nil, err
	}
	if len(index.IDs) == 0 {
		return nil, nil
	}

	start := len(index.IDs) - limit
	if start < 0 {
		start = 0
	}

	records := make([]SessionRecord, 0, len(index.IDs)-start)
	for _, id := range index.IDs[start:] {
		record, err := s.loadRecord(sessionID, id)
		if err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	return records, nil
}

func (s *SessionStore) loadIndex(sessionID string) (sessionIndex, error) {
	encoded, ok, err := s.loadString(s.indexKey(sessionID))
	if err != nil {
		return sessionIndex{}, err
	}
	if !ok || strings.TrimSpace(encoded) == "" {
		return sessionIndex{}, nil
	}

	var index sessionIndex
	if err := json.Unmarshal([]byte(encoded), &index); err != nil {
		return sessionIndex{}, fmt.Errorf("decode session index: %w", err)
	}
	return index, nil
}

func (s *SessionStore) loadRecord(sessionID string, id string) (SessionRecord, error) {
	encoded, ok, err := s.loadString(s.recordKey(sessionID, id))
	if err != nil {
		return SessionRecord{}, err
	}
	if !ok || strings.TrimSpace(encoded) == "" {
		return SessionRecord{}, fmt.Errorf("session record %q not found", id)
	}

	var record SessionRecord
	if err := json.Unmarshal([]byte(encoded), &record); err != nil {
		return SessionRecord{}, fmt.Errorf("decode session record: %w", err)
	}
	return record, nil
}

func (s *SessionStore) loadString(key string) (string, bool, error) {
	value, err := s.memory.Retrieve(key)
	if err != nil {
		var dspyErr *dspyerrors.Error
		if goerrors.As(err, &dspyErr) && dspyErr.Code() == dspyerrors.ResourceNotFound {
			return "", false, nil
		}
		return "", false, err
	}

	switch typed := value.(type) {
	case string:
		return typed, true, nil
	case []byte:
		return string(typed), true, nil
	default:
		return "", false, fmt.Errorf("session store key %q has unsupported value type %T", key, value)
	}
}

func (s *SessionStore) indexKey(sessionID string) string {
	return fmt.Sprintf("%s:%s:index", s.prefix, sessionID)
}

func (s *SessionStore) recordKey(sessionID string, id string) string {
	return fmt.Sprintf("%s:%s:record:%s", s.prefix, sessionID, id)
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
