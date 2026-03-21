package agents

import (
	"encoding/json"
	goerrors "errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
)

const defaultSessionStorePrefix = "agent_sessions"

var sessionStoreLocks sync.Map

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

type sessionLog struct {
	Records []SessionRecord `json:"records"`
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

// Append writes or replaces a session record inside the ordered session log.
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

	lock := sessionStoreLock(s.sessionKey(record.SessionID))
	lock.Lock()
	defer lock.Unlock()

	log, err := s.loadSessionLog(record.SessionID)
	if err != nil {
		return err
	}
	if idx := indexOfRecord(log.Records, record.ID); idx >= 0 {
		log.Records[idx] = record
	} else {
		log.Records = append(log.Records, record)
	}

	encodedLog, err := json.Marshal(log)
	if err != nil {
		return fmt.Errorf("marshal session log: %w", err)
	}
	return s.memory.Store(s.sessionKey(record.SessionID), string(encodedLog))
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

	log, err := s.loadSessionLog(sessionID)
	if err != nil {
		return nil, err
	}
	if len(log.Records) == 0 {
		return nil, nil
	}

	start := len(log.Records) - limit
	if start < 0 {
		start = 0
	}

	records := make([]SessionRecord, 0, len(log.Records)-start)
	for _, record := range log.Records[start:] {
		records = append(records, record.Clone())
	}
	return records, nil
}

func (s *SessionStore) loadSessionLog(sessionID string) (sessionLog, error) {
	encoded, ok, err := s.loadString(s.sessionKey(sessionID))
	if err != nil {
		return sessionLog{}, err
	}
	if !ok || strings.TrimSpace(encoded) == "" {
		return sessionLog{}, nil
	}

	var log sessionLog
	if err := json.Unmarshal([]byte(encoded), &log); err != nil {
		return sessionLog{}, fmt.Errorf("decode session log: %w", err)
	}
	return log, nil
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

func (s *SessionStore) sessionKey(sessionID string) string {
	return fmt.Sprintf("%s:%s", s.prefix, sessionID)
}

func sessionStoreLock(key string) *sync.Mutex {
	lock, _ := sessionStoreLocks.LoadOrStore(key, &sync.Mutex{})
	return lock.(*sync.Mutex)
}

func indexOfRecord(records []SessionRecord, id string) int {
	for i := range records {
		if records[i].ID == id {
			return i
		}
	}
	return -1
}
