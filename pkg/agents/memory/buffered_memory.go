package memory

import (
	"context"
	"encoding/json"
	goerrors "errors"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// defaultHistoryKey is the default key used to store history in the underlying store.
const defaultHistoryKey = "conversation_log"

// Message represents a single entry in the conversation history.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// BufferedMemory provides an in-memory store that keeps only the last N messages.
// It wraps an underlying agents.Memory implementation.
type BufferedMemory struct {
	store      agents.Memory
	maxSize    int
	historyKey string // Keep internal field, but set to default
	mu         sync.RWMutex
}

// NewBufferedMemory creates a new BufferedMemory instance with a default history key.
// It initializes an InMemoryStore as the underlying storage.
func NewBufferedMemory(maxSize int) *BufferedMemory {
	if maxSize <= 0 {
		maxSize = 1 // Ensure maxSize is at least 1
	}
	// Use the existing NewInMemoryStore for the underlying storage
	underlyingStore := agents.NewInMemoryStore()
	return &BufferedMemory{
		store:      underlyingStore,
		maxSize:    maxSize,
		historyKey: defaultHistoryKey, // Use the default key here
	}
}

// Add appends a new message to the history, ensuring the buffer size limit is maintained.
func (m *BufferedMemory) Add(ctx context.Context, role string, content string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	history, err := m.getHistoryInternal(ctx)
	if err != nil {
		// Check if the error is specifically ResourceNotFound using errors.As
		var dspyErr *errors.Error
		if goerrors.As(err, &dspyErr) && dspyErr.Code() == errors.ResourceNotFound {
			// It's a ResourceNotFound error, initialize history
			history = make([]Message, 0)
		} else {
			// It's some other error, wrap and return it
			return errors.Wrap(err, errors.Unknown, "failed to retrieve history for adding")
		}
	}

	// Append the new message
	history = append(history, Message{Role: role, Content: content})

	// Enforce maxSize limit
	if len(history) > m.maxSize {
		startIndex := len(history) - m.maxSize
		history = history[startIndex:]
	}

	// Marshal and store the updated history
	historyBytes, err := json.Marshal(history)
	if err != nil {
		// Use Unknown or a more appropriate general code if SerializationFailed doesn't exist
		return errors.Wrap(err, errors.Unknown, "failed to marshal history") // Changed to Unknown
	}

	// Use m.historyKey which is now set to the default
	return m.store.Store(m.historyKey, historyBytes)
}

// Get retrieves the conversation history.
func (m *BufferedMemory) Get(ctx context.Context) ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	// Use m.historyKey which is now set to the default
	return m.getHistoryInternal(ctx)
}

// getHistoryInternal retrieves and unmarshals the history from the store.
// This internal version doesn't lock, assuming the caller handles locking.
func (m *BufferedMemory) getHistoryInternal(ctx context.Context) ([]Message, error) {
	// Use m.historyKey which is now set to the default
	value, err := m.store.Retrieve(m.historyKey)
	if err != nil {
		// Check if the error is specifically ResourceNotFound using errors.As
		var dspyErr *errors.Error
		if goerrors.As(err, &dspyErr) && dspyErr.Code() == errors.ResourceNotFound {
			// Key not found means empty history, return empty slice and nil error
			return make([]Message, 0), nil
		} else {
			// It's some other error retrieving from the store
			return nil, err
		}
	}

	// If we got here, the key was found. Proceed with type assertion/unmarshalling
	historyBytes, ok := value.([]byte)
	if !ok {
		// Attempt conversion if it was stored as string initially (less likely with Store)
		if historyString, okStr := value.(string); okStr {
			historyBytes = []byte(historyString)
		} else {
			// Use InvalidResponse or Unknown if TypeAssertionFailed doesn't exist
			return nil, errors.New(errors.InvalidResponse, "stored history is not []byte or string") // Changed to InvalidResponse
		}
	}

	if len(historyBytes) == 0 {
		return make([]Message, 0), nil // Return empty slice if stored value is empty bytes
	}

	var history []Message
	if err := json.Unmarshal(historyBytes, &history); err != nil {
		// Use InvalidResponse or Unknown if DeserializationFailed doesn't exist
		return nil, errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal history") // Changed to InvalidResponse
	}

	return history, nil
}

// Clear removes the conversation history from the store.
func (m *BufferedMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Use m.historyKey which is now set to the default
	emptyHistory := make([]Message, 0)
	historyBytes, _ := json.Marshal(emptyHistory) // Error handling omitted for brevity
	return m.store.Store(m.historyKey, historyBytes)
}
