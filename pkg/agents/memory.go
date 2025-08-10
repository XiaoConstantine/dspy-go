package agents

import (
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// Memory provides storage capabilities for agents.
type Memory interface {
	// Store saves a value with a given key
	Store(key string, value interface{}) error

	// Retrieve gets a value by key
	Retrieve(key string) (interface{}, error)

	// Delete removes a value by key
	Delete(key string) error

	// List returns all stored keys
	List() ([]string, error)

	// Clear removes all stored values
	Clear() error
}

// Simple in-memory implementation.
type InMemoryStore struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		data: make(map[string]interface{}),
	}
}

func (s *InMemoryStore) Store(key string, value interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = value
	return nil
}

func (s *InMemoryStore) Retrieve(key string) (interface{}, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	value, exists := s.data[key]
	if !exists {
		return nil, errors.WithFields(
			errors.New(errors.ResourceNotFound, "key not found in memory store"),
			errors.Fields{
				"key":         key,
				"access_time": time.Now().UTC(),
			})
	}
	return value, nil
}

func (s *InMemoryStore) Delete(key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.data[key]; !exists {
		return errors.WithFields(
			errors.New(errors.ResourceNotFound, "key not found in memory store"),
			errors.Fields{
				"key":         key,
				"access_time": time.Now().UTC(),
			})
	}

	delete(s.data, key)
	return nil
}

func (s *InMemoryStore) List() ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}

	return keys, nil
}

func (s *InMemoryStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Create a new map rather than ranging and deleting
	// This is more efficient for clearing everything
	s.data = make(map[string]interface{})
	return nil
}
