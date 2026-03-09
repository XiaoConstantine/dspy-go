package skills

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrInvalidSkillName    = errors.New("skills: skill name must not be empty")
	ErrInvalidSkillDomain  = errors.New("skills: skill domain must not be empty")
	ErrInvalidSkillContent = errors.New("skills: skill content must not be empty")
	ErrInvalidSkillVersion = errors.New("skills: skill version must be greater than zero")
)

// Store persists versioned skills by domain.
// Best selects the highest published version for a domain, so callers should
// treat version numbers as the canonical ordering of deployable skill quality.
type Store interface {
	Save(ctx context.Context, skill Skill) error
	Load(ctx context.Context, domain string) ([]Skill, error)
	Best(ctx context.Context, domain string) (*Skill, error)
}

// MemoryStore is a process-local skill store used by tests and lightweight workflows.
type MemoryStore struct {
	mu     sync.RWMutex
	skills []Skill
}

// NewMemoryStore returns an empty in-memory skill store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{skills: make([]Skill, 0)}
}

// Save inserts or replaces a skill by (domain, name, version).
func (s *MemoryStore) Save(ctx context.Context, skill Skill) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	normalized, err := normalizeSkill(skill, time.Now().UTC())
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for i := range s.skills {
		if sameSkillVersion(s.skills[i], normalized) {
			normalized.CreatedAt = preserveCreatedAt(s.skills[i], normalized.CreatedAt)
			s.skills[i] = normalized
			return nil
		}
	}

	s.skills = append(s.skills, normalized)
	return nil
}

// Load returns all skills for a domain ordered from newest to oldest version.
func (s *MemoryStore) Load(ctx context.Context, domain string) ([]Skill, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	normalizedDomain := strings.TrimSpace(domain)
	if normalizedDomain == "" {
		return []Skill{}, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return filterAndSortSkills(s.skills, normalizedDomain), nil
}

// Best returns the highest-version published skill for the domain, or nil when none exist.
func (s *MemoryStore) Best(ctx context.Context, domain string) (*Skill, error) {
	skills, err := s.Load(ctx, domain)
	if err != nil {
		return nil, err
	}
	return bestFromLoaded(skills), nil
}

// FileStore persists skills as JSON on disk.
type FileStore struct {
	Path string
	mu   sync.RWMutex
}

// NewFileStore creates a file-backed skill store for the given path.
func NewFileStore(path string) *FileStore {
	return &FileStore{Path: path}
}

// Save inserts or replaces a skill by (domain, name, version) and persists the full set atomically.
func (s *FileStore) Save(ctx context.Context, skill Skill) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	normalized, err := normalizeSkill(skill, time.Now().UTC())
	if err != nil {
		return err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	skills, err := s.readAllLocked()
	if err != nil {
		return err
	}

	replaced := false
	for i := range skills {
		if sameSkillVersion(skills[i], normalized) {
			normalized.CreatedAt = preserveCreatedAt(skills[i], normalized.CreatedAt)
			skills[i] = normalized
			replaced = true
			break
		}
	}
	if !replaced {
		skills = append(skills, normalized)
	}

	return s.writeAllLocked(skills)
}

// Load returns all skills for a domain ordered from newest to oldest version.
func (s *FileStore) Load(ctx context.Context, domain string) ([]Skill, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	normalizedDomain := strings.TrimSpace(domain)
	if normalizedDomain == "" {
		return []Skill{}, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	skills, err := s.readAllLocked()
	if err != nil {
		return nil, err
	}

	return filterAndSortSkills(skills, normalizedDomain), nil
}

// Best returns the highest-version published skill for the domain, or nil when none exist.
func (s *FileStore) Best(ctx context.Context, domain string) (*Skill, error) {
	skills, err := s.Load(ctx, domain)
	if err != nil {
		return nil, err
	}
	return bestFromLoaded(skills), nil
}

func (s *FileStore) readAllLocked() ([]Skill, error) {
	data, err := os.ReadFile(s.Path)
	if os.IsNotExist(err) {
		return []Skill{}, nil
	}
	if err != nil {
		return nil, err
	}

	var skills []Skill
	if err := json.Unmarshal(data, &skills); err != nil {
		return nil, err
	}

	return cloneSkills(skills), nil
}

func (s *FileStore) writeAllLocked(skills []Skill) error {
	if err := os.MkdirAll(filepath.Dir(s.Path), 0755); err != nil {
		return err
	}

	encoded, err := json.MarshalIndent(cloneSkills(skills), "", "  ")
	if err != nil {
		return err
	}

	tmpPath := s.Path + ".tmp"
	if err := os.WriteFile(tmpPath, encoded, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, s.Path); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}

	return nil
}

func sameSkillVersion(left, right Skill) bool {
	return left.Domain == right.Domain && left.Name == right.Name && left.Version == right.Version
}

func preserveCreatedAt(existing Skill, fallback time.Time) time.Time {
	if !existing.CreatedAt.IsZero() {
		return existing.CreatedAt
	}
	return fallback
}

func bestFromLoaded(skills []Skill) *Skill {
	if len(skills) == 0 {
		return nil
	}

	best := skills[0].Clone()
	return &best
}

func filterAndSortSkills(skills []Skill, domain string) []Skill {
	filtered := make([]Skill, 0)
	for _, skill := range skills {
		if skill.Domain != domain {
			continue
		}
		filtered = append(filtered, skill.Clone())
	}

	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].Version != filtered[j].Version {
			return filtered[i].Version > filtered[j].Version
		}
		if !filtered[i].UpdatedAt.Equal(filtered[j].UpdatedAt) {
			return filtered[i].UpdatedAt.After(filtered[j].UpdatedAt)
		}
		return filtered[i].Name < filtered[j].Name
	})

	return filtered
}

func cloneSkills(skills []Skill) []Skill {
	cloned := make([]Skill, len(skills))
	for i := range skills {
		cloned[i] = skills[i].Clone()
	}
	return cloned
}
