package skills

import (
	"maps"
	"strings"
	"time"
)

// Skill is a versioned runtime artifact that can be injected into an agent as a skill pack.
type Skill struct {
	Name      string            `json:"name"`
	Domain    string            `json:"domain"`
	Content   string            `json:"content"`
	Version   int               `json:"version"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// Clone returns a deep copy of the skill so stores do not leak shared mutable state.
func (s Skill) Clone() Skill {
	cloned := s
	cloned.Metadata = maps.Clone(s.Metadata)
	return cloned
}

func normalizeSkill(skill Skill, now time.Time) (Skill, error) {
	normalized := skill.Clone()
	normalized.Name = strings.TrimSpace(normalized.Name)
	normalized.Domain = strings.TrimSpace(normalized.Domain)

	if normalized.Name == "" {
		return Skill{}, ErrInvalidSkillName
	}
	if normalized.Domain == "" {
		return Skill{}, ErrInvalidSkillDomain
	}
	if strings.TrimSpace(normalized.Content) == "" {
		return Skill{}, ErrInvalidSkillContent
	}
	if normalized.Version <= 0 {
		return Skill{}, ErrInvalidSkillVersion
	}

	if normalized.CreatedAt.IsZero() {
		normalized.CreatedAt = now
	}
	normalized.UpdatedAt = now

	return normalized, nil
}
