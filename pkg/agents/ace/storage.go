package ace

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

// LearningsFile handles plain text storage of learnings.
// It uses a mutex for in-process concurrency and file locking for cross-process safety.
type LearningsFile struct {
	Path string
	mu   sync.Mutex // Protects concurrent access within the same process
}

// NewLearningsFile creates a new file handler for the given path.
func NewLearningsFile(path string) *LearningsFile {
	return &LearningsFile{Path: path}
}

var learningRegex = regexp.MustCompile(`^\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)$`)

// Load reads all learnings from the file.
func (f *LearningsFile) Load() ([]Learning, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	lockFile, err := f.acquireFileLock(syscall.LOCK_SH)
	if err != nil {
		return nil, err
	}
	defer f.releaseFileLock(lockFile)

	data, err := os.ReadFile(f.Path)
	if os.IsNotExist(err) {
		return []Learning{}, nil
	}
	if err != nil {
		return nil, err
	}

	return ParseLearnings(string(data))
}

// Save writes all learnings to the file atomically.
func (f *LearningsFile) Save(learnings []Learning) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	lockFile, err := f.acquireFileLock(syscall.LOCK_EX)
	if err != nil {
		return err
	}
	defer f.releaseFileLock(lockFile)

	if err := os.MkdirAll(filepath.Dir(f.Path), 0755); err != nil {
		return err
	}

	content := FormatLearnings(learnings)
	tmpPath := f.Path + ".tmp"

	if err := os.WriteFile(tmpPath, []byte(content), 0644); err != nil {
		return err
	}

	if err := os.Rename(tmpPath, f.Path); err != nil {
		os.Remove(tmpPath)
		return err
	}

	return nil
}

// LoadContent returns the raw file content for context injection.
func (f *LearningsFile) LoadContent() (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	lockFile, err := f.acquireFileLock(syscall.LOCK_SH)
	if err != nil {
		return "", err
	}
	defer f.releaseFileLock(lockFile)

	data, err := os.ReadFile(f.Path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// Exists returns true if the learnings file exists.
func (f *LearningsFile) Exists() bool {
	_, err := os.Stat(f.Path)
	return err == nil
}

// EstimateTokens returns an estimated token count for the file.
func (f *LearningsFile) EstimateTokens() (int, error) {
	content, err := f.LoadContent()
	if err != nil {
		return 0, err
	}
	return len(content) / 4, nil
}

// acquireFileLock acquires a file lock and returns the lock file handle.
// The caller is responsible for calling releaseFileLock when done.
func (f *LearningsFile) acquireFileLock(lockType int) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(f.Path), 0755); err != nil {
		return nil, err
	}

	lockPath := f.Path + ".lock"
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	if err := syscall.Flock(int(lockFile.Fd()), lockType); err != nil {
		lockFile.Close()
		return nil, err
	}

	return lockFile, nil
}

// releaseFileLock releases a file lock acquired by acquireFileLock.
func (f *LearningsFile) releaseFileLock(lockFile *os.File) {
	if lockFile != nil {
		_ = syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)
		lockFile.Close()
	}
}

// ParseLearnings parses learnings from text content.
func ParseLearnings(content string) ([]Learning, error) {
	var learnings []Learning
	var currentCategory string

	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "## ") {
			currentCategory = strings.ToLower(strings.TrimPrefix(line, "## "))
			continue
		}

		if matches := learningRegex.FindStringSubmatch(line); matches != nil {
			helpful, _ := strconv.Atoi(matches[2])
			harmful, _ := strconv.Atoi(matches[3])

			category := currentCategory
			if category == "" {
				category = extractCategoryFromID(matches[1])
			}

			learnings = append(learnings, Learning{
				ID:       matches[1],
				Category: category,
				Helpful:  helpful,
				Harmful:  harmful,
				Content:  matches[4],
			})
		}
	}

	return learnings, scanner.Err()
}

// FormatLearnings formats learnings as text content grouped by category.
func FormatLearnings(learnings []Learning) string {
	if len(learnings) == 0 {
		return ""
	}

	byCategory := make(map[string][]Learning)
	var categoryOrder []string

	for _, l := range learnings {
		cat := l.Category
		if cat == "" {
			cat = "general"
		}
		if _, exists := byCategory[cat]; !exists {
			categoryOrder = append(categoryOrder, cat)
		}
		byCategory[cat] = append(byCategory[cat], l)
	}

	sort.Strings(categoryOrder)

	var sb strings.Builder
	for i, category := range categoryOrder {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf("## %s\n", strings.ToUpper(category)))

		items := byCategory[category]
		sort.Slice(items, func(i, j int) bool {
			return items[i].ID < items[j].ID
		})

		for _, l := range items {
			sb.WriteString(l.String() + "\n")
		}
	}

	return sb.String()
}

// FormatForInjection formats learnings for context injection with short codes.
func FormatForInjection(learnings []Learning) string {
	if len(learnings) == 0 {
		return ""
	}

	var strategies, mistakes, other []Learning

	for _, l := range learnings {
		switch l.Category {
		case "mistakes":
			mistakes = append(mistakes, l)
		case "strategies", "patterns":
			strategies = append(strategies, l)
		default:
			other = append(other, l)
		}
	}

	var sb strings.Builder

	if len(strategies) > 0 || len(other) > 0 {
		sb.WriteString("## Learned Strategies (cite by ID if using)\n")
		for _, l := range append(strategies, other...) {
			successPct := int(l.SuccessRate() * 100)
			sb.WriteString(fmt.Sprintf("[%s] %s (%d%% success)\n", l.ShortCode(), l.Content, successPct))
		}
		sb.WriteString("\n")
	}

	if len(mistakes) > 0 {
		sb.WriteString("## Mistakes to Avoid (cite by ID if avoiding)\n")
		for _, l := range mistakes {
			sb.WriteString(fmt.Sprintf("[%s] %s\n", l.ShortCode(), l.Content))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// GetNextID returns the next available ID for a category.
func GetNextID(learnings []Learning, category string) string {
	maxNum := 0
	prefix := strings.ToLower(category)

	for _, l := range learnings {
		if strings.HasPrefix(l.ID, prefix+"-") {
			parts := strings.Split(l.ID, "-")
			if len(parts) == 2 {
				if num, err := strconv.Atoi(parts[1]); err == nil && num > maxNum {
					maxNum = num
				}
			}
		}
	}

	return fmt.Sprintf("%s-%05d", prefix, maxNum+1)
}

func extractCategoryFromID(id string) string {
	parts := strings.Split(id, "-")
	if len(parts) >= 1 {
		return parts[0]
	}
	return "general"
}

// FindByID returns the learning with the given ID, or nil if not found.
func FindByID(learnings []Learning, id string) *Learning {
	for i := range learnings {
		if learnings[i].ID == id {
			return &learnings[i]
		}
	}
	return nil
}

// FindByShortCode returns the learning with the given short code, or nil if not found.
func FindByShortCode(learnings []Learning, shortCode string) *Learning {
	for i := range learnings {
		if learnings[i].ShortCode() == shortCode {
			return &learnings[i]
		}
	}
	return nil
}
