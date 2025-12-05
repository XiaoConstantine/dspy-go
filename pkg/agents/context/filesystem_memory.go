package context

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// FileSystemMemory implements Manus's filesystem-as-context pattern.
// This provides unlimited external memory by storing large observations,
// contexts, and other data on disk while keeping short references in context.
type FileSystemMemory struct {
	mu sync.RWMutex

	baseDir   string
	sessionID string
	agentID   string

	// File patterns for different memory types
	patterns map[string]string

	// Configuration
	config MemoryConfig

	// Metrics
	totalFiles    int64
	totalSize     int64
	accessCount   int64
	compressionSavings int64
}

// MemoryReference represents a reference to externally stored content.
type MemoryReference struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Path        string                 `json:"path"`
	Size        int64                  `json:"size"`
	Checksum    string                 `json:"checksum"`
	Timestamp   time.Time              `json:"timestamp"`
	Compressed  bool                   `json:"compressed"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// MemoryContent represents content stored in filesystem memory.
type MemoryContent struct {
	Reference MemoryReference `json:"reference"`
	Content   []byte          `json:"content"`
}

// NewFileSystemMemory creates unlimited external memory using the filesystem.
func NewFileSystemMemory(baseDir, sessionID, agentID string, config MemoryConfig) (*FileSystemMemory, error) {
	fullPath := filepath.Join(baseDir, "memory", sessionID, agentID)
	if err := os.MkdirAll(fullPath, 0755); err != nil {
		return nil, errors.Wrap(err, errors.ConfigurationError, "failed to create memory directory")
	}

	// Set default patterns if not provided
	patterns := config.FilePatterns
	if patterns == nil {
		patterns = map[string]string{
			"todo":         "todo.md",
			"context":      "context_%d.json",
			"observations": "observations_%s.txt",
			"errors":       "errors.log",
			"plan":         "plan.md",
			"memory":       "memory_%s.json",
			"cache":        "cache_%s.dat",
		}
	}

	return &FileSystemMemory{
		baseDir:   fullPath,
		sessionID: sessionID,
		agentID:   agentID,
		patterns:  patterns,
		config:    config,
	}, nil
}

// StoreLargeObservation stores large observations externally and returns a short reference.
// This is the core pattern from Manus - instead of including huge content in context,
// store it externally and include only a reference.
func (fsm *FileSystemMemory) StoreLargeObservation(ctx context.Context, id string, content []byte, metadata map[string]interface{}) (string, error) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	logger := logging.GetLogger()

	// Generate filename using pattern
	filename := fmt.Sprintf(fsm.patterns["observations"], id)
	fullPath := filepath.Join(fsm.baseDir, filename)

	// Calculate checksum for integrity verification
	hasher := sha256.New()
	hasher.Write(content)
	checksum := hex.EncodeToString(hasher.Sum(nil))

	// Store metadata separately for quick access
	metaPath := fullPath + ".meta.json"
	reference := MemoryReference{
		ID:        id,
		Type:      "observation",
		Path:      filename,
		Size:      int64(len(content)),
		Checksum:  checksum,
		Timestamp: time.Now(),
		Compressed: false,
		Metadata:  metadata,
	}

	metaBytes, err := json.MarshalIndent(reference, "", "  ")
	if err != nil {
		return "", errors.Wrap(err, errors.InvalidResponse, "failed to marshal metadata")
	}

	if err := os.WriteFile(metaPath, metaBytes, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write metadata")
	}

	// Store content
	if err := os.WriteFile(fullPath, content, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write observation")
	}

	// Update metrics
	fsm.totalFiles++
	fsm.totalSize += int64(len(content))

	logger.Debug(ctx, "Stored large observation %s (%d bytes) as file reference", id, len(content))

	// Return short reference for context inclusion
	return fmt.Sprintf("[Observation stored: file://%s (size: %d bytes, checksum: %s)]",
		filename, len(content), checksum[:8]), nil
}

// StoreContext stores agent context data for later retrieval.
func (fsm *FileSystemMemory) StoreContext(ctx context.Context, contextID string, data map[string]interface{}) (string, error) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	// Generate filename using pattern
	filename := fmt.Sprintf(fsm.patterns["context"], time.Now().Unix())
	fullPath := filepath.Join(fsm.baseDir, filename)

	// Create memory reference
	contentBytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return "", errors.Wrap(err, errors.InvalidResponse, "failed to marshal context")
	}

	hasher := sha256.New()
	hasher.Write(contentBytes)
	checksum := hex.EncodeToString(hasher.Sum(nil))

	reference := MemoryReference{
		ID:        contextID,
		Type:      "context",
		Path:      filename,
		Size:      int64(len(contentBytes)),
		Checksum:  checksum,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"context_id": contextID},
	}

	// Store reference metadata
	metaPath := fullPath + ".meta.json"
	metaBytes, err := json.MarshalIndent(reference, "", "  ")
	if err != nil {
		return "", errors.Wrap(err, errors.InvalidResponse, "failed to marshal reference")
	}

	if err := os.WriteFile(metaPath, metaBytes, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write reference")
	}

	// Store actual content
	if err := os.WriteFile(fullPath, contentBytes, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write context")
	}

	fsm.totalFiles++
	fsm.totalSize += int64(len(contentBytes))

	return fmt.Sprintf("file://%s", filename), nil
}

// RetrieveObservation retrieves a stored observation by its reference.
func (fsm *FileSystemMemory) RetrieveObservation(ctx context.Context, reference string) (*MemoryContent, error) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	// Parse reference to get filename
	if !strings.HasPrefix(reference, "file://") {
		return nil, errors.New(errors.InvalidInput, "invalid file reference format")
	}

	filename := strings.TrimPrefix(reference, "file://")
	fullPath := filepath.Join(fsm.baseDir, filename)

	// Read metadata
	metaPath := fullPath + ".meta.json"
	metaBytes, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, "failed to read metadata")
	}

	var memRef MemoryReference
	if err := json.Unmarshal(metaBytes, &memRef); err != nil {
		return nil, errors.Wrap(err, errors.InvalidResponse, "failed to unmarshal metadata")
	}

	// Read content
	content, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, "failed to read content")
	}

	// Verify integrity
	hasher := sha256.New()
	hasher.Write(content)
	checksum := hex.EncodeToString(hasher.Sum(nil))

	if checksum != memRef.Checksum {
		return nil, errors.New(errors.ValidationFailed, "content checksum mismatch")
	}

	fsm.accessCount++

	return &MemoryContent{
		Reference: memRef,
		Content:   content,
	}, nil
}

// StoreFile stores arbitrary file content with a specific type.
func (fsm *FileSystemMemory) StoreFile(ctx context.Context, contentType, id string, content []byte, metadata map[string]interface{}) (string, error) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	// Get pattern for content type, fallback to generic
	pattern, exists := fsm.patterns[contentType]
	if !exists {
		pattern = fmt.Sprintf("%s_%%s.dat", contentType)
	}

	// Handle patterns with and without format specifiers
	// Some patterns like "todo.md" are singletons without %s placeholder
	var filename string
	if strings.Contains(pattern, "%") {
		filename = fmt.Sprintf(pattern, id)
	} else {
		filename = pattern
	}
	fullPath := filepath.Join(fsm.baseDir, filename)

	// Calculate checksum
	hasher := sha256.New()
	hasher.Write(content)
	checksum := hex.EncodeToString(hasher.Sum(nil))

	// Create reference
	reference := MemoryReference{
		ID:        id,
		Type:      contentType,
		Path:      filename,
		Size:      int64(len(content)),
		Checksum:  checksum,
		Timestamp: time.Now(),
		Metadata:  metadata,
	}

	// Store metadata
	metaPath := fullPath + ".meta.json"
	metaBytes, err := json.MarshalIndent(reference, "", "  ")
	if err != nil {
		return "", errors.Wrap(err, errors.InvalidResponse, "failed to marshal metadata")
	}

	if err := os.WriteFile(metaPath, metaBytes, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write metadata")
	}

	// Store content
	if err := os.WriteFile(fullPath, content, 0644); err != nil {
		return "", errors.Wrap(err, errors.ConfigurationError, "failed to write content")
	}

	fsm.totalFiles++
	fsm.totalSize += int64(len(content))

	return fmt.Sprintf("file://%s", filename), nil
}

// ListFiles returns all stored files of a specific type.
func (fsm *FileSystemMemory) ListFiles(contentType string) ([]MemoryReference, error) {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()

	var references []MemoryReference

	err := filepath.Walk(fsm.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Look for metadata files
		if strings.HasSuffix(path, ".meta.json") {
			metaBytes, err := os.ReadFile(path)
			if err != nil {
				return nil // Skip files we can't read
			}

			var ref MemoryReference
			if err := json.Unmarshal(metaBytes, &ref); err != nil {
				return nil // Skip invalid metadata
			}

			if contentType == "" || ref.Type == contentType {
				references = append(references, ref)
			}
		}

		return nil
	})

	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "failed to walk memory directory")
	}

	return references, nil
}

// CleanExpired removes files older than the retention period.
func (fsm *FileSystemMemory) CleanExpired(ctx context.Context) (int64, error) {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	logger := logging.GetLogger()
	cutoff := time.Now().Add(-fsm.config.RetentionPeriod)
	var cleanedFiles int64

	err := filepath.Walk(fsm.baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if strings.HasSuffix(path, ".meta.json") {
			metaBytes, err := os.ReadFile(path)
			if err != nil {
				return nil
			}

			var ref MemoryReference
			if err := json.Unmarshal(metaBytes, &ref); err != nil {
				return nil
			}

			if ref.Timestamp.Before(cutoff) {
				// Remove both metadata and content files
				contentPath := filepath.Join(fsm.baseDir, ref.Path)

				if err := os.Remove(path); err != nil {
					logger.Warn(ctx, "Failed to remove metadata file %s: %v", path, err)
				}

				if err := os.Remove(contentPath); err != nil {
					logger.Warn(ctx, "Failed to remove content file %s: %v", contentPath, err)
				}

				fsm.totalFiles--
				fsm.totalSize -= ref.Size
				cleanedFiles++

				logger.Debug(ctx, "Cleaned expired file %s (age: %v)", ref.Path, time.Since(ref.Timestamp))
			}
		}

		return nil
	})

	if err != nil {
		return cleanedFiles, errors.Wrap(err, errors.Unknown, "failed to clean expired files")
	}

	logger.Info(ctx, "Cleaned %d expired files from filesystem memory", cleanedFiles)
	return cleanedFiles, nil
}

// GetMetrics returns filesystem memory usage metrics.
func (fsm *FileSystemMemory) GetMetrics() map[string]interface{} {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()

	return map[string]interface{}{
		"total_files":         fsm.totalFiles,
		"total_size_bytes":    fsm.totalSize,
		"access_count":        fsm.accessCount,
		"compression_savings": fsm.compressionSavings,
		"base_directory":      fsm.baseDir,
		"session_id":          fsm.sessionID,
		"agent_id":            fsm.agentID,
	}
}

// GetTotalSize returns the total size of stored content.
func (fsm *FileSystemMemory) GetTotalSize() int64 {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()
	return fsm.totalSize
}

// GetFileCount returns the total number of stored files.
func (fsm *FileSystemMemory) GetFileCount() int64 {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()
	return fsm.totalFiles
}
