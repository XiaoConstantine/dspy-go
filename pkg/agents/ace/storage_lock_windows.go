//go:build windows

package ace

import (
	"os"
)

// File locking constants for Windows (no-op implementation).
// On Windows, cross-process file locking is not supported in this package.
// The mutex provides in-process concurrency safety.
const (
	lockShared    = 0
	lockExclusive = 0
	lockUnlock    = 0
)

// acquireFileLock is a no-op on Windows.
// Cross-process file locking is not supported, but the mutex provides
// in-process concurrency safety which covers most use cases.
func (f *LearningsFile) acquireFileLock(lockType int) (*os.File, error) {
	// No-op on Windows - rely on mutex for in-process safety
	return nil, nil
}

// releaseFileLock is a no-op on Windows.
func (f *LearningsFile) releaseFileLock(lockFile *os.File) {
	// No-op on Windows
}
