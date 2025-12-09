//go:build !windows

package ace

import (
	"os"
	"path/filepath"
	"syscall"
)

// File locking constants for Unix systems.
const (
	lockShared    = syscall.LOCK_SH
	lockExclusive = syscall.LOCK_EX
	lockUnlock    = syscall.LOCK_UN
)

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
