package logging

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlightRecorder(t *testing.T) {
	t.Run("NewFlightRecorder with defaults", func(t *testing.T) {
		fr := NewFlightRecorder()
		assert.NotNil(t, fr)
		assert.NotNil(t, fr.recorder)
		assert.Equal(t, 10*time.Second, fr.config.MinAge)
		assert.False(t, fr.running)
	})

	t.Run("NewFlightRecorder with custom min age", func(t *testing.T) {
		fr := NewFlightRecorder(WithMinAge(30 * time.Second))
		assert.Equal(t, 30*time.Second, fr.config.MinAge)
	})

	t.Run("NewFlightRecorder with max bytes", func(t *testing.T) {
		fr := NewFlightRecorder(WithMaxBytes(1024 * 1024))
		assert.Equal(t, uint64(1024*1024), fr.config.MaxBytes)
	})

	t.Run("Start and Stop", func(t *testing.T) {
		fr := NewFlightRecorder(WithMinAge(1 * time.Second))

		err := fr.Start()
		require.NoError(t, err)
		assert.True(t, fr.running)

		// Starting again should be idempotent
		err = fr.Start()
		require.NoError(t, err)

		fr.Stop()
		assert.False(t, fr.running)

		// Stopping again should be idempotent
		fr.Stop()
		assert.False(t, fr.running)
	})

	t.Run("Snapshot creates file", func(t *testing.T) {
		fr := NewFlightRecorder(WithMinAge(1 * time.Second))
		err := fr.Start()
		require.NoError(t, err)
		defer fr.Stop()

		// Let some trace data accumulate
		time.Sleep(10 * time.Millisecond)

		tmpFile := t.TempDir() + "/test.trace"
		err = fr.Snapshot(tmpFile)
		require.NoError(t, err)

		// Verify file was created
		info, err := os.Stat(tmpFile)
		require.NoError(t, err)
		assert.Greater(t, info.Size(), int64(0))
	})

	t.Run("Snapshot when not running does nothing", func(t *testing.T) {
		fr := NewFlightRecorder()
		// Don't start the recorder

		tmpFile := t.TempDir() + "/test.trace"
		err := fr.Snapshot(tmpFile)
		require.NoError(t, err)

		// File should not exist
		_, err = os.Stat(tmpFile)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("SnapshotOnError", func(t *testing.T) {
		fr := NewFlightRecorder(WithMinAge(1 * time.Second))
		err := fr.Start()
		require.NoError(t, err)
		defer fr.Stop()

		time.Sleep(10 * time.Millisecond)

		// Test with error
		tmpFile := t.TempDir() + "/error.trace"
		testErr := errors.New("test error")
		returnedErr := fr.SnapshotOnError(testErr, tmpFile)

		assert.Equal(t, testErr, returnedErr)
		info, err := os.Stat(tmpFile)
		require.NoError(t, err)
		assert.Greater(t, info.Size(), int64(0))

		// Test with nil error
		tmpFile2 := t.TempDir() + "/no_error.trace"
		returnedErr = fr.SnapshotOnError(nil, tmpFile2)

		assert.Nil(t, returnedErr)
		_, err = os.Stat(tmpFile2)
		assert.True(t, os.IsNotExist(err))
	})
}

func TestTraceHelpers(t *testing.T) {
	t.Run("TraceRegion", func(t *testing.T) {
		ctx := context.Background()
		endRegion := TraceRegion(ctx, "TestRegion")
		assert.NotNil(t, endRegion)
		endRegion() // Should not panic
	})

	t.Run("TraceTask", func(t *testing.T) {
		ctx := context.Background()
		newCtx, endTask := TraceTask(ctx, "TestTask")
		assert.NotNil(t, newCtx)
		assert.NotNil(t, endTask)
		endTask() // Should not panic
	})

	t.Run("TraceLog", func(t *testing.T) {
		ctx := context.Background()
		// Should not panic
		TraceLog(ctx, "test", "test message")
	})
}
