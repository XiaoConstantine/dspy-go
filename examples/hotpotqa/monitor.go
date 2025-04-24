package main

import (
	"context"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// GoroutineStats tracks goroutine statistics.
type GoroutineStats struct {
	Current   int32
	Peak      int32
	Started   int64
	Completed int64
}

// GoroutineMonitor tracks goroutine usage.
type GoroutineMonitor struct {
	stats    GoroutineStats
	logger   *logging.Logger
	ctx      context.Context
	cancel   context.CancelFunc
	interval time.Duration
}

// NewGoroutineMonitor creates a new monitor with the given logging interval.
func NewGoroutineMonitor(interval time.Duration, ctx context.Context) *GoroutineMonitor {
	ctx, cancel := context.WithCancel(ctx)
	return &GoroutineMonitor{
		stats:    GoroutineStats{},
		logger:   logging.GetLogger(),
		ctx:      ctx,
		cancel:   cancel,
		interval: interval,
	}
}

// Start begins monitoring goroutine usage.
func (m *GoroutineMonitor) Start() {
	go func() {
		ticker := time.NewTicker(m.interval)
		defer ticker.Stop()

		for {
			select {
			case <-m.ctx.Done():
				return
			case <-ticker.C:
				current := runtime.NumGoroutine()

				// Update peak if current count is higher
				for {
					peak := atomic.LoadInt32(&m.stats.Peak)
					if int32(current) <= peak {
						break
					}
					if atomic.CompareAndSwapInt32(&m.stats.Peak, peak, int32(current)) {
						break
					}
				}

				m.logger.Info(m.ctx, "Goroutine Stats - Current: %d, Peak: %d, Started: %d, Completed: %d",
					current,
					atomic.LoadInt32(&m.stats.Peak),
					atomic.LoadInt64(&m.stats.Started),
					atomic.LoadInt64(&m.stats.Completed))
			}
		}
	}()
}

// Stop terminates the monitoring.
func (m *GoroutineMonitor) Stop() {
	m.cancel()
}

// TrackGoroutine increments the counter when a goroutine starts.
func (m *GoroutineMonitor) TrackGoroutine() {
	atomic.AddInt64(&m.stats.Started, 1)
}

// ReleaseGoroutine decrements the counter when a goroutine completes.
func (m *GoroutineMonitor) ReleaseGoroutine() {
	atomic.AddInt64(&m.stats.Completed, 1)
}

// GetStats returns a copy of the current statistics.
func (m *GoroutineMonitor) GetStats() GoroutineStats {
	return GoroutineStats{
		Current:   int32(runtime.NumGoroutine()),
		Peak:      atomic.LoadInt32(&m.stats.Peak),
		Started:   atomic.LoadInt64(&m.stats.Started),
		Completed: atomic.LoadInt64(&m.stats.Completed),
	}
}
