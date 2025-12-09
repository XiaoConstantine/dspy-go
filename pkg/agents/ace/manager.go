package ace

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// Manager coordinates all ACE components for self-improving agents.
// It is safe for concurrent use - each StartTrajectory call returns an
// independent TrajectoryRecorder handle.
type Manager struct {
	config    Config
	file      *LearningsFile
	generator *Generator
	reflector *UnifiedReflector
	curator   *Curator
	quality   *QualityCalculator

	// Async processing
	trajectoryQueue chan *Trajectory
	done            chan struct{}
	wg              sync.WaitGroup

	// Cached learnings for injection
	learningsCache   []Learning
	learningsCacheMu sync.RWMutex
	cacheVersion     int64

	// Metrics
	trajectoriesProcessed atomic.Int64
	insightsExtracted     atomic.Int64
	learningsAdded        atomic.Int64
	learningsPruned       atomic.Int64

	// Curation tracking
	pendingCount atomic.Int64
}

// NewManager creates an ACE manager with the given configuration.
func NewManager(config Config, reflector *UnifiedReflector) (*Manager, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	m := &Manager{
		config:          config,
		file:            NewLearningsFile(config.LearningsPath),
		generator:       NewGenerator(),
		reflector:       reflector,
		curator:         NewCurator(config),
		quality:         NewQualityCalculator(),
		trajectoryQueue: make(chan *Trajectory, 100),
		done:            make(chan struct{}),
	}

	// Load initial learnings
	learnings, err := m.file.Load()
	if err != nil {
		return nil, err
	}
	m.learningsCache = learnings

	// Start async processor if enabled
	if config.AsyncReflection {
		m.wg.Add(1)
		go m.processLoop()
	}

	return m, nil
}

// StartTrajectory begins recording a new execution trajectory and returns a handle.
// The returned TrajectoryRecorder is independent and safe for concurrent use.
// Each concurrent execution should use its own handle.
func (m *Manager) StartTrajectory(agentID, taskType, query string) *TrajectoryRecorder {
	recorder := m.generator.Start(agentID, taskType, query)

	// Inject current learnings
	m.learningsCacheMu.RLock()
	ids := make([]string, len(m.learningsCache))
	for i, l := range m.learningsCache {
		ids[i] = l.ShortCode()
	}
	m.learningsCacheMu.RUnlock()

	recorder.SetInjectedLearnings(ids)
	return recorder
}

// EndTrajectory finalizes a trajectory and queues it for processing.
// The recorder should not be used after calling this method.
// The context is used for synchronous processing; async processing uses a background context.
func (m *Manager) EndTrajectory(ctx context.Context, recorder *TrajectoryRecorder, outcome Outcome) {
	if recorder == nil {
		return
	}

	trajectory := recorder.Current()
	if trajectory == nil {
		return
	}

	quality := m.quality.Calculate(trajectory)
	trajectory = recorder.End(outcome, quality)

	if trajectory == nil {
		return
	}

	if m.config.AsyncReflection {
		select {
		case m.trajectoryQueue <- trajectory:
			m.pendingCount.Add(1)
		default:
			// Queue full, process synchronously with provided context
			m.processTrajectory(ctx, trajectory)
		}
	} else {
		m.processTrajectory(ctx, trajectory)
	}
}

// LearningsContext returns formatted learnings for context injection.
func (m *Manager) LearningsContext() string {
	// Copy slice under lock, then format outside lock to minimize lock duration
	m.learningsCacheMu.RLock()
	learnings := make([]Learning, len(m.learningsCache))
	copy(learnings, m.learningsCache)
	m.learningsCacheMu.RUnlock()

	return FormatForInjection(learnings)
}

// Learnings returns a copy of current learnings.
func (m *Manager) Learnings() []Learning {
	m.learningsCacheMu.RLock()
	defer m.learningsCacheMu.RUnlock()

	result := make([]Learning, len(m.learningsCache))
	copy(result, m.learningsCache)
	return result
}

// Close shuts down the manager and flushes pending work.
func (m *Manager) Close() error {
	if m.config.AsyncReflection {
		close(m.done)
		m.wg.Wait()
	}
	return nil
}

// Metrics returns current performance metrics.
func (m *Manager) Metrics() map[string]int64 {
	return map[string]int64{
		"trajectories_processed": m.trajectoriesProcessed.Load(),
		"insights_extracted":     m.insightsExtracted.Load(),
		"learnings_added":        m.learningsAdded.Load(),
		"learnings_pruned":       m.learningsPruned.Load(),
		"pending_trajectories":   m.pendingCount.Load(),
	}
}

func (m *Manager) processLoop() {
	defer m.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	var batch []*Trajectory

	for {
		select {
		case <-m.done:
			// Flush remaining
			for len(m.trajectoryQueue) > 0 {
				t := <-m.trajectoryQueue
				batch = append(batch, t)
			}
			if len(batch) > 0 {
				m.processBatch(context.Background(), batch)
			}
			return

		case t := <-m.trajectoryQueue:
			batch = append(batch, t)
			if len(batch) >= m.config.CurationFrequency {
				m.processBatch(context.Background(), batch)
				batch = nil
			}

		case <-ticker.C:
			if len(batch) > 0 {
				m.processBatch(context.Background(), batch)
				batch = nil
			}
		}
	}
}

func (m *Manager) processBatch(ctx context.Context, batch []*Trajectory) {
	for _, t := range batch {
		m.processTrajectory(ctx, t)
		m.pendingCount.Add(-1)
	}
}

func (m *Manager) processTrajectory(ctx context.Context, trajectory *Trajectory) {
	m.trajectoriesProcessed.Add(1)

	// Get current learnings for reflection
	m.learningsCacheMu.RLock()
	learnings := make([]Learning, len(m.learningsCache))
	copy(learnings, m.learningsCache)
	m.learningsCacheMu.RUnlock()

	// Always use UnifiedReflector to get feedback correlation
	reflector := m.reflector
	if reflector == nil {
		reflector = NewUnifiedReflector(nil, NewSimpleReflector())
	}

	result, err := reflector.Reflect(ctx, trajectory, learnings)
	if err != nil || result == nil {
		return
	}

	m.insightsExtracted.Add(int64(len(result.SuccessPatterns) + len(result.FailurePatterns)))

	// Curate learnings
	curationResult, err := m.curator.Curate(ctx, m.file, result)
	if err != nil {
		return
	}

	m.learningsAdded.Add(int64(len(curationResult.Added)))
	m.learningsPruned.Add(int64(len(curationResult.Pruned)))

	// Refresh cache
	m.refreshCache()
}

func (m *Manager) refreshCache() {
	learnings, err := m.file.Load()
	if err != nil {
		return
	}

	m.learningsCacheMu.Lock()
	m.learningsCache = learnings
	m.cacheVersion++
	m.learningsCacheMu.Unlock()
}
