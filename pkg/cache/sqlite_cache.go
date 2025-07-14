package cache

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// SQLiteCache implements Cache interface using SQLite as storage.
type SQLiteCache struct {
	db         *sql.DB
	config     CacheConfig
	stats      CacheStats
	mu         sync.RWMutex
	closeChan  chan struct{}
	cleanupWG  sync.WaitGroup
	vacuumWG   sync.WaitGroup
}

// NewSQLiteCache creates a new SQLite-based cache.
func NewSQLiteCache(config CacheConfig) (*SQLiteCache, error) {
	if config.SQLiteConfig.Path == "" {
		config.SQLiteConfig.Path = "dspy_cache.db"
	}

	db, err := sql.Open("sqlite3", config.SQLiteConfig.Path)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite database: %w", err)
	}

	// Set connection pool settings
	if config.SQLiteConfig.MaxConnections > 0 {
		db.SetMaxOpenConns(config.SQLiteConfig.MaxConnections)
	} else {
		db.SetMaxOpenConns(10)
	}
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Hour)

	cache := &SQLiteCache{
		db:        db,
		config:    config,
		closeChan: make(chan struct{}),
	}

	// Initialize database
	if err := cache.initDB(); err != nil {
		db.Close()
		return nil, err
	}

	// Enable WAL mode for better concurrent performance
	if config.SQLiteConfig.EnableWAL {
		if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
			db.Close()
			return nil, fmt.Errorf("failed to enable WAL mode: %w", err)
		}
	}

	// Set other pragmas for performance
	pragmas := []string{
		"PRAGMA synchronous=NORMAL",
		"PRAGMA cache_size=10000",
		"PRAGMA temp_store=MEMORY",
		"PRAGMA mmap_size=268435456", // 256MB
	}
	for _, pragma := range pragmas {
		if _, err := db.Exec(pragma); err != nil {
			// Log warning but don't fail
			log.Printf("Warning: failed to set pragma %s: %v", pragma, err)
		}
	}

	// Start cleanup goroutine
	cache.cleanupWG.Add(1)
	go cache.cleanupRoutine()

	// Start vacuum routine if configured
	if config.SQLiteConfig.VacuumInterval > 0 {
		cache.vacuumWG.Add(1)
		go cache.vacuumRoutine()
	}

	// Load initial stats
	cache.loadStats()

	return cache, nil
}

func (c *SQLiteCache) initDB() error {
	query := `
	CREATE TABLE IF NOT EXISTS cache_entries (
		key TEXT PRIMARY KEY,
		value BLOB NOT NULL,
		expires_at INTEGER,
		created_at INTEGER NOT NULL,
		accessed_at INTEGER NOT NULL,
		size INTEGER NOT NULL
	);

	CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at) WHERE expires_at > 0;
	CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at);
	CREATE INDEX IF NOT EXISTS idx_size ON cache_entries(size);
	`

	_, err := c.db.Exec(query)
	return err
}

func (c *SQLiteCache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	atomic.AddInt64(&c.stats.Misses, 1) // Assume miss, will correct if hit

	query := `
	SELECT value, expires_at FROM cache_entries 
	WHERE key = ? AND (expires_at = 0 OR expires_at > ?)
	`

	var value []byte
	var expiresAt int64
	now := time.Now().UnixNano()

	err := c.db.QueryRowContext(ctx, query, key, now).Scan(&value, &expiresAt)
	if err == sql.ErrNoRows {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("failed to get cache entry: %w", err)
	}

	// Update access time
	updateQuery := `UPDATE cache_entries SET accessed_at = ? WHERE key = ?`
	if _, err := c.db.ExecContext(ctx, updateQuery, now, key); err != nil {
		// Log warning but don't fail the get operation
		log.Printf("Warning: failed to update access time: %v", err)
	}

	// Correct the stats
	atomic.AddInt64(&c.stats.Misses, -1)
	atomic.AddInt64(&c.stats.Hits, 1)
	c.stats.LastAccess = time.Now()

	return value, true, nil
}

func (c *SQLiteCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	now := time.Now()
	var expiresAt int64
	if ttl > 0 {
		expiresAt = now.Add(ttl).UnixNano()
	} else if c.config.DefaultTTL > 0 {
		expiresAt = now.Add(c.config.DefaultTTL).UnixNano()
	}

	size := int64(len(value))

	// Check if key already exists to handle replacement correctly
	var existingSize int64
	existingQuery := `SELECT size FROM cache_entries WHERE key = ?`
	err := c.db.QueryRowContext(ctx, existingQuery, key).Scan(&existingSize)
	exists := err == nil

	// Check size limit if configured
	if c.config.MaxSize > 0 {
		currentSize := atomic.LoadInt64(&c.stats.Size)
		neededSize := size
		if exists {
			neededSize = size - existingSize // Net change in size
		}
		if currentSize+neededSize > c.config.MaxSize {
			// Need to evict entries
			if err := c.evictEntries(ctx, neededSize); err != nil {
				return fmt.Errorf("failed to evict entries: %w", err)
			}
		}
	}

	query := `
	INSERT OR REPLACE INTO cache_entries (key, value, expires_at, created_at, accessed_at, size)
	VALUES (?, ?, ?, ?, ?, ?)
	`

	_, err = c.db.ExecContext(ctx, query, key, value, expiresAt, now.UnixNano(), now.UnixNano(), size)
	if err != nil {
		return fmt.Errorf("failed to set cache entry: %w", err)
	}

	// Update stats
	atomic.AddInt64(&c.stats.Sets, 1)
	if exists {
		atomic.AddInt64(&c.stats.Size, size-existingSize)
	} else {
		atomic.AddInt64(&c.stats.Size, size)
	}
	c.stats.LastAccess = now

	return nil
}

func (c *SQLiteCache) Delete(ctx context.Context, key string) error {
	// Get size before deletion
	var size int64
	sizeQuery := `SELECT size FROM cache_entries WHERE key = ?`
	err := c.db.QueryRowContext(ctx, sizeQuery, key).Scan(&size)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to get entry size: %w", err)
	}

	query := `DELETE FROM cache_entries WHERE key = ?`
	result, err := c.db.ExecContext(ctx, query, key)
	if err != nil {
		return fmt.Errorf("failed to delete cache entry: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected > 0 {
		atomic.AddInt64(&c.stats.Deletes, 1)
		atomic.AddInt64(&c.stats.Size, -size)
	}

	return nil
}

func (c *SQLiteCache) Clear(ctx context.Context) error {
	query := `DELETE FROM cache_entries`
	_, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to clear cache: %w", err)
	}

	// Reset stats
	atomic.StoreInt64(&c.stats.Hits, 0)
	atomic.StoreInt64(&c.stats.Misses, 0)
	atomic.StoreInt64(&c.stats.Sets, 0)
	atomic.StoreInt64(&c.stats.Deletes, 0)
	atomic.StoreInt64(&c.stats.Size, 0)

	// Vacuum to reclaim space
	if _, err := c.db.Exec("VACUUM"); err != nil {
		// Log warning but don't fail
		log.Printf("Warning: failed to vacuum after clear: %v", err)
	}

	return nil
}

func (c *SQLiteCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	stats := CacheStats{
		Hits:       atomic.LoadInt64(&c.stats.Hits),
		Misses:     atomic.LoadInt64(&c.stats.Misses),
		Sets:       atomic.LoadInt64(&c.stats.Sets),
		Deletes:    atomic.LoadInt64(&c.stats.Deletes),
		Size:       atomic.LoadInt64(&c.stats.Size),
		MaxSize:    c.config.MaxSize,
		LastAccess: c.stats.LastAccess,
	}
	return stats
}

func (c *SQLiteCache) Close() error {
	close(c.closeChan)
	c.cleanupWG.Wait()
	c.vacuumWG.Wait()
	return c.db.Close()
}

func (c *SQLiteCache) evictEntries(ctx context.Context, neededSpace int64) error {
	// Simple LRU eviction - remove entries until we have enough space
	for {
		// Get current total size
		currentSize := atomic.LoadInt64(&c.stats.Size)
		if currentSize+neededSpace <= c.config.MaxSize {
			break
		}

		// Find the oldest accessed entry
		var oldestKey string
		var deletedSize int64
		selectQuery := `SELECT key, size FROM cache_entries ORDER BY accessed_at ASC LIMIT 1`
		
		err := c.db.QueryRowContext(ctx, selectQuery).Scan(&oldestKey, &deletedSize)
		if err != nil {
			if err == sql.ErrNoRows {
				// No more entries to evict
				break
			}
			return err
		}

		// Delete the oldest entry
		deleteQuery := `DELETE FROM cache_entries WHERE key = ?`
		result, err := c.db.ExecContext(ctx, deleteQuery, oldestKey)
		if err != nil {
			return err
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected > 0 {
			// Update size atomically instead of recalculating
			atomic.AddInt64(&c.stats.Size, -deletedSize)
		} else {
			// Entry was not found, break to avoid infinite loop
			break
		}
	}

	return nil
}

func (c *SQLiteCache) cleanupRoutine() {
	defer c.cleanupWG.Done()

	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-c.closeChan:
			return
		case <-ticker.C:
			c.cleanupExpired()
		}
	}
}

func (c *SQLiteCache) cleanupExpired() {
	// Get the sum of sizes of expired entries before deleting
	var deletedSize int64
	sumQuery := `SELECT COALESCE(SUM(size), 0) FROM cache_entries WHERE expires_at > 0 AND expires_at < ?`
	if err := c.db.QueryRow(sumQuery, time.Now().UnixNano()).Scan(&deletedSize); err != nil {
		log.Printf("Warning: failed to get expired entries size: %v", err)
		return
	}

	if deletedSize == 0 {
		return // No expired entries
	}

	query := `DELETE FROM cache_entries WHERE expires_at > 0 AND expires_at < ?`
	result, err := c.db.Exec(query, time.Now().UnixNano())
	if err != nil {
		log.Printf("Warning: failed to cleanup expired entries: %v", err)
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected > 0 {
		// Update size atomically instead of recalculating
		atomic.AddInt64(&c.stats.Size, -deletedSize)
	}
}

func (c *SQLiteCache) vacuumRoutine() {
	defer c.vacuumWG.Done()

	ticker := time.NewTicker(c.config.SQLiteConfig.VacuumInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.closeChan:
			return
		case <-ticker.C:
			if _, err := c.db.Exec("VACUUM"); err != nil {
				log.Printf("Warning: failed to vacuum database: %v", err)
			}
		}
	}
}

func (c *SQLiteCache) loadStats() {
	var totalSize int64
	query := `SELECT COALESCE(SUM(size), 0) FROM cache_entries`
	if err := c.db.QueryRow(query).Scan(&totalSize); err != nil {
		log.Printf("Warning: failed to load cache size: %v", err)
		return
	}
	atomic.StoreInt64(&c.stats.Size, totalSize)
}

// Export exports cache entries to JSON for backup/migration.
func (c *SQLiteCache) Export(ctx context.Context, writer func(entry CacheEntry) error) error {
	query := `SELECT key, value, expires_at, created_at, accessed_at, size FROM cache_entries`
	rows, err := c.db.QueryContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to query cache entries: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var entry CacheEntry
		var expiresAt, createdAt, accessedAt int64

		err := rows.Scan(&entry.Key, &entry.Value, &expiresAt, &createdAt, &accessedAt, &entry.Size)
		if err != nil {
			return fmt.Errorf("failed to scan row: %w", err)
		}

		if expiresAt > 0 {
			entry.ExpiresAt = time.Unix(0, expiresAt)
		}
		entry.CreatedAt = time.Unix(0, createdAt)
		entry.AccessedAt = time.Unix(0, accessedAt)

		if err := writer(entry); err != nil {
			return err
		}
	}

	return rows.Err()
}

// Import imports cache entries from a source.
func (c *SQLiteCache) Import(ctx context.Context, entries []CacheEntry) error {
	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	
	// Track if transaction was committed successfully
	var committed bool
	defer func() {
		if !committed {
			if rollbackErr := tx.Rollback(); rollbackErr != nil {
				log.Printf("Warning: failed to rollback transaction: %v", rollbackErr)
			}
		}
	}()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT OR REPLACE INTO cache_entries (key, value, expires_at, created_at, accessed_at, size)
		VALUES (?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, entry := range entries {
		var expiresAt int64
		if !entry.ExpiresAt.IsZero() {
			expiresAt = entry.ExpiresAt.UnixNano()
		}

		_, err := stmt.ExecContext(ctx, entry.Key, entry.Value, expiresAt,
			entry.CreatedAt.UnixNano(), entry.AccessedAt.UnixNano(), entry.Size)
		if err != nil {
			return fmt.Errorf("failed to insert entry: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	committed = true

	// Reload stats after import
	c.loadStats()

	return nil
}