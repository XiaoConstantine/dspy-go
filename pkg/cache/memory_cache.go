package cache

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MemoryCache implements an in-memory cache with LRU eviction.
type MemoryCache struct {
	config    CacheConfig
	mu        sync.RWMutex
	entries   map[string]*memoryCacheEntry
	lruList   *lruList
	stats     CacheStats
	closeChan chan struct{}
	cleanupWG sync.WaitGroup
}

type memoryCacheEntry struct {
	key       string
	value     []byte
	expiresAt time.Time
	createdAt time.Time
	size      int64
	element   *lruElement
}

// LRU list implementation.
type lruElement struct {
	key  string
	prev *lruElement
	next *lruElement
}

type lruList struct {
	head *lruElement
	tail *lruElement
	size int
}

func newLRUList() *lruList {
	head := &lruElement{}
	tail := &lruElement{}
	head.next = tail
	tail.prev = head
	return &lruList{head: head, tail: tail}
}

func (l *lruList) moveToFront(elem *lruElement) {
	if elem.prev == l.head {
		return // Already at front
	}
	// Remove from current position
	elem.prev.next = elem.next
	elem.next.prev = elem.prev
	// Insert at front
	elem.prev = l.head
	elem.next = l.head.next
	l.head.next.prev = elem
	l.head.next = elem
}

func (l *lruList) pushFront(key string) *lruElement {
	elem := &lruElement{key: key}
	elem.prev = l.head
	elem.next = l.head.next
	l.head.next.prev = elem
	l.head.next = elem
	l.size++
	return elem
}

func (l *lruList) removeElement(elem *lruElement) {
	elem.prev.next = elem.next
	elem.next.prev = elem.prev
	l.size--
}

func (l *lruList) back() *lruElement {
	if l.tail.prev == l.head {
		return nil
	}
	return l.tail.prev
}

// NewMemoryCache creates a new in-memory cache.
func NewMemoryCache(config CacheConfig) (*MemoryCache, error) {
	cache := &MemoryCache{
		config:    config,
		entries:   make(map[string]*memoryCacheEntry),
		lruList:   newLRUList(),
		closeChan: make(chan struct{}),
		stats: CacheStats{
			MaxSize: config.MaxSize,
		},
	}

	// Set default cleanup interval if not specified
	if config.MemoryConfig.CleanupInterval == 0 {
		config.MemoryConfig.CleanupInterval = time.Minute
	}
	
	// Store the config with the corrected cleanup interval
	cache.config = config

	// Start cleanup routine
	cache.cleanupWG.Add(1)
	go cache.cleanupRoutine()

	return cache, nil
}

func (c *MemoryCache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.entries[key]
	if !exists {
		atomic.AddInt64(&c.stats.Misses, 1)
		return nil, false, nil
	}

	// Check if expired
	if entry.expiresAt.After(time.Time{}) && time.Now().After(entry.expiresAt) {
		delete(c.entries, key)
		c.lruList.removeElement(entry.element)
		atomic.AddInt64(&c.stats.Size, -entry.size)
		atomic.AddInt64(&c.stats.Misses, 1)
		return nil, false, nil
	}

	// Move to front of LRU list
	c.lruList.moveToFront(entry.element)

	atomic.AddInt64(&c.stats.Hits, 1)
	c.stats.LastAccess = time.Now() // Safe: protected by c.mu.Lock

	return entry.value, true, nil
}

func (c *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	size := int64(len(value))

	// Check size limit
	if c.config.MaxSize > 0 && size > c.config.MaxSize {
		return fmt.Errorf("value size %d exceeds max cache size %d", size, c.config.MaxSize)
	}

	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = time.Now().Add(ttl)
	} else if c.config.DefaultTTL > 0 {
		expiresAt = time.Now().Add(c.config.DefaultTTL)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if key already exists
	if existing, exists := c.entries[key]; exists {
		// Update existing entry
		atomic.AddInt64(&c.stats.Size, size-existing.size)
		existing.value = value
		existing.size = size
		existing.expiresAt = expiresAt
		c.lruList.moveToFront(existing.element)
	} else {
		// Evict entries if necessary
		currentSize := atomic.LoadInt64(&c.stats.Size)
		if c.config.MaxSize > 0 && currentSize+size > c.config.MaxSize {
			c.evictLRU(size)
		}

		// Add new entry
		element := c.lruList.pushFront(key)
		c.entries[key] = &memoryCacheEntry{
			key:       key,
			value:     value,
			expiresAt: expiresAt,
			createdAt: time.Now(),
			size:      size,
			element:   element,
		}
		atomic.AddInt64(&c.stats.Size, size)
	}

	atomic.AddInt64(&c.stats.Sets, 1)
	c.stats.LastAccess = time.Now() // Safe: protected by c.mu.Lock

	return nil
}

func (c *MemoryCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry, exists := c.entries[key]; exists {
		delete(c.entries, key)
		c.lruList.removeElement(entry.element)
		atomic.AddInt64(&c.stats.Size, -entry.size)
		atomic.AddInt64(&c.stats.Deletes, 1)
	}

	return nil
}

func (c *MemoryCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*memoryCacheEntry)
	c.lruList = newLRUList()

	// Reset stats
	atomic.StoreInt64(&c.stats.Hits, 0)
	atomic.StoreInt64(&c.stats.Misses, 0)
	atomic.StoreInt64(&c.stats.Sets, 0)
	atomic.StoreInt64(&c.stats.Deletes, 0)
	atomic.StoreInt64(&c.stats.Size, 0)

	return nil
}

func (c *MemoryCache) Stats() CacheStats {
	c.mu.RLock()
	lastAccess := c.stats.LastAccess
	c.mu.RUnlock()
	
	return CacheStats{
		Hits:       atomic.LoadInt64(&c.stats.Hits),
		Misses:     atomic.LoadInt64(&c.stats.Misses),
		Sets:       atomic.LoadInt64(&c.stats.Sets),
		Deletes:    atomic.LoadInt64(&c.stats.Deletes),
		Size:       atomic.LoadInt64(&c.stats.Size),
		MaxSize:    c.config.MaxSize,
		LastAccess: lastAccess,
	}
}

func (c *MemoryCache) Close() error {
	close(c.closeChan)
	c.cleanupWG.Wait()
	return nil
}

func (c *MemoryCache) evictLRU(neededSpace int64) {
	// Evict from the back of the LRU list
	currentSize := atomic.LoadInt64(&c.stats.Size)
	targetSize := c.config.MaxSize - neededSpace

	for currentSize > targetSize && c.lruList.size > 0 {
		elem := c.lruList.back()
		if elem == nil {
			break
		}

		if entry, exists := c.entries[elem.key]; exists {
			delete(c.entries, elem.key)
			c.lruList.removeElement(elem)
			currentSize -= entry.size
			atomic.AddInt64(&c.stats.Size, -entry.size)
		}
	}
}

func (c *MemoryCache) cleanupRoutine() {
	defer c.cleanupWG.Done()

	ticker := time.NewTicker(c.config.MemoryConfig.CleanupInterval)
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

func (c *MemoryCache) cleanupExpired() {
	now := time.Now()
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var keysToDelete []string
	for key, entry := range c.entries {
		if entry.expiresAt.After(time.Time{}) && now.After(entry.expiresAt) {
			keysToDelete = append(keysToDelete, key)
		}
	}

	// Delete expired entries - check expiration again in case time passed
	for _, key := range keysToDelete {
		if entry, exists := c.entries[key]; exists {
			// Double-check expiration to avoid race conditions
			if entry.expiresAt.After(time.Time{}) && now.After(entry.expiresAt) {
				delete(c.entries, key)
				c.lruList.removeElement(entry.element)
				atomic.AddInt64(&c.stats.Size, -entry.size)
			}
		}
	}
}

// Export exports cache entries for backup/migration.
func (c *MemoryCache) Export(ctx context.Context, writer func(entry CacheEntry) error) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for key, entry := range c.entries {
		cacheEntry := CacheEntry{
			Key:        key,
			Value:      entry.value,
			ExpiresAt:  entry.expiresAt,
			CreatedAt:  entry.createdAt,
			AccessedAt: time.Now(), // We don't track this in memory
			Size:       entry.size,
		}

		if err := writer(cacheEntry); err != nil {
			return err
		}
	}

	return nil
}

// Import imports cache entries from a source.
func (c *MemoryCache) Import(ctx context.Context, entries []CacheEntry) error {
	for _, entry := range entries {
		// Calculate TTL if entry has expiration
		var ttl time.Duration
		if !entry.ExpiresAt.IsZero() {
			ttl = time.Until(entry.ExpiresAt)
			if ttl <= 0 {
				continue // Skip expired entries
			}
		}

		if err := c.Set(ctx, entry.Key, entry.Value, ttl); err != nil {
			return err
		}
	}

	return nil
}
