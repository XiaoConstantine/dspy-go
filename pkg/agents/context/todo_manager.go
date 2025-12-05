package context

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// TodoManager implements Manus's todo.md attention manipulation pattern.
// This is a brilliant technique: by constantly rewriting a todo.md file,
// the agent keeps its objectives in the recent attention span, avoiding
// "lost-in-the-middle" issues and reducing goal misalignment.
type TodoManager struct {
	mu sync.RWMutex

	memory      *FileSystemMemory
	todos       []TodoItem
	completed   []TodoItem
	lastWrite   time.Time
	updateCount int64

	// Configuration
	config TodoConfig
}

// TodoItem represents a task in the todo list.
type TodoItem struct {
	ID          string                 `json:"id"`
	Description string                 `json:"description"`
	Status      TodoStatus             `json:"status"`
	Created     time.Time              `json:"created"`
	Updated     time.Time              `json:"updated"`
	Priority    int                    `json:"priority"`
	Context     map[string]interface{} `json:"context"`
	Progress    float64                `json:"progress"` // 0.0 to 1.0
}

// TodoStatus represents the current state of a todo item.
type TodoStatus string

const (
	TodoPending    TodoStatus = "pending"
	TodoInProgress TodoStatus = "in_progress"
	TodoCompleted  TodoStatus = "completed"
	TodoFailed     TodoStatus = "failed"
	TodoBlocked    TodoStatus = "blocked"
)

// NewTodoManager creates a manager for the todo.md attention pattern.
func NewTodoManager(memory *FileSystemMemory, config TodoConfig) *TodoManager {
	return &TodoManager{
		memory:    memory,
		todos:     make([]TodoItem, 0),
		completed: make([]TodoItem, 0),
		config:    config,
		lastWrite: time.Now(),
	}
}

// UpdateTodos updates the todo list and triggers attention manipulation.
// CRITICAL: This method constantly rewrites todo.md to keep objectives
// in the model's recent attention span.
func (tm *TodoManager) UpdateTodos(ctx context.Context, todos []TodoItem) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	logger := logging.GetLogger()

	// Update todo items with timestamps
	for i := range todos {
		if todos[i].Created.IsZero() {
			todos[i].Created = time.Now()
		}
		todos[i].Updated = time.Now()

		// Assign ID if not present
		if todos[i].ID == "" {
			todos[i].ID = fmt.Sprintf("todo_%d_%d", i, time.Now().UnixNano())
		}
	}

	tm.todos = todos

	// CRITICAL: Rewrite todo.md file to manipulate attention
	if err := tm.writeTodoFile(ctx); err != nil {
		return fmt.Errorf("failed to write todo file: %w", err)
	}

	tm.updateCount++
	logger.Debug(ctx, "Updated todos and rewrote attention file (%d updates total)", tm.updateCount)

	return nil
}

// AddTodo adds a new todo item and updates the attention file.
func (tm *TodoManager) AddTodo(ctx context.Context, description string, priority int) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	todo := TodoItem{
		ID:          fmt.Sprintf("todo_%d", time.Now().UnixNano()),
		Description: description,
		Status:      TodoPending,
		Created:     time.Now(),
		Updated:     time.Now(),
		Priority:    priority,
		Progress:    0.0,
	}

	tm.todos = append(tm.todos, todo)

	return tm.writeTodoFile(ctx)
}

// SetActive marks a todo as currently active and updates attention.
func (tm *TodoManager) SetActive(ctx context.Context, todoID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// First, ensure only one task is active at a time
	for i := range tm.todos {
		if tm.todos[i].Status == TodoInProgress {
			tm.todos[i].Status = TodoPending
		}
	}

	// Set the specified task as active
	for i := range tm.todos {
		if tm.todos[i].ID == todoID {
			tm.todos[i].Status = TodoInProgress
			tm.todos[i].Updated = time.Now()
			break
		}
	}

	return tm.writeTodoFile(ctx)
}

// CompleteTodo marks a todo as completed and moves it to completed list.
func (tm *TodoManager) CompleteTodo(ctx context.Context, todoID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	for i, todo := range tm.todos {
		if todo.ID == todoID {
			// Mark as completed
			todo.Status = TodoCompleted
			todo.Updated = time.Now()
			todo.Progress = 1.0

			// Move to completed list
			tm.completed = append(tm.completed, todo)

			// Remove from active todos
			tm.todos = append(tm.todos[:i], tm.todos[i+1:]...)

			// Trim completed list if too long
			if len(tm.completed) > tm.config.MaxCompletedTasks {
				tm.completed = tm.completed[len(tm.completed)-tm.config.MaxCompletedTasks:]
			}

			break
		}
	}

	return tm.writeTodoFile(ctx)
}

// UpdateProgress updates the progress of a todo item.
func (tm *TodoManager) UpdateProgress(ctx context.Context, todoID string, progress float64) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	for i := range tm.todos {
		if tm.todos[i].ID == todoID {
			tm.todos[i].Progress = progress
			tm.todos[i].Updated = time.Now()
			break
		}
	}

	// Only rewrite if significant time has passed to avoid too frequent updates
	if time.Since(tm.lastWrite) >= tm.config.UpdateInterval {
		return tm.writeTodoFile(ctx)
	}

	return nil
}

// writeTodoFile creates the todo.md file optimized for LLM attention.
// This is the core of the attention manipulation pattern.
func (tm *TodoManager) writeTodoFile(ctx context.Context) error {
	var content strings.Builder

	// HEADER: Time-stamped for freshness indication
	content.WriteString("# Current Objectives\n\n")
	content.WriteString(fmt.Sprintf("*Last Updated: %s*\n\n", time.Now().Format("15:04:05")))

	// SECTION 1: Active tasks get maximum attention
	activeTasks := tm.getTasksByStatus(TodoInProgress)
	if len(activeTasks) > 0 {
		if tm.config.EnableEmojis {
			content.WriteString("## 🔴 Active Tasks\n\n")
		} else {
			content.WriteString("## Active Tasks\n\n")
		}

		for _, todo := range activeTasks {
			progressBar := tm.formatProgressBar(todo.Progress)
			content.WriteString(fmt.Sprintf("- [ ] **[ACTIVE]** %s %s\n", todo.Description, progressBar))
			if len(todo.Context) > 0 {
				content.WriteString(fmt.Sprintf("  *Context: %v*\n", todo.Context))
			}
		}
		content.WriteString("\n")
	}

	// SECTION 2: High priority pending tasks
	highPriorityTasks := tm.getHighPriorityTasks()
	if len(highPriorityTasks) > 0 {
		if tm.config.EnableEmojis {
			content.WriteString("## 🟠 High Priority Pending\n\n")
		} else {
			content.WriteString("## High Priority Pending\n\n")
		}

		for _, todo := range highPriorityTasks {
			content.WriteString(fmt.Sprintf("- [ ] **[HIGH]** %s\n", todo.Description))
		}
		content.WriteString("\n")
	}

	// SECTION 3: Regular pending tasks
	pendingTasks := tm.getTasksByStatus(TodoPending)
	if len(pendingTasks) > 0 {
		if tm.config.EnableEmojis {
			content.WriteString("## 🟡 Pending Tasks\n\n")
		} else {
			content.WriteString("## Pending Tasks\n\n")
		}

		displayCount := min(len(pendingTasks), tm.config.MaxPendingTasks)
		for i := 0; i < displayCount; i++ {
			todo := pendingTasks[i]
			content.WriteString(fmt.Sprintf("- [ ] %s\n", todo.Description))
		}

		if len(pendingTasks) > displayCount {
			content.WriteString(fmt.Sprintf("- ... and %d more pending tasks\n", len(pendingTasks)-displayCount))
		}
		content.WriteString("\n")
	}

	// SECTION 4: Blocked tasks (need attention)
	blockedTasks := tm.getTasksByStatus(TodoBlocked)
	if len(blockedTasks) > 0 {
		if tm.config.EnableEmojis {
			content.WriteString("## 🚫 Blocked Tasks\n\n")
		} else {
			content.WriteString("## Blocked Tasks\n\n")
		}

		for _, todo := range blockedTasks {
			content.WriteString(fmt.Sprintf("- [ ] **[BLOCKED]** %s\n", todo.Description))
		}
		content.WriteString("\n")
	}

	// SECTION 5: Recently completed (for continuity and motivation)
	if len(tm.completed) > 0 {
		if tm.config.EnableEmojis {
			content.WriteString("## ✅ Recently Completed\n\n")
		} else {
			content.WriteString("## Recently Completed\n\n")
		}

		displayCount := min(len(tm.completed), tm.config.MaxCompletedTasks)
		for i := len(tm.completed) - displayCount; i < len(tm.completed); i++ {
			todo := tm.completed[i]
			content.WriteString(fmt.Sprintf("- [x] ~~%s~~ *(completed %s)*\n",
				todo.Description, todo.Updated.Format("15:04")))
		}
		content.WriteString("\n")
	}

	// FOOTER: Summary statistics for context
	totalTasks := len(tm.todos) + len(tm.completed)
	activeTasks = tm.getTasksByStatus(TodoInProgress)
	content.WriteString("---\n")
	content.WriteString(fmt.Sprintf("**Progress:** %d active, %d pending, %d completed (%d total)\n",
		len(activeTasks), len(tm.getTasksByStatus(TodoPending)), len(tm.completed), totalTasks))

	// Write to filesystem
	todoPath := filepath.Join(tm.memory.baseDir, tm.memory.patterns["todo"])
	if err := os.WriteFile(todoPath, []byte(content.String()), 0644); err != nil {
		return fmt.Errorf("failed to write todo file: %w", err)
	}

	tm.lastWrite = time.Now()

	// Also store as a memory reference for potential retrieval
	_, err := tm.memory.StoreFile(ctx, "todo", "current", []byte(content.String()), map[string]interface{}{
		"update_count": tm.updateCount,
		"last_write":   tm.lastWrite,
	})

	return err
}

// GetTodoContent returns the current todo.md content for inclusion in context.
func (tm *TodoManager) GetTodoContent() string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	todoPath := filepath.Join(tm.memory.baseDir, tm.memory.patterns["todo"])
	content, err := os.ReadFile(todoPath)
	if err != nil {
		return "# No current objectives\n"
	}

	return string(content)
}

// GetActiveTodos returns currently active todo items.
func (tm *TodoManager) GetActiveTodos() []TodoItem {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.getTasksByStatus(TodoInProgress)
}

// GetPendingTodos returns pending todo items.
func (tm *TodoManager) GetPendingTodos() []TodoItem {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.getTasksByStatus(TodoPending)
}

// GetMetrics returns todo management metrics.
func (tm *TodoManager) GetMetrics() map[string]interface{} {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return map[string]interface{}{
		"total_todos":      len(tm.todos),
		"completed_todos":  len(tm.completed),
		"active_todos":     len(tm.getTasksByStatus(TodoInProgress)),
		"pending_todos":    len(tm.getTasksByStatus(TodoPending)),
		"blocked_todos":    len(tm.getTasksByStatus(TodoBlocked)),
		"update_count":     tm.updateCount,
		"last_write":       tm.lastWrite,
	}
}

// Helper methods

func (tm *TodoManager) getTasksByStatus(status TodoStatus) []TodoItem {
	var tasks []TodoItem
	for _, todo := range tm.todos {
		if todo.Status == status {
			tasks = append(tasks, todo)
		}
	}
	return tasks
}

func (tm *TodoManager) getHighPriorityTasks() []TodoItem {
	var tasks []TodoItem
	for _, todo := range tm.todos {
		if todo.Status == TodoPending && todo.Priority >= 8 { // Priority 8-10 considered high
			tasks = append(tasks, todo)
		}
	}
	return tasks
}

func (tm *TodoManager) formatProgressBar(progress float64) string {
	if progress <= 0 {
		return ""
	}

	if progress >= 1.0 {
		return "[████████████] 100%"
	}

	barLength := 12
	filled := int(progress * float64(barLength))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barLength-filled)
	percentage := int(progress * 100)

	return fmt.Sprintf("[%s] %d%%", bar, percentage)
}
