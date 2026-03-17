package datasets

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	tbliteDatasetName  = "NousResearch/openthoughts-tblite"
	tbliteDefaultSplit = "train"
)

var tbliteRowsEndpoint = "https://datasets-server.huggingface.co/rows"

// TBLiteTaskSelection describes a curated benchmark slice. It can reference
// existing HuggingFace tasks by name and/or embed full local task payloads.
type TBLiteTaskSelection struct {
	Label     string       `json:"label,omitempty"`
	Split     string       `json:"split,omitempty"`
	TaskNames []string     `json:"task_names,omitempty"`
	Tasks     []TBLiteTask `json:"tasks,omitempty"`
}

// TBLiteTask represents a single OpenThoughts-TBLite benchmark task.
// It supports both HuggingFace row payloads and local JSON fixtures.
type TBLiteTask struct {
	TaskName        string   `json:"task_name"`
	Instruction     string   `json:"instruction"`
	DockerImage     string   `json:"docker_image"`
	Category        string   `json:"category"`
	Difficulty      string   `json:"difficulty"`
	Tags            []string `json:"tags,omitempty"`
	AgentTimeoutSec int      `json:"agent_timeout_sec"`
	TestTimeoutSec  int      `json:"test_timeout_sec"`
	EnvironmentTar  string   `json:"environment_tar"`
	TestsTar        string   `json:"tests_tar"`
	TestScript      string   `json:"test_sh"`
}

// UnmarshalJSON accepts HuggingFace rows where tags may be a JSON string
// and timeout fields may be numbers or strings.
func (t *TBLiteTask) UnmarshalJSON(data []byte) error {
	type taskAlias TBLiteTask
	var raw struct {
		taskAlias
		Tags            interface{} `json:"tags"`
		AgentTimeoutSec interface{} `json:"agent_timeout_sec"`
		TestTimeoutSec  interface{} `json:"test_timeout_sec"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*t = TBLiteTask(raw.taskAlias)

	tags, err := parseTBLiteTags(raw.Tags)
	if err != nil {
		return fmt.Errorf("parse tags: %w", err)
	}
	t.Tags = tags

	agentTimeout, err := parseTBLiteInt(raw.AgentTimeoutSec)
	if err != nil {
		return fmt.Errorf("parse agent timeout: %w", err)
	}
	testTimeout, err := parseTBLiteInt(raw.TestTimeoutSec)
	if err != nil {
		return fmt.Errorf("parse test timeout: %w", err)
	}
	t.AgentTimeoutSec = agentTimeout
	t.TestTimeoutSec = testTimeout

	return nil
}

// Normalize fills defaults used by the benchmark harness.
func (t TBLiteTask) Normalize() TBLiteTask {
	t.TaskName = strings.TrimSpace(t.TaskName)
	if t.Tags == nil {
		t.Tags = []string{}
	}
	return t
}

// LoadTBLiteTasksFromFile loads TBLite tasks from a local JSON file.
func LoadTBLiteTasksFromFile(path string) ([]TBLiteTask, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var tasks []TBLiteTask
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	for i := range tasks {
		tasks[i] = tasks[i].Normalize()
	}
	if err := validateLocalTBLiteTasks(tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

// LoadTBLiteTaskSelectionFromFile loads a curated TBLite benchmark manifest.
// Supported JSON shapes:
// - ["task-a", "task-b"]
// - [{...full task...}, {...full task...}]
// - {"label":"...", "split":"train", "task_names":[...], "tasks":[...]}.
func LoadTBLiteTaskSelectionFromFile(path string) (*TBLiteTaskSelection, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var taskNames []string
	if err := json.Unmarshal(data, &taskNames); err == nil {
		taskNames = normalizeTaskNames(taskNames)
		if err := validateTBLiteTaskNames(taskNames); err != nil {
			return nil, err
		}
		return &TBLiteTaskSelection{TaskNames: taskNames}, nil
	}

	var tasks []TBLiteTask
	if err := json.Unmarshal(data, &tasks); err == nil {
		for i := range tasks {
			tasks[i] = tasks[i].Normalize()
		}
		if err := validateLocalTBLiteTasks(tasks); err != nil {
			return nil, err
		}
		return &TBLiteTaskSelection{Tasks: tasks}, nil
	}

	var selection TBLiteTaskSelection
	if err := json.Unmarshal(data, &selection); err != nil {
		return nil, fmt.Errorf("unmarshal task selection: %w", err)
	}
	selection.TaskNames = normalizeTaskNames(selection.TaskNames)
	if err := validateTBLiteTaskNames(selection.TaskNames); err != nil {
		return nil, err
	}
	for i := range selection.Tasks {
		selection.Tasks[i] = selection.Tasks[i].Normalize()
	}
	if err := validateLocalTBLiteTasks(selection.Tasks); err != nil {
		return nil, err
	}
	if len(selection.TaskNames) == 0 && len(selection.Tasks) == 0 {
		return nil, fmt.Errorf("task selection must include task_names and/or tasks")
	}
	return &selection, nil
}

// FetchTBLiteTasksFromHuggingFace loads TBLite rows from the public datasets server.
func FetchTBLiteTasksFromHuggingFace(limit int) ([]TBLiteTask, error) {
	return FetchTBLiteTasksFromHuggingFaceContext(context.Background(), limit)
}

// FetchTBLiteTasksFromHuggingFaceContext loads TBLite rows from the public datasets server.
func FetchTBLiteTasksFromHuggingFaceContext(ctx context.Context, limit int) ([]TBLiteTask, error) {
	return FetchTBLiteTasksFromHuggingFaceRangeContext(ctx, tbliteDefaultSplit, 0, limit)
}

// FetchTBLiteTasksFromHuggingFaceRange loads a deterministic slice of TBLite rows.
func FetchTBLiteTasksFromHuggingFaceRange(split string, offset, limit int) ([]TBLiteTask, error) {
	return FetchTBLiteTasksFromHuggingFaceRangeContext(context.Background(), split, offset, limit)
}

// FetchTBLiteTasksFromHuggingFaceRangeContext loads a deterministic slice of TBLite rows.
func FetchTBLiteTasksFromHuggingFaceRangeContext(ctx context.Context, split string, offset, limit int) ([]TBLiteTask, error) {
	if split == "" {
		split = tbliteDefaultSplit
	}

	url := fmt.Sprintf(
		"%s?dataset=%s&config=default&split=%s&offset=%d&length=%d",
		tbliteRowsEndpoint,
		tbliteDatasetName,
		split,
		offset,
		limit,
	)

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build TBLite request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch TBLite rows: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("datasets server returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var hfResp struct {
		Rows []struct {
			Row TBLiteTask `json:"row"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(body, &hfResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	tasks := make([]TBLiteTask, len(hfResp.Rows))
	for i, row := range hfResp.Rows {
		tasks[i] = row.Row.Normalize()
	}
	return tasks, nil
}

// FetchTBLiteTasksByNamesContext resolves named tasks from the datasets server
// while preserving the requested order.
func FetchTBLiteTasksByNamesContext(ctx context.Context, split string, taskNames []string) ([]TBLiteTask, error) {
	orderedNames := normalizeTaskNames(taskNames)
	if len(orderedNames) == 0 {
		return nil, fmt.Errorf("at least one task name is required")
	}

	pending := make(map[string]struct{}, len(orderedNames))
	for _, name := range orderedNames {
		pending[name] = struct{}{}
	}
	found := make(map[string]TBLiteTask, len(orderedNames))

	const pageSize = 100
	for offset := 0; len(pending) > 0; offset += pageSize {
		page, err := FetchTBLiteTasksFromHuggingFaceRangeContext(ctx, split, offset, pageSize)
		if err != nil {
			return nil, err
		}
		if len(page) == 0 {
			break
		}
		for _, task := range page {
			if _, ok := pending[task.TaskName]; ok {
				found[task.TaskName] = task.Normalize()
				delete(pending, task.TaskName)
			}
		}
		if len(page) < pageSize {
			break
		}
	}

	if len(pending) > 0 {
		missing := make([]string, 0, len(pending))
		for _, name := range orderedNames {
			if _, ok := pending[name]; ok {
				missing = append(missing, name)
			}
		}
		return nil, fmt.Errorf("missing requested tblite tasks: %s", strings.Join(missing, ", "))
	}

	resolved := make([]TBLiteTask, 0, len(orderedNames))
	for _, name := range orderedNames {
		resolved = append(resolved, found[name])
	}
	return resolved, nil
}

// SliceTBLiteTasks returns a deterministic slice from a task set.
func SliceTBLiteTasks(tasks []TBLiteTask, offset, limit int) []TBLiteTask {
	if offset < 0 {
		offset = 0
	}
	if offset >= len(tasks) {
		return nil
	}

	remaining := tasks[offset:]
	if limit > 0 && limit < len(remaining) {
		remaining = remaining[:limit]
	}

	result := make([]TBLiteTask, len(remaining))
	copy(result, remaining)
	return result
}

// DecodeEnvironmentArchive decodes the base64 environment tarball payload.
func (t TBLiteTask) DecodeEnvironmentArchive() ([]byte, error) {
	return decodeTBLiteArchive(t.EnvironmentTar)
}

// DecodeTestsArchive decodes the base64 tests tarball payload.
func (t TBLiteTask) DecodeTestsArchive() ([]byte, error) {
	return decodeTBLiteArchive(t.TestsTar)
}

func decodeTBLiteArchive(payload string) ([]byte, error) {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil, nil
	}

	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("decode archive: %w", err)
	}
	return decoded, nil
}

func normalizeTaskNames(taskNames []string) []string {
	if len(taskNames) == 0 {
		return nil
	}

	normalized := make([]string, 0, len(taskNames))
	seen := make(map[string]struct{}, len(taskNames))
	for _, name := range taskNames {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		normalized = append(normalized, name)
	}
	return normalized
}

// ValidateTBLiteTaskName ensures task names are safe to materialize as a
// single directory beneath a benchmark root.
func ValidateTBLiteTaskName(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("task name is required")
	}
	if filepath.IsAbs(name) {
		return fmt.Errorf("task name %q must not be absolute", name)
	}
	if name == "." || name == ".." {
		return fmt.Errorf("task name %q is invalid", name)
	}
	if strings.ContainsRune(name, '/') || strings.ContainsRune(name, '\\') {
		return fmt.Errorf("task name %q must not contain path separators", name)
	}
	return nil
}

func validateLocalTBLiteTasks(tasks []TBLiteTask) error {
	for i := range tasks {
		if err := ValidateTBLiteTaskName(tasks[i].TaskName); err != nil {
			return fmt.Errorf("invalid local tblite task at index %d: %w", i, err)
		}
	}
	return nil
}

func validateTBLiteTaskNames(taskNames []string) error {
	for i, name := range taskNames {
		if err := ValidateTBLiteTaskName(name); err != nil {
			return fmt.Errorf("invalid tblite task name at index %d: %w", i, err)
		}
	}
	return nil
}

func parseTBLiteTags(value interface{}) ([]string, error) {
	switch typed := value.(type) {
	case nil:
		return nil, nil
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return nil, nil
		}

		var tags []string
		if strings.HasPrefix(trimmed, "[") {
			if err := json.Unmarshal([]byte(trimmed), &tags); err != nil {
				return nil, err
			}
			return tags, nil
		}

		return []string{trimmed}, nil
	case []interface{}:
		tags := make([]string, 0, len(typed))
		for _, item := range typed {
			if item == nil {
				continue
			}
			tags = append(tags, fmt.Sprint(item))
		}
		return tags, nil
	case []string:
		tags := make([]string, len(typed))
		copy(tags, typed)
		return tags, nil
	default:
		return []string{fmt.Sprint(typed)}, nil
	}
}

func parseTBLiteInt(value interface{}) (int, error) {
	switch typed := value.(type) {
	case nil:
		return 0, nil
	case float64:
		return int(typed), nil
	case float32:
		return int(typed), nil
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case json.Number:
		parsed, err := typed.Int64()
		if err != nil {
			return 0, err
		}
		return int(parsed), nil
	case string:
		if strings.TrimSpace(typed) == "" {
			return 0, nil
		}
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		if err != nil {
			return 0, err
		}
		return int(parsed), nil
	default:
		return 0, fmt.Errorf("unsupported int value %T", value)
	}
}
