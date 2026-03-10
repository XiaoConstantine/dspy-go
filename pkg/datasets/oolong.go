package datasets

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	oolongNonDigitRegexp         = regexp.MustCompile(`[^\d]`)
	oolongStructuredAnswerRegexp = regexp.MustCompile(`^\s*(?:the\s+)?(?:answer|label|result|user)\s*(?:is)?[:=]\s*["']?([^"'\n,]+)["']?\s*$`)
	oolongNumericAnswerRegexp    = regexp.MustCompile(`^\s*(?:answer|result|user)?[:=]?\s*(\d+)\s*$`)
)

// OolongTask represents a single OOLONG benchmark task.
// It supports both the HuggingFace schema and the local example schema.
type OolongTask struct {
	ID                string `json:"id"`
	ContextLen        int    `json:"context_len"`
	Dataset           string `json:"dataset"`
	ContextWindowText string `json:"context_window_text"`
	Question          string `json:"question"`
	TaskGroup         string `json:"task_group"`
	Task              string `json:"task"`
	Answer            string `json:"answer"`
	AnswerType        string `json:"answer_type"`

	TaskID  string `json:"task_id"`
	Context string `json:"context"`
}

// UnmarshalJSON accepts HuggingFace rows where id may be a string or a number.
func (t *OolongTask) UnmarshalJSON(data []byte) error {
	type taskAlias OolongTask
	var raw struct {
		taskAlias
		ID interface{} `json:"id"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*t = OolongTask(raw.taskAlias)
	switch value := raw.ID.(type) {
	case nil:
	case string:
		t.ID = value
	case float64:
		t.ID = strconv.FormatInt(int64(value), 10)
	default:
		t.ID = fmt.Sprint(value)
	}
	return nil
}

// Normalize fills the canonical fields from alternate schema variants.
func (t OolongTask) Normalize() OolongTask {
	if t.ID == "" && t.TaskID != "" {
		t.ID = t.TaskID
	}
	if t.ContextWindowText == "" && t.Context != "" {
		t.ContextWindowText = t.Context
	}
	if t.ContextLen == 0 && t.ContextWindowText != "" {
		t.ContextLen = len(t.ContextWindowText)
	}
	return t
}

// LoadOolongTasksFromFile loads OOLONG tasks from a JSON file.
func LoadOolongTasksFromFile(path string) ([]OolongTask, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var tasks []OolongTask
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	for i := range tasks {
		tasks[i] = tasks[i].Normalize()
	}
	return tasks, nil
}

// SampleOolongTasks returns embedded OOLONG-style tasks for local smoke testing.
func SampleOolongTasks() []OolongTask {
	reviewContext := `
User: alice | Date: 2024-01-15 | Product: Widget A | Rating: 5 | Review: Excellent product, highly recommend!
User: bob | Date: 2024-01-16 | Product: Widget B | Rating: 2 | Review: Poor quality, broke after a week.
User: alice | Date: 2024-01-17 | Product: Widget C | Rating: 4 | Review: Good value for money.
User: charlie | Date: 2024-01-18 | Product: Widget A | Rating: 5 | Review: Best purchase ever!
User: bob | Date: 2024-01-19 | Product: Widget A | Rating: 1 | Review: Terrible, completely unusable.
User: diana | Date: 2024-01-20 | Product: Widget B | Rating: 3 | Review: It's okay, nothing special.
User: alice | Date: 2024-02-01 | Product: Widget A | Rating: 5 | Review: Still works great after weeks!
User: charlie | Date: 2024-02-02 | Product: Widget C | Rating: 4 | Review: Solid choice.
User: eve | Date: 2024-02-03 | Product: Widget A | Rating: 5 | Review: Amazing quality.
User: bob | Date: 2024-02-04 | Product: Widget C | Rating: 2 | Review: Not worth the price.
User: diana | Date: 2024-02-05 | Product: Widget A | Rating: 4 | Review: Pretty good overall.
User: eve | Date: 2024-02-06 | Product: Widget B | Rating: 3 | Review: Average product.
User: charlie | Date: 2024-02-07 | Product: Widget B | Rating: 4 | Review: Better than expected.
User: alice | Date: 2024-02-08 | Product: Widget B | Rating: 5 | Review: Love it!
User: diana | Date: 2024-02-09 | Product: Widget C | Rating: 4 | Review: Would buy again.
`

	longContext := strings.Repeat(reviewContext, 50)

	return []OolongTask{
		{
			ID:                "counting_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "How many reviews are there in total?",
			TaskGroup:         "counting",
			Task:              "count_total",
			Answer:            "750",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "counting_2",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "How many reviews have a rating of 5 stars?",
			TaskGroup:         "counting",
			Task:              "count_by_label",
			Answer:            "250",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "user_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "What is the average rating given by user 'alice'?",
			TaskGroup:         "user",
			Task:              "user_average",
			Answer:            "4.75",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "comparison_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "Which product has the most 5-star reviews: Widget A, Widget B, or Widget C?",
			TaskGroup:         "comparison",
			Task:              "most_frequent",
			Answer:            "Widget A",
			AnswerType:        "LABEL",
		},
		{
			ID:                "temporal_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "Are there more reviews in January or February?",
			TaskGroup:         "temporal",
			Task:              "temporal_comparison",
			Answer:            "February",
			AnswerType:        "LABEL",
		},
	}
}

// FetchOolongTasksFromHuggingFace loads OOLONG validation rows from the public datasets server.
func FetchOolongTasksFromHuggingFace(limit int) ([]OolongTask, error) {
	return FetchOolongTasksFromHuggingFaceRange(0, limit)
}

// FetchOolongTasksFromHuggingFaceRange loads a deterministic slice of OOLONG validation rows.
func FetchOolongTasksFromHuggingFaceRange(offset, limit int) ([]OolongTask, error) {
	url := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=oolongbench/oolong-synth&config=default&split=validation&offset=%d&length=%d", offset, limit)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("fetch OOLONG rows: %w", err)
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
			Row OolongTask `json:"row"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(body, &hfResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	tasks := make([]OolongTask, len(hfResp.Rows))
	for i, row := range hfResp.Rows {
		tasks[i] = row.Row.Normalize()
	}
	return tasks, nil
}

// SliceOolongTasks returns a deterministic slice from a task set.
func SliceOolongTasks(tasks []OolongTask, offset, limit int) []OolongTask {
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

	result := make([]OolongTask, len(remaining))
	copy(result, remaining)
	return result
}

// CheckOolongAnswer applies the same answer-matching logic used by the benchmark examples.
func CheckOolongAnswer(expected, actual string) bool {
	expectedNorm := strings.ToLower(strings.TrimSpace(expected))
	if strings.HasPrefix(expectedNorm, "[") && strings.HasSuffix(expectedNorm, "]") {
		inner := strings.TrimSpace(expectedNorm[1 : len(expectedNorm)-1])
		if (strings.HasPrefix(inner, "'") && strings.HasSuffix(inner, "'")) ||
			(strings.HasPrefix(inner, "\"") && strings.HasSuffix(inner, "\"")) {
			inner = inner[1 : len(inner)-1]
		}
		expectedNorm = inner
	}

	actualNorm := strings.ToLower(strings.TrimSpace(actual))
	responseLen := len(actualNorm)

	if expectedNorm == actualNorm {
		return true
	}

	if responseLen < 50 {
		pattern := `(?:^|[\s'":=-])` + regexp.QuoteMeta(expectedNorm) + `(?:$|[\s'".,;:=-])`
		if matched, _ := regexp.MatchString(pattern, actualNorm); matched {
			return true
		}

		if isNumeric(expectedNorm) {
			cleaned := oolongNonDigitRegexp.ReplaceAllString(actualNorm, "")
			if cleaned == expectedNorm {
				return true
			}
		}

		return false
	}

	lines := strings.Split(strings.TrimSpace(actualNorm), "\n")
	lastLine := ""
	if len(lines) > 0 {
		lastLine = strings.TrimSpace(lines[len(lines)-1])
	}

	if match := oolongStructuredAnswerRegexp.FindStringSubmatch(lastLine); len(match) > 1 {
		extracted := strings.Trim(strings.TrimSpace(match[1]), ".,;:")
		if expectedNorm == extracted {
			return true
		}
	}

	if len(lastLine) < 30 {
		cleaned := strings.Trim(lastLine, ".,;:\"'")
		if expectedNorm == cleaned {
			return true
		}
	}

	if isNumeric(expectedNorm) {
		if match := oolongNumericAnswerRegexp.FindStringSubmatch(lastLine); len(match) > 1 && match[1] == expectedNorm {
			return true
		}
	}

	return false
}

func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}
