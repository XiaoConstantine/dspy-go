package datasets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOolongTaskNormalize_FillsAlternateFields(t *testing.T) {
	task := OolongTask{
		TaskID:  "task-1",
		Context: "ctx",
	}

	normalized := task.Normalize()
	assert.Equal(t, "task-1", normalized.ID)
	assert.Equal(t, "ctx", normalized.ContextWindowText)
	assert.Equal(t, 3, normalized.ContextLen)
}

func TestCheckOolongAnswer_MatchesStructuredAndNumericFormats(t *testing.T) {
	assert.True(t, CheckOolongAnswer("Paris", "The answer is Paris."))
	assert.True(t, CheckOolongAnswer("750", "Answer: 750"))
	assert.True(t, CheckOolongAnswer("incorrect", "['incorrect']"))
	assert.False(t, CheckOolongAnswer("Paris", "Lyon"))
}

func TestLoadOolongTasksFromFile_NormalizesTasks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tasks.json")
	content := `[{"task_id":"sample","context":"hello","question":"q","answer":"a"}]`
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	tasks, err := LoadOolongTasksFromFile(path)
	require.NoError(t, err)
	require.Len(t, tasks, 1)
	assert.Equal(t, "sample", tasks[0].ID)
	assert.Equal(t, "hello", tasks[0].ContextWindowText)
	assert.Equal(t, 5, tasks[0].ContextLen)
}

func TestOolongTaskUnmarshalJSON_AcceptsNumericID(t *testing.T) {
	var task OolongTask
	err := json.Unmarshal([]byte(`{"id":110010000,"question":"q","answer":"a"}`), &task)
	require.NoError(t, err)
	assert.Equal(t, "110010000", task.ID)
}

func TestSliceOolongTasks_UsesDeterministicOffset(t *testing.T) {
	tasks := SampleOolongTasks()
	sliced := SliceOolongTasks(tasks, 1, 2)

	require.Len(t, sliced, 2)
	assert.Equal(t, tasks[1].Normalize().ID, sliced[0].Normalize().ID)
	assert.Equal(t, tasks[2].Normalize().ID, sliced[1].Normalize().ID)
}
