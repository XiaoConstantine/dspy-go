package datasets

import (
	"encoding/json"
	jsonv2 "encoding/json/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil/jsonv2test"
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

func TestOolongTaskJSONV2Contract(t *testing.T) {
	jsonv2test.Check(t, func(data []byte) (OolongTask, error) {
		var task OolongTask
		err := jsonv2.Unmarshal(data, &task)
		return task, err
	}, jsonv2test.Contract[OolongTask]{
		Valid:           []byte(`{"id":"expected"}`),
		DuplicateMember: []byte(`{"id":"first","id":"second"}`),
		InvalidUTF8:     jsonv2test.InvalidUTF8(`{"id":"`, `"}`),
		CaseMismatch:    []byte(`{"ID":"wrong"}`),
		UnknownMember:   []byte(`{"id":"expected","unknown":true}`),
		CheckValid: func(t testing.TB, task OolongTask) {
			assert.Equal(t, "expected", task.ID)
		},
		CheckCaseMismatch: func(t testing.TB, task OolongTask) {
			assert.Empty(t, task.ID)
		},
		CheckUnknownMember: func(t testing.TB, task OolongTask) {
			assert.Equal(t, "expected", task.ID)
		},
	})
}

func TestSliceOolongTasks_UsesDeterministicOffset(t *testing.T) {
	tasks := SampleOolongTasks()
	sliced := SliceOolongTasks(tasks, 1, 2)

	require.Len(t, sliced, 2)
	assert.Equal(t, tasks[1].Normalize().ID, sliced[0].Normalize().ID)
	assert.Equal(t, tasks[2].Normalize().ID, sliced[1].Normalize().ID)
}
