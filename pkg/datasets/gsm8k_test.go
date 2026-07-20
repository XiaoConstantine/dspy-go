//go:build !skip

package datasets

import (
	"strings"
	"testing"

	"github.com/apache/arrow/go/v13/arrow"
	"github.com/apache/arrow/go/v13/arrow/array"
	"github.com/apache/arrow/go/v13/arrow/memory"
)

// newStringColumn builds an arrow.Column whose data is split across one
// chunk per input slice, mimicking how parquet readers chunk large tables.
func newStringColumn(t *testing.T, name string, chunks ...[]string) *arrow.Column {
	t.Helper()

	mem := memory.NewGoAllocator()
	arrays := make([]arrow.Array, 0, len(chunks))
	for _, values := range chunks {
		b := array.NewStringBuilder(mem)
		b.AppendValues(values, nil)
		arrays = append(arrays, b.NewStringArray())
		b.Release()
	}

	chunked := arrow.NewChunked(arrow.BinaryTypes.String, arrays)
	for _, a := range arrays {
		a.Release()
	}
	field := arrow.Field{Name: name, Type: arrow.BinaryTypes.String}
	col := arrow.NewColumn(field, chunked)
	chunked.Release()
	t.Cleanup(col.Release)
	return col
}

func TestStringColumnValuesMultipleChunks(t *testing.T) {
	// Regression test: values must be flattened across all chunks in row
	// order, not read from Chunk(0) with a global row index.
	col := newStringColumn(t, "question",
		[]string{"q1", "q2", "q3"},
		[]string{"q4", "q5"},
		[]string{"q6"},
	)

	got, err := stringColumnValues(col)
	if err != nil {
		t.Fatalf("stringColumnValues returned error: %v", err)
	}

	want := []string{"q1", "q2", "q3", "q4", "q5", "q6"}
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestStringColumnValuesEmptyColumn(t *testing.T) {
	col := newStringColumn(t, "question")

	got, err := stringColumnValues(col)
	if err != nil {
		t.Fatalf("stringColumnValues returned error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no values, got %v", got)
	}
}

func TestStringColumnValuesNonStringChunk(t *testing.T) {
	mem := memory.NewGoAllocator()
	b := array.NewInt64Builder(mem)
	b.AppendValues([]int64{1, 2, 3}, nil)
	ints := b.NewInt64Array()
	b.Release()

	chunked := arrow.NewChunked(arrow.PrimitiveTypes.Int64, []arrow.Array{ints})
	ints.Release()
	field := arrow.Field{Name: "answer", Type: arrow.PrimitiveTypes.Int64}
	col := arrow.NewColumn(field, chunked)
	chunked.Release()
	t.Cleanup(col.Release)

	_, err := stringColumnValues(col)
	if err == nil {
		t.Fatal("expected error for non-string chunk, got nil")
	}
	if !strings.Contains(err.Error(), "answer") || !strings.Contains(err.Error(), "expected string chunk") {
		t.Errorf("error should name the column and chunk type mismatch, got: %v", err)
	}
}
