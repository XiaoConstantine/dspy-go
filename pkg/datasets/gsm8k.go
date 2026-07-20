//go:build !skip

package datasets

import (
	"context"
	"fmt"

	"github.com/apache/arrow/go/v13/arrow"
	"github.com/apache/arrow/go/v13/arrow/array"
	"github.com/apache/arrow/go/v13/arrow/memory"
	"github.com/apache/arrow/go/v13/parquet/file"
	"github.com/apache/arrow/go/v13/parquet/pqarrow"
)

type GSM8KExample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func LoadGSM8K() ([]GSM8KExample, error) {
	datasetPath, err := EnsureDataset("gsm8k")
	if err != nil {
		return nil, err
	}
	reader, err := file.OpenParquetFile(datasetPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file %s: %w", datasetPath, err)
	}
	defer reader.Close()

	arrowReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
	if err != nil {
		return nil, fmt.Errorf("failed to create arrow reader: %w", err)
	}

	schema, err := arrowReader.Schema()
	if err != nil {
		return nil, fmt.Errorf("failed to read parquet schema: %w", err)
	}
	questionIndices := schema.FieldIndices("question")
	answerIndices := schema.FieldIndices("answer")
	if len(questionIndices) == 0 || len(answerIndices) == 0 {
		return nil, fmt.Errorf("required columns 'question' and 'answer' not found in schema")
	}

	table, err := arrowReader.ReadTable(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to read parquet table: %w", err)
	}
	defer table.Release()

	questions, err := stringColumnValues(table.Column(questionIndices[0]))
	if err != nil {
		return nil, err
	}
	answers, err := stringColumnValues(table.Column(answerIndices[0]))
	if err != nil {
		return nil, err
	}
	if len(questions) != len(answers) {
		return nil, fmt.Errorf("column length mismatch: %d questions vs %d answers", len(questions), len(answers))
	}

	examples := make([]GSM8KExample, len(questions))
	for i := range questions {
		examples[i] = GSM8KExample{
			Question: questions[i],
			Answer:   answers[i],
		}
	}
	return examples, nil
}

// stringColumnValues flattens a chunked Arrow string column into a []string,
// handling tables whose columns span multiple chunks.
func stringColumnValues(col *arrow.Column) ([]string, error) {
	out := make([]string, 0, col.Len())
	for _, chunk := range col.Data().Chunks() {
		strs, ok := chunk.(*array.String)
		if !ok {
			return nil, fmt.Errorf("column %q: expected string chunk, got %T", col.Name(), chunk)
		}
		for i := 0; i < strs.Len(); i++ {
			out = append(out, strs.Value(i))
		}
	}
	return out, nil
}
