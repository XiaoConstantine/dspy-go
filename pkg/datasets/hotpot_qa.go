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

type HotPotQAExample struct {
	ID              string  `json:"_id"`
	SupportingFacts [][]any `json:"supporting_facts"`
	Context         [][]any `json:"context"`
	Question        string  `json:"question"`
	Answer          string  `json:"answer"`
	Type            string  `json:"type"`
	Level           string  `json:"level"`
}

// LoadHotpotQA loads the HotpotQA fullwiki validation split. Each
// SupportingFacts entry is [title, sentence index] and each Context entry is
// [title, sentences], the layout of the original HotpotQA JSON release.
func LoadHotpotQA() ([]HotPotQAExample, error) {
	datasetPath, err := EnsureDataset("hotpotqa")
	if err != nil {
		return nil, err
	}
	return loadHotpotQAFile(datasetPath)
}

func loadHotpotQAFile(datasetPath string) ([]HotPotQAExample, error) {
	reader, err := file.OpenParquetFile(datasetPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to open parquet file %s: %w", datasetPath, err)
	}
	defer reader.Close()

	arrowReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
	if err != nil {
		return nil, fmt.Errorf("failed to create arrow reader: %w", err)
	}
	table, err := arrowReader.ReadTable(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to read parquet table: %w", err)
	}
	defer table.Release()

	column := func(name string) (*arrow.Column, error) {
		indices := table.Schema().FieldIndices(name)
		if len(indices) == 0 {
			return nil, fmt.Errorf("required column %q not found in HotpotQA dataset", name)
		}
		return table.Column(indices[0]), nil
	}

	examples := make([]HotPotQAExample, table.NumRows())
	for name, set := range map[string]func(*HotPotQAExample, string){
		"id":       func(ex *HotPotQAExample, v string) { ex.ID = v },
		"question": func(ex *HotPotQAExample, v string) { ex.Question = v },
		"answer":   func(ex *HotPotQAExample, v string) { ex.Answer = v },
		"type":     func(ex *HotPotQAExample, v string) { ex.Type = v },
		"level":    func(ex *HotPotQAExample, v string) { ex.Level = v },
	} {
		col, err := column(name)
		if err != nil {
			return nil, err
		}
		values, err := stringColumnValues(col)
		if err != nil {
			return nil, err
		}
		for i, value := range values {
			set(&examples[i], value)
		}
	}

	for name, set := range map[string]func(*HotPotQAExample, [][]any){
		"supporting_facts": func(ex *HotPotQAExample, v [][]any) { ex.SupportingFacts = v },
		"context":          func(ex *HotPotQAExample, v [][]any) { ex.Context = v },
	} {
		col, err := column(name)
		if err != nil {
			return nil, err
		}
		row := 0
		for _, chunk := range col.Data().Chunks() {
			pairs, err := titledPairs(col.Name(), chunk)
			if err != nil {
				return nil, err
			}
			for _, value := range pairs {
				set(&examples[row], value)
				row++
			}
		}
	}
	return examples, nil
}

// titledPairs converts a struct of two parallel lists, titles and one value
// per title, into [title, value] pairs for each row.
func titledPairs(name string, chunk arrow.Array) ([][][]any, error) {
	structs, ok := chunk.(*array.Struct)
	if !ok || structs.NumField() != 2 {
		return nil, fmt.Errorf("column %q: expected a struct of two lists, got %T", name, chunk)
	}
	titles, ok := structs.Field(0).(*array.List)
	if !ok {
		return nil, fmt.Errorf("column %q: expected a list of titles, got %T", name, structs.Field(0))
	}
	values, ok := structs.Field(1).(*array.List)
	if !ok {
		return nil, fmt.Errorf("column %q: expected a list of values, got %T", name, structs.Field(1))
	}

	rows := make([][][]any, structs.Len())
	for i := range rows {
		rowTitles := listValues(titles, i)
		rowValues := listValues(values, i)
		if len(rowTitles) != len(rowValues) {
			return nil, fmt.Errorf("column %q row %d: %d titles but %d values", name, i, len(rowTitles), len(rowValues))
		}
		rows[i] = make([][]any, len(rowTitles))
		for j := range rowTitles {
			rows[i][j] = []any{rowTitles[j], rowValues[j]}
		}
	}
	return rows, nil
}

// listValues returns the elements of row i of a list array. Strings and
// integers are returned as such, and nested lists as []any.
func listValues(list *array.List, i int) []any {
	start, end := list.ValueOffsets(i)
	out := make([]any, 0, end-start)
	for j := int(start); j < int(end); j++ {
		switch values := list.ListValues().(type) {
		case *array.String:
			out = append(out, values.Value(j))
		case *array.Int32:
			out = append(out, int(values.Value(j)))
		case *array.Int64:
			out = append(out, int(values.Value(j)))
		case *array.List:
			out = append(out, listValues(values, j))
		default:
			out = append(out, values.ValueStr(j))
		}
	}
	return out
}
