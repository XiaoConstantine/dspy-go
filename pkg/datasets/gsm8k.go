package datasets

import (
	"context"
	"fmt"
	"log"

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
	// Open the Parquet file
	reader, err := file.OpenParquetFile(datasetPath, false)
	if err != nil {
		log.Fatalf("Error opening Parquet file: %v", err)
	}
	defer reader.Close()
	arrowReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)

	// Get the schema
	schema, _ := arrowReader.Schema()
	fmt.Println(schema)
	// Find question and answer fields
	// Find question and answer field indices
	questionIndices := schema.FieldIndices("question")
	answerIndices := schema.FieldIndices("answer")
	if len(questionIndices) == 0 || len(answerIndices) == 0 {
		log.Fatalf("Required columns 'question' and 'answer' not found in the schema")
	}
	questionIndex := questionIndices[0]
	answerIndex := answerIndices[0]
	fmt.Printf("Question index: %d, Answer index: %d\n", questionIndex, answerIndex)

	// Prepare a slice to hold all examples
	// Read the entire table
	table, err := arrowReader.ReadTable(context.Background())
	if err != nil {
		log.Fatalf("Error reading table: %v", err)
	}
	defer table.Release()

	fmt.Printf("Table number of columns: %d\n", table.NumCols())
	fmt.Printf("Table number of rows: %d\n", table.NumRows())
	// Get question and answer columns
	questionCol := table.Column(questionIndex)
	answerCol := table.Column(answerIndex)

	// Prepare a slice to hold all examples
	examples := make([]GSM8KExample, table.NumRows())

	// Create GSM8KExample structs
	for i := 0; i < int(table.NumRows()); i++ {
		questionChunk := questionCol.Data().Chunk(0)
		answerChunk := answerCol.Data().Chunk(0)

		questionValue := questionChunk.(*array.String).Value(i)
		answerValue := answerChunk.(*array.String).Value(i)
		examples[i] = GSM8KExample{
			Question: questionValue,
			Answer:   answerValue,
		}
	}

	fmt.Printf("Total examples read: %d\n", len(examples))
	return examples, nil
}
