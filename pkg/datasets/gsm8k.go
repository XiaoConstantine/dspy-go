package datasets

import (
	"fmt"
	"log"
	"os"

	"github.com/apache/arrow/go/parquet/file"
	"github.com/apache/arrow/go/parquet/schema"
)

type GSM8KExample struct {
	Question string
	Answer   string
}

func LoadGSM8K() ([]GSM8KExample, error) {
	datasetPath, err := EnsureDataset("gsm8k")
	if err != nil {
		return nil, err
	}

	f, err := os.Open(datasetPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader, err := file.OpenParquetFile(datasetPath, true)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	defer reader.Close()
	fileMetadata := reader.MetaData()

	fmt.Println("Version:", fileMetadata.Version())
	fmt.Println("Created By:", fileMetadata.GetCreatedBy())
	fmt.Println("Num Rows:", reader.NumRows())

	fmt.Println("Number of RowGroups:", reader.NumRowGroups())
	fmt.Println("Number of Real Columns:", fileMetadata.Schema.Root().NumFields())
	fmt.Println("Number of Columns:", fileMetadata.Schema.NumColumns())
	selectedColumns := []int{}

	if len(selectedColumns) == 0 {
		for i := 0; i < fileMetadata.Schema.NumColumns(); i++ {
			selectedColumns = append(selectedColumns, i)
		}
	} else {
		for _, c := range selectedColumns {
			if c < 0 || c >= fileMetadata.Schema.NumColumns() {
				fmt.Fprintln(os.Stderr, "selected column is out of range")
				os.Exit(1)
			}
		}
	}
	var examples []GSM8KExample
	for r := 0; r < reader.NumRowGroups(); r++ {
		fmt.Println("--- Row Group:", r, " ---")

		rgr := reader.RowGroup(r)
		rowGroupMeta := rgr.MetaData()
		fmt.Println("--- Total Bytes:", rowGroupMeta.TotalByteSize(), " ---")
		fmt.Println("--- Rows:", rgr.NumRows(), " ---")

		fmt.Println("--- Values ---")
		for i := 0; i < int(rgr.NumRows()); i++ {
			example := GSM8KExample{}
			for _, c := range selectedColumns {
				// col := rgr.Column(c)
				colDesc := fileMetadata.Schema.Column(c)
				colName := colDesc.Path()[0] // Assuming the first path component is the name
				fmt.Println(string(colName))

				switch colDesc.ConvertedType() {
				case schema.ConvertedTypes.UTF8:
					// values, err := col.(*encoding.StringDecoder).Decode()
					if err != nil {
						log.Fatalf("Error decoding column %d: %v", c, err)
					}
					if string(colName) == "Question" {
						// example.Question = values[i]
						fmt.Println(colName)
					} else if string(colName) == "Answer" {
						// example.Answer = values[i]
						fmt.Println(colName)

					}
				}
			}
			examples = append(examples, example)
		}
	}
	return examples, nil
}
