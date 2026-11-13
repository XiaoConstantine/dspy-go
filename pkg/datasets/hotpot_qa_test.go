//go:build !skip

package datasets

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/apache/arrow/go/v13/arrow"
	"github.com/apache/arrow/go/v13/arrow/array"
	"github.com/apache/arrow/go/v13/arrow/memory"
	"github.com/apache/arrow/go/v13/parquet"
	"github.com/apache/arrow/go/v13/parquet/pqarrow"
)

// serveHotpotQA makes LoadHotpotQA download body into an empty dataset cache.
func serveHotpotQA(t *testing.T, body []byte) {
	t.Helper()
	homeDir := t.TempDir()
	t.Setenv("HOME", homeDir)
	t.Setenv("USERPROFILE", homeDir) // Windows
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	t.Cleanup(server.Close)
	gsm8k, hotpotqa := gsm8kDatasetURL, hotPotQADatasetURL
	t.Cleanup(func() { setTestURLs(gsm8k, hotpotqa) })
	setTestURLs(server.URL, server.URL)
}

// TestLoadHotpotQAReadsHuggingFaceLayout serves a parquet file with the
// column layout of the Hugging Face HotpotQA release and checks that rows come
// back in the [title, value] pair layout callers index into.
func TestLoadHotpotQAReadsHuggingFaceLayout(t *testing.T) {
	stringList := arrow.ListOf(arrow.BinaryTypes.String)
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.BinaryTypes.String},
		{Name: "question", Type: arrow.BinaryTypes.String},
		{Name: "answer", Type: arrow.BinaryTypes.String},
		{Name: "type", Type: arrow.BinaryTypes.String},
		{Name: "level", Type: arrow.BinaryTypes.String},
		{Name: "supporting_facts", Type: arrow.StructOf(
			arrow.Field{Name: "title", Type: stringList},
			arrow.Field{Name: "sent_id", Type: arrow.ListOf(arrow.PrimitiveTypes.Int32)},
		)},
		{Name: "context", Type: arrow.StructOf(
			arrow.Field{Name: "title", Type: stringList},
			arrow.Field{Name: "sentences", Type: arrow.ListOf(stringList)},
		)},
	}, nil)

	builder := array.NewRecordBuilder(memory.DefaultAllocator, schema)
	defer builder.Release()
	type row struct {
		id, question, answer, kind, level string
		factTitles                        []string
		factSentences                     []int32
		contextTitles                     []string
		contextSentences                  [][]string
	}
	rows := []row{
		{"a1", "Who?", "Ada", "bridge", "hard",
			[]string{"Ada", "Babbage"}, []int32{0, 2},
			[]string{"Ada", "Babbage"}, [][]string{{"Ada wrote notes.", "She was a countess."}, {"Babbage built engines."}}},
		{"b2", "Where?", "Paris", "comparison", "easy",
			[]string{"Paris"}, []int32{1},
			[]string{"Paris"}, [][]string{{"Paris is in France."}}},
	}
	for _, r := range rows {
		for i, v := range []string{r.id, r.question, r.answer, r.kind, r.level} {
			builder.Field(i).(*array.StringBuilder).Append(v)
		}
		facts := builder.Field(5).(*array.StructBuilder)
		facts.Append(true)
		titles := facts.FieldBuilder(0).(*array.ListBuilder)
		titles.Append(true)
		titles.ValueBuilder().(*array.StringBuilder).AppendValues(r.factTitles, nil)
		sentIDs := facts.FieldBuilder(1).(*array.ListBuilder)
		sentIDs.Append(true)
		sentIDs.ValueBuilder().(*array.Int32Builder).AppendValues(r.factSentences, nil)

		context := builder.Field(6).(*array.StructBuilder)
		context.Append(true)
		titles = context.FieldBuilder(0).(*array.ListBuilder)
		titles.Append(true)
		titles.ValueBuilder().(*array.StringBuilder).AppendValues(r.contextTitles, nil)
		sentences := context.FieldBuilder(1).(*array.ListBuilder)
		sentences.Append(true)
		inner := sentences.ValueBuilder().(*array.ListBuilder)
		for _, group := range r.contextSentences {
			inner.Append(true)
			inner.ValueBuilder().(*array.StringBuilder).AppendValues(group, nil)
		}
	}
	record := builder.NewRecord()
	defer record.Release()

	var file bytes.Buffer
	writer, err := pqarrow.NewFileWriter(schema, &file, parquet.NewWriterProperties(), pqarrow.DefaultWriterProps())
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Write(record); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	serveHotpotQA(t, file.Bytes())
	got, err := LoadHotpotQA()
	if err != nil {
		t.Fatalf("LoadHotpotQA: %v", err)
	}
	want := []HotPotQAExample{
		{
			ID: "a1", Question: "Who?", Answer: "Ada", Type: "bridge", Level: "hard",
			SupportingFacts: [][]any{{"Ada", 0}, {"Babbage", 2}},
			Context: [][]any{
				{"Ada", []any{"Ada wrote notes.", "She was a countess."}},
				{"Babbage", []any{"Babbage built engines."}},
			},
		},
		{
			ID: "b2", Question: "Where?", Answer: "Paris", Type: "comparison", Level: "easy",
			SupportingFacts: [][]any{{"Paris", 1}},
			Context:         [][]any{{"Paris", []any{"Paris is in France."}}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %#v\nwant %#v", got, want)
	}
}

func TestLoadHotpotQARejectsNonParquet(t *testing.T) {
	serveHotpotQA(t, []byte(`[{"_id":"x"}]`))
	_, err := LoadHotpotQA()
	if err == nil || !strings.Contains(err.Error(), "parquet") {
		t.Fatalf("want a parquet error, got %v", err)
	}
}
