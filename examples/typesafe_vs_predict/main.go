package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type supportCategory string

const (
	categoryBilling   supportCategory = "billing"
	categoryTechnical supportCategory = "technical"
	categoryAccount   supportCategory = "account"
	categoryProduct   supportCategory = "product"
)

type labeledTicket struct {
	Text  string
	Label supportCategory
}

var comparisonTickets = []labeledTicket{
	{Text: "I was charged twice for the same subscription renewal.", Label: categoryBilling},
	{Text: "Every API request returns a 503 response.", Label: categoryTechnical},
	{Text: "How can I reset MFA after losing my phone?", Label: categoryAccount},
	{Text: "Where can I export project data as CSV?", Label: categoryProduct},
	{Text: "I need to change the company address shown on our invoices.", Label: categoryBilling},
	{Text: "The settings page crashes whenever I save a webhook.", Label: categoryTechnical},
	{Text: "How do I let a teammate administer our workspace?", Label: categoryAccount},
	{Text: "Can the dashboard filter results by custom date ranges?", Label: categoryProduct},
}

type comparisonItem struct {
	Ticket      labeledTicket
	Decide      supportCategory
	Predict     supportCategory
	DecideTime  time.Duration
	PredictTime time.Duration
}

type comparisonReport struct {
	Items            []comparisonItem
	DecideCorrect    int
	PredictCorrect   int
	SystemOneCalls   int
	SystemOneInput   int64
	SystemOneOutput  int64
	LLM              llmCallStats
	DecideDurations  []time.Duration
	PredictDurations []time.Duration
}

func main() {
	replay := flag.Bool("replay", false, "run offline with recorded System One and LLM responses")
	model := flag.String("model", "", "TypeSafe model override (replay fixtures require jev-replay; live defaults to TYPESAFE_DEFAULT_MODEL or jev-latest)")
	llmModel := flag.String("llm-model", string(core.ModelGoogleGeminiFlash), "generative model used by Predict")
	llmAPIKey := flag.String("llm-api-key", "", "generative provider key (otherwise the provider's environment variable)")
	timeout := flag.Duration("timeout", 60*time.Second, "total example deadline")
	flag.Parse()
	log.SetFlags(0)

	systemOne, closeReplay, err := newComparisonClient(*replay, *model)
	if err != nil {
		log.Fatalf("create System One client: %v", err)
	}
	defer closeReplay()

	var generator core.LLM
	if *replay {
		generator, err = newComparisonReplayLLM()
	} else {
		generator, err = newLiveLLM(*llmAPIKey, core.ModelID(*llmModel))
	}
	if err != nil {
		log.Fatalf("create Predict LLM: %v", err)
	}
	recordedLLM := newRecordingLLM(generator)

	decision, predictor, err := newComparisonModules(systemOne, recordedLLM)
	if err != nil {
		log.Fatalf("create comparison modules: %v", err)
	}

	interruptContext, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(interruptContext, *timeout)
	defer cancel()

	report, err := runComparison(ctx, decision, predictor, recordedLLM)
	if err != nil {
		log.Fatal(err)
	}
	printComparisonReport(report, *replay)
}

func newComparisonModules(client decide.SystemOneClient, llm core.LLM) (*decide.Decide, *modules.Predict, error) {
	// Both modules receive the same signature value. Decide adds its closed-set
	// answer space while Predict reads the allowed labels from the description.
	signature := comparisonSignature()
	decision, err := decide.New(client, signature, decide.Choice[supportCategory]("category",
		decide.Option(categoryBilling, "Charges, invoices, payments, or refunds"),
		decide.Option(categoryTechnical, "Failures, bugs, outages, or unexpected behavior"),
		decide.Option(categoryAccount, "Login, identity, access, permissions, or account settings"),
		decide.Option(categoryProduct, "Product usage, features, exports, or how-to questions"),
	))
	if err != nil {
		return nil, nil, err
	}
	decision.WithName("Jev comparison classifier")
	predictor := modules.NewPredict(signature).WithName("Predict comparison classifier").WithTextOutput()
	predictor.SetLLM(llm)
	return decision, predictor, nil
}

func comparisonSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))}},
		[]core.OutputField{{Field: core.NewField("category", core.WithDescription("Exactly one of: billing, technical, account, product"))}},
	).WithInstruction("Classify into exactly one category. Use billing for charges, invoices, payments, or refunds; technical for failures, bugs, outages, or unexpected behavior; account for login, identity, access, permissions, or account settings; and product for product usage, features, exports, or how-to questions.")
}

func runComparison(ctx context.Context, decision *decide.Decide, predictor *modules.Predict, recordedLLM *recordingLLM) (comparisonReport, error) {
	report := comparisonReport{Items: make([]comparisonItem, 0, len(comparisonTickets))}
	for _, ticket := range comparisonTickets {
		item := comparisonItem{Ticket: ticket}

		started := time.Now()
		result, err := decision.ProcessDecision(ctx, map[string]any{"ticket": ticket.Text})
		item.DecideTime = time.Since(started)
		if err != nil {
			return comparisonReport{}, fmt.Errorf("classify %q with Decide: %w", ticket.Text, err)
		}
		evidence, ok := decide.Get[decide.ChoiceDecision[supportCategory]](result, "category")
		if !ok {
			return comparisonReport{}, fmt.Errorf("classify %q with Decide: category evidence is missing", ticket.Text)
		}
		item.Decide = evidence.Value
		report.SystemOneCalls++
		report.SystemOneInput += result.Usage.InputTokens
		report.SystemOneOutput += result.Usage.OutputTokens
		report.DecideDurations = append(report.DecideDurations, item.DecideTime)
		if item.Decide == ticket.Label {
			report.DecideCorrect++
		}

		started = time.Now()
		outputs, err := predictor.Process(ctx, map[string]any{"ticket": ticket.Text})
		item.PredictTime = time.Since(started)
		if err != nil {
			return comparisonReport{}, fmt.Errorf("classify %q with Predict: %w", ticket.Text, err)
		}
		item.Predict, err = parseCategory(outputs["category"])
		if err != nil {
			return comparisonReport{}, fmt.Errorf("classify %q with Predict: %w", ticket.Text, err)
		}
		report.PredictDurations = append(report.PredictDurations, item.PredictTime)
		if item.Predict == ticket.Label {
			report.PredictCorrect++
		}
		report.Items = append(report.Items, item)
	}
	report.LLM = recordedLLM.Snapshot()
	return report, nil
}

func parseCategory(value any) (supportCategory, error) {
	text, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("category has type %T, want string", value)
	}
	category := supportCategory(strings.ToLower(strings.TrimSpace(text)))
	switch category {
	case categoryBilling, categoryTechnical, categoryAccount, categoryProduct:
		return category, nil
	default:
		return "", fmt.Errorf("unknown category %q", text)
	}
}

func printComparisonReport(report comparisonReport, replay bool) {
	mode := "live"
	if replay {
		mode = "offline replay"
	}
	fmt.Printf("TypeSafe Decide vs modules.Predict (%s)\n", mode)
	for index, item := range report.Items {
		fmt.Printf("%d. label=%-9s Decide=%-9s Predict=%-9s  %s\n",
			index+1, item.Ticket.Label, item.Decide, item.Predict, item.Ticket.Text)
	}

	fmt.Println("\nComparison summary")
	fmt.Printf("  Decide accuracy:  %d/%d (%.1f%%)\n", report.DecideCorrect, len(report.Items), accuracy(report.DecideCorrect, len(report.Items)))
	fmt.Printf("  Predict accuracy: %d/%d (%.1f%%)\n", report.PredictCorrect, len(report.Items), accuracy(report.PredictCorrect, len(report.Items)))
	fmt.Printf("  System One: %d logical calls, %d input + %d output tokens\n", report.SystemOneCalls, report.SystemOneInput, report.SystemOneOutput)
	fmt.Printf("  generative LLM: %d calls, %d prompt + %d completion tokens (usage on %d/%d calls)\n", report.LLM.Calls, report.LLM.Prompt, report.LLM.Completion, report.LLM.UsageCalls, report.LLM.Calls)
	if replay {
		fmt.Println("  latency: not meaningful in replay mode (local fixture server and deterministic LLM)")
	} else {
		fmt.Printf("  Decide latency/call:  p50=%s p95=%s\n",
			percentile(report.DecideDurations, 0.50).Round(time.Millisecond),
			percentile(report.DecideDurations, 0.95).Round(time.Millisecond),
		)
		fmt.Printf("  Predict latency/call: p50=%s p95=%s\n",
			percentile(report.PredictDurations, 0.50).Round(time.Millisecond),
			percentile(report.PredictDurations, 0.95).Round(time.Millisecond),
		)
	}
	fmt.Println("  This tiny harness is illustrative; use a larger held-out labeled set before drawing conclusions.")
}

func accuracy(correct, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(correct) / float64(total) * 100
}
