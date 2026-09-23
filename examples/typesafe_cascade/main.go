package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
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

type cascadeTicket struct {
	Name string
	Text string
}

var cascadeTickets = []cascadeTicket{
	{Name: "invoice copy", Text: "Where can I download a copy of last month's invoice?"},
	{Name: "intermittent checkout", Text: "Checkout sometimes freezes after I confirm a card."},
	{Name: "locked account", Text: "I am locked out after replacing my phone."},
	{Name: "unclear export request", Text: "The export option isn't doing what I expected."},
	{Name: "mixed access and charge", Text: "I cannot access the workspace that appears on my latest charge."},
	{Name: "service outage", Text: "Every API request returns a 503 response."},
}

type cascadeItem struct {
	Ticket       cascadeTicket
	Jev          supportCategory
	Confidence   float64
	Escalated    bool
	Predict      supportCategory
	DecisionTime time.Duration
	PredictTime  time.Duration
	EndToEndTime time.Duration
}

type cascadeReport struct {
	Items              []cascadeItem
	Escalated          int
	Agreements         int
	SystemOneCalls     int
	SystemOneInput     int64
	SystemOneOutput    int64
	LLM                llmCallStats
	DecisionDurations  []time.Duration
	PredictDurations   []time.Duration
	DirectDurations    []time.Duration
	EscalatedDurations []time.Duration
}

func main() {
	replay := flag.Bool("replay", false, "run offline with recorded System One and LLM responses")
	model := flag.String("model", "", "TypeSafe model override (replay fixtures require jev-replay; live defaults to TYPESAFE_DEFAULT_MODEL or jev-latest)")
	llmModel := flag.String("llm-model", string(core.ModelGoogleGeminiFlash), "generative model used on escalated tickets")
	llmAPIKey := flag.String("llm-api-key", "", "generative provider key (otherwise the provider's environment variable)")
	confidenceThreshold := flag.Float64("confidence-threshold", 0.75, "escalate Choice provider confidence below this value")
	timeout := flag.Duration("timeout", 45*time.Second, "total example deadline")
	flag.Parse()
	log.SetFlags(0)

	if math.IsNaN(*confidenceThreshold) || math.IsInf(*confidenceThreshold, 0) || *confidenceThreshold < 0 || *confidenceThreshold > 1 {
		log.Fatal("confidence-threshold must be a finite value in [0, 1]")
	}

	systemOne, closeReplay, err := newCascadeClient(*replay, *model)
	if err != nil {
		log.Fatalf("create System One client: %v", err)
	}
	defer closeReplay()

	var generator core.LLM
	if *replay {
		generator, err = newCascadeReplayLLM()
	} else {
		generator, err = newLiveLLM(*llmAPIKey, core.ModelID(*llmModel))
	}
	if err != nil {
		log.Fatalf("create Predict LLM: %v", err)
	}
	recordedLLM := newRecordingLLM(generator)

	decision, predictor, err := newCascadeModules(systemOne, recordedLLM)
	if err != nil {
		log.Fatalf("create cascade modules: %v", err)
	}

	interruptContext, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(interruptContext, *timeout)
	defer cancel()

	report, err := runCascade(ctx, decision, predictor, recordedLLM, *confidenceThreshold)
	if err != nil {
		log.Fatal(err)
	}
	printCascadeReport(report, *replay, *confidenceThreshold)
}

func newCascadeModules(client decide.SystemOneClient, llm core.LLM) (*decide.Decide, *modules.Predict, error) {
	signature := categorySignature()
	decision, err := decide.New(client, signature, decide.Choice[supportCategory]("category",
		decide.Option(categoryBilling, "Charges, invoices, payments, or refunds"),
		decide.Option(categoryTechnical, "Failures, bugs, outages, or unexpected behavior"),
		decide.Option(categoryAccount, "Login, identity, access, permissions, or account settings"),
		decide.Option(categoryProduct, "Product usage, features, exports, or how-to questions"),
	))
	if err != nil {
		return nil, nil, err
	}
	decision.WithName("Jev category gate")
	predictor := modules.NewPredict(signature).WithName("generative category fallback").WithTextOutput()
	predictor.SetLLM(llm)
	return decision, predictor, nil
}

func categorySignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))}},
		[]core.OutputField{{Field: core.NewField("category", core.WithDescription("Exactly one of: billing, technical, account, product"))}},
	).WithInstruction("Classify into exactly one category. Use billing for charges, invoices, payments, or refunds; technical for failures, bugs, outages, or unexpected behavior; account for login, identity, access, permissions, or account settings; and product for product usage, features, exports, or how-to questions.")
}

func runCascade(ctx context.Context, decision *decide.Decide, predictor *modules.Predict, recordedLLM *recordingLLM, threshold float64) (cascadeReport, error) {
	report := cascadeReport{Items: make([]cascadeItem, 0, len(cascadeTickets))}
	for _, ticket := range cascadeTickets {
		item := cascadeItem{Ticket: ticket}
		started := time.Now()
		decisionStarted := time.Now()
		result, err := decision.ProcessDecision(ctx, map[string]any{"ticket": ticket.Text})
		item.DecisionTime = time.Since(decisionStarted)
		if err != nil {
			return cascadeReport{}, fmt.Errorf("classify %q with System One: %w", ticket.Name, err)
		}
		evidence, ok := decide.Get[decide.ChoiceDecision[supportCategory]](result, "category")
		if !ok {
			return cascadeReport{}, fmt.Errorf("classify %q: category evidence is missing", ticket.Name)
		}
		item.Jev = evidence.Value
		item.Confidence = evidence.ProviderConfidence
		report.SystemOneCalls++
		report.SystemOneInput += result.Usage.InputTokens
		report.SystemOneOutput += result.Usage.OutputTokens
		report.DecisionDurations = append(report.DecisionDurations, item.DecisionTime)

		if evidence.ProviderConfidence < threshold {
			item.Escalated = true
			report.Escalated++
			predictStarted := time.Now()
			outputs, err := predictor.Process(ctx, map[string]any{"ticket": ticket.Text})
			item.PredictTime = time.Since(predictStarted)
			if err != nil {
				return cascadeReport{}, fmt.Errorf("classify %q with Predict: %w", ticket.Name, err)
			}
			item.Predict, err = parseCategory(outputs["category"])
			if err != nil {
				return cascadeReport{}, fmt.Errorf("classify %q with Predict: %w", ticket.Name, err)
			}
			if item.Predict == item.Jev {
				report.Agreements++
			}
			report.PredictDurations = append(report.PredictDurations, item.PredictTime)
		}
		item.EndToEndTime = time.Since(started)
		if item.Escalated {
			report.EscalatedDurations = append(report.EscalatedDurations, item.EndToEndTime)
		} else {
			report.DirectDurations = append(report.DirectDurations, item.EndToEndTime)
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

func printCascadeReport(report cascadeReport, replay bool, threshold float64) {
	mode := "live"
	if replay {
		mode = "offline replay"
	}
	fmt.Printf("TypeSafe Jev-first cascade (%s)\n", mode)
	fmt.Printf("Escalate when raw Choice provider confidence is below %.2f.\n", threshold)
	for index, item := range report.Items {
		fmt.Printf("\n%d. %s\n", index+1, item.Ticket.Name)
		fmt.Printf("   Jev: %s (provider confidence %.2f)\n", item.Jev, item.Confidence)
		if item.Escalated {
			fmt.Printf("   Route: Predict -> %s\n", item.Predict)
		} else {
			fmt.Println("   Route: accept Jev decision")
		}
	}

	fmt.Println("\nCascade summary")
	fmt.Printf("  direct Jev path: %d items\n", len(report.Items)-report.Escalated)
	fmt.Printf("  escalated Predict path: %d items (%.1f%%)\n", report.Escalated, percentage(report.Escalated, len(report.Items)))
	fmt.Printf("  Jev/Predict agreement on escalations: %d/%d (%s)\n", report.Agreements, report.Escalated, formatFraction(report.Agreements, report.Escalated))
	fmt.Printf("  System One: %d logical calls, %d input + %d output tokens\n", report.SystemOneCalls, report.SystemOneInput, report.SystemOneOutput)
	fmt.Printf("  generative LLM: %d calls, %d prompt + %d completion tokens (usage on %d/%d calls)\n", report.LLM.Calls, report.LLM.Prompt, report.LLM.Completion, report.LLM.UsageCalls, report.LLM.Calls)
	if replay {
		fmt.Println("  latency: not meaningful in replay mode (local fixture server and deterministic LLM)")
	} else {
		printLatency("System One decision", report.DecisionDurations)
		printLatency("direct end-to-end", report.DirectDurations)
		printLatency("Predict escalation", report.PredictDurations)
		printLatency("escalated end-to-end", report.EscalatedDurations)
	}
	fmt.Println("  Raw provider confidence is not calibrated correctness probability; choose escalation policy on a labeled calibration split.")
}

func printLatency(name string, durations []time.Duration) {
	if len(durations) == 0 {
		fmt.Printf("  %s latency: no calls\n", name)
		return
	}
	fmt.Printf("  %s latency: p50=%s p95=%s\n", name,
		percentile(durations, 0.50).Round(time.Millisecond),
		percentile(durations, 0.95).Round(time.Millisecond),
	)
}

func percentage(numerator, denominator int) float64 {
	if denominator == 0 {
		return 0
	}
	return float64(numerator) / float64(denominator) * 100
}

func formatFraction(numerator, denominator int) string {
	if denominator == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", percentage(numerator, denominator))
}
