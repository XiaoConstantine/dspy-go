package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

type ticketCategory string

const (
	categoryBilling   ticketCategory = "billing"
	categoryTechnical ticketCategory = "technical"
	categoryAccount   ticketCategory = "account"
	categoryOther     ticketCategory = "other"
)

func main() {
	model := flag.String("model", "", "TypeSafe model override (otherwise TYPESAFE_DEFAULT_MODEL or jev-latest)")
	ticket := flag.String("ticket", "Payment failed and checkout is unavailable for all users.", "support ticket to classify")
	urgentThreshold := flag.Float64("urgent-threshold", 0.5, "local P(true) threshold for the urgent decision")
	timeout := flag.Duration("timeout", 20*time.Second, "total example deadline")
	flag.Parse()

	clientOptions := make([]typesafe.ClientOption, 0, 1)
	if *model != "" {
		clientOptions = append(clientOptions, typesafe.WithDefaultModel(*model))
	}
	client, err := typesafe.NewClient(clientOptions...)
	if err != nil {
		log.Fatalf("create TypeSafe client: %v", err)
	}

	module, err := decide.New(
		client,
		ticketSignature(),
		decide.Noul("urgent"),
		decide.Score("severity",
			decide.Level(0, "Minor: little or no user impact"),
			decide.Level(2, "Disruptive: an important workflow is degraded"),
			decide.Level(10, "Blocking: users cannot complete a critical workflow"),
		),
		decide.Choice[ticketCategory]("category",
			decide.Option(categoryBilling, "Payments, invoices, refunds, or charges"),
			decide.Option(categoryTechnical, "Product malfunction or service outage"),
			decide.Option(categoryAccount, "Login, identity, permissions, or account access"),
			decide.Option(categoryOther, "Anything not covered by the other categories"),
		),
	)
	if err != nil {
		log.Fatalf("create Decide module: %v", err)
	}
	if err := module.SetThreshold("urgent", *urgentThreshold); err != nil {
		log.Fatalf("configure urgent threshold: %v", err)
	}

	interruptContext, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(interruptContext, *timeout)
	defer cancel()

	result, err := module.ProcessDecision(ctx, map[string]any{"ticket": *ticket})
	if err != nil {
		log.Fatalf("classify ticket: %v", err)
	}

	urgent := result.Decisions["urgent"].(decide.NoulDecision)
	severity := result.Decisions["severity"].(decide.ScoreDecision)
	category := result.Decisions["category"].(decide.ChoiceDecision[ticketCategory])
	severityProbabilities := severity.Probabilities()
	categoryProbabilities := category.Probabilities()

	fmt.Printf("Ticket: %s\n\n", *ticket)
	fmt.Println("Native outputs")
	fmt.Printf("  urgent:  %t\n", result.Outputs["urgent"])
	fmt.Printf("  severity: %.2f\n", result.Outputs["severity"])
	fmt.Printf("  category: %s\n", result.Outputs["category"])

	fmt.Println("\nDecision evidence")
	fmt.Printf("  urgent: P(true)=%.3f, threshold=%.2f, boundary confidence=%.3f\n",
		urgent.Probability, urgent.Threshold, urgent.Confidence)
	fmt.Printf("  severity: value=%.2f, provider score=%.2f, provider confidence=%.3f\n",
		severity.Value, severity.ProviderScore, severity.ProviderConfidence)
	fmt.Printf("    level probabilities: minor=%.3f disruptive=%.3f blocking=%.3f\n",
		severityProbabilities[0], severityProbabilities[1], severityProbabilities[2])
	fmt.Printf("  category: local=%s, provider=%s, provider confidence=%.3f\n",
		category.Value, category.ProviderValue, category.ProviderConfidence)
	fmt.Printf("    probabilities: billing=%.3f technical=%.3f account=%.3f other=%.3f\n",
		categoryProbabilities[string(categoryBilling)],
		categoryProbabilities[string(categoryTechnical)],
		categoryProbabilities[string(categoryAccount)],
		categoryProbabilities[string(categoryOther)],
	)

	fmt.Printf("\nProvider model: %s\n", result.Model)
	fmt.Printf("Usage: %d input tokens, %d output tokens\n", result.Usage.InputTokens, result.Usage.OutputTokens)
	if result.RequestID != "" {
		fmt.Printf("Request ID: %s\n", result.RequestID)
	}
}

func ticketSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))},
		},
		[]core.OutputField{
			{Field: core.NewField("urgent", core.WithDescription("Does this ticket require immediate attention?"))},
			{Field: core.NewField("severity", core.WithDescription("How severe is the user impact?"))},
			{Field: core.NewField("category", core.WithDescription("What kind of support issue is this?"))},
		},
	).WithInstruction("Triage the support ticket using only the supplied ticket text.")
}
