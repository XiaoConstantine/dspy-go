package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"reflect"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
)

type labeledTicket struct {
	Text   string
	Urgent bool
}

var calibrationTickets = []labeledTicket{
	{Text: "Checkout is down for every customer; no payments can complete.", Urgent: true},
	{Text: "The production database is unavailable and all writes are failing.", Urgent: true},
	{Text: "I think my account was taken over and unfamiliar sessions are active.", Urgent: true},
	{Text: "I cannot sign in and I have a customer presentation in an hour.", Urgent: true},
	{Text: "The invoice total looks wrong and finance closes the books today.", Urgent: false},
	{Text: "Please refund a duplicate charge from last week.", Urgent: false},
	{Text: "How can I export my project data as CSV?", Urgent: false},
	{Text: "Where do I change my profile avatar?", Urgent: false},
}

var heldOutTickets = []labeledTicket{
	{Text: "All production uploads are failing with data-loss errors.", Urgent: true},
	{Text: "An API secret was exposed in a public repository.", Urgent: true},
	{Text: "The monthly analytics report is taking longer than usual.", Urgent: false},
	{Text: "Can you refund the add-on I purchased yesterday?", Urgent: false},
	{Text: "There is a typo in the getting-started guide.", Urgent: false},
	{Text: "How do I replace my profile picture?", Urgent: false},
}

type thresholdBand struct {
	Low  float64
	High float64
}

var candidateBands = []thresholdBand{
	{Low: 0.05, High: 0.95},
	{Low: 0.10, High: 0.90},
	{Low: 0.20, High: 0.80},
	{Low: 0.30, High: 0.70},
	{Low: 0.40, High: 0.60},
}

type bandMetrics struct {
	Band               thresholdBand
	UrgentPrecision    float64
	NotUrgentPrecision float64
	Coverage           float64
	AutoUrgent         int
	AutoNotUrgent      int
	HumanReview        int
}

type persistedBandPolicy struct {
	SchemaVersion    int            `json:"schema_version"`
	LowThreshold     float64        `json:"low_threshold"`
	DecideParameters map[string]any `json:"decide_parameters"`
}

func main() {
	replay := flag.Bool("replay", false, "run offline against recorded System One responses")
	model := flag.String("model", "", "TypeSafe model override (replay fixtures require jev-replay; live defaults to TYPESAFE_DEFAULT_MODEL or jev-latest)")
	targetPrecision := flag.Float64("target-precision", 0.95, "minimum precision required on each automatic side")
	timeout := flag.Duration("timeout", 45*time.Second, "total example deadline")
	flag.Parse()
	log.SetFlags(0)

	if math.IsNaN(*targetPrecision) || math.IsInf(*targetPrecision, 0) || *targetPrecision < 0 || *targetPrecision > 1 {
		log.Fatal("target-precision must be a finite value in [0, 1]")
	}

	client, closeReplay, err := newTuningClient(*replay, *model)
	if err != nil {
		log.Fatalf("create System One client: %v", err)
	}
	defer closeReplay()
	counted := &countingClient{inner: client}

	module, err := newUrgencyModule(counted)
	if err != nil {
		log.Fatalf("create urgency module: %v", err)
	}

	interruptContext, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(interruptContext, *timeout)
	defer cancel()

	calibrationResults, err := collectResults(ctx, module, calibrationTickets)
	if err != nil {
		log.Fatalf("record calibration set: %v", err)
	}
	calibrationCalls := counted.Calls()

	rows := make([]bandMetrics, 0, len(candidateBands))
	for _, band := range candidateBands {
		row, err := evaluateBand(module, calibrationTickets, calibrationResults, band)
		if err != nil {
			log.Fatalf("evaluate calibration band [%.2f, %.2f]: %v", band.Low, band.High, err)
		}
		rows = append(rows, row)
	}
	if counted.Calls() != calibrationCalls {
		log.Fatal("local calibration sweep unexpectedly made a provider call")
	}

	chosen, found := chooseBand(rows, *targetPrecision)
	if !found {
		log.Fatalf("no threshold band met %.0f%% precision on both automatic sides", *targetPrecision*100)
	}
	if err := module.SetThreshold("urgent", chosen.Band.High); err != nil {
		log.Fatalf("set chosen urgent threshold: %v", err)
	}

	policy := persistedBandPolicy{
		SchemaVersion:    1,
		LowThreshold:     chosen.Band.Low,
		DecideParameters: module.GetTunedParameters(),
	}
	encoded, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		log.Fatalf("encode chosen policy: %v", err)
	}
	var persisted persistedBandPolicy
	if err := json.Unmarshal(encoded, &persisted); err != nil {
		log.Fatalf("decode chosen policy: %v", err)
	}
	if persisted.SchemaVersion != 1 {
		log.Fatalf("unsupported policy schema version %d", persisted.SchemaVersion)
	}
	restored, err := newUrgencyModule(counted)
	if err != nil {
		log.Fatalf("create restored module: %v", err)
	}
	if err := restored.SetTunedParameters(persisted.DecideParameters); err != nil {
		log.Fatalf("reload Decide parameters: %v", err)
	}
	restoredParameters := restored.GetTunedParameters()
	if !reflect.DeepEqual(module.GetTunedParameters(), restoredParameters) {
		log.Fatal("reloaded Decide parameters do not match the chosen parameters")
	}
	frozenHigh, err := urgencyThreshold(restoredParameters)
	if err != nil {
		log.Fatalf("read restored urgent threshold: %v", err)
	}
	for _, result := range calibrationResults {
		originalProjection, err := module.Reinterpret(result)
		if err != nil {
			log.Fatalf("reinterpret with selected module: %v", err)
		}
		restoredProjection, err := restored.Reinterpret(result)
		if err != nil {
			log.Fatalf("reinterpret with restored module: %v", err)
		}
		if !reflect.DeepEqual(originalProjection.Outputs, restoredProjection.Outputs) {
			log.Fatal("restored parameters produced a different local decision")
		}
	}
	callsAfterTuning := counted.Calls()

	// The held-out labels are touched only after the band is frozen and restored.
	heldOutResults, err := collectResults(ctx, restored, heldOutTickets)
	if err != nil {
		log.Fatalf("record held-out test set: %v", err)
	}
	heldOut, err := evaluateBand(restored, heldOutTickets, heldOutResults, thresholdBand{
		Low:  persisted.LowThreshold,
		High: frozenHigh,
	})
	if err != nil {
		log.Fatalf("evaluate held-out test set: %v", err)
	}

	mode := "live"
	if *replay {
		mode = "offline replay"
	}
	fmt.Printf("TypeSafe two-sided threshold tuning (%s)\n", mode)
	fmt.Printf("Calibration set: %d tickets, model %s\n", len(calibrationResults), calibrationResults[0].Model)
	fmt.Println("Automatic decisions are urgent at P >= high and not urgent at P <= low; the middle routes to a human.")
	fmt.Println("\n  low   high  urgent precision  not-urgent precision  coverage  auto U/N  review")
	for _, row := range rows {
		fmt.Printf(" %.2f   %.2f        %8s            %8s       %6.1f%%    %d/%d       %d\n",
			row.Band.Low, row.Band.High, formatRate(row.UrgentPrecision),
			formatRate(row.NotUrgentPrecision), row.Coverage*100,
			row.AutoUrgent, row.AutoNotUrgent, row.HumanReview,
		)
	}
	fmt.Printf("\nFrozen band from calibration: low=%.2f high=%.2f (coverage %.1f%%)\n",
		chosen.Band.Low, chosen.Band.High, chosen.Coverage*100)
	fmt.Printf("Held-out test, reported once: urgent precision %s, not-urgent precision %s, coverage %.1f%% (%d/%d automatic)\n",
		formatRate(heldOut.UrgentPrecision), formatRate(heldOut.NotUrgentPrecision),
		heldOut.Coverage*100, heldOut.AutoUrgent+heldOut.AutoNotUrgent, len(heldOutTickets))
	fmt.Printf("Provider calls: %d calibration + %d during sweep/reload + %d held-out\n",
		calibrationCalls, callsAfterTuning-calibrationCalls, counted.Calls()-callsAfterTuning)
	fmt.Printf("Persisted and reloaded policy:\n%s\n", encoded)
	fmt.Println("\nThese splits are intentionally tiny and the numbers are illustrative only; use adequately sized, disjoint calibration and test sets before deployment.")
}

func newUrgencyModule(client decide.SystemOneClient) (*decide.Decide, error) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))}},
		[]core.OutputField{{Field: core.NewField("urgent", core.WithDescription("Does this require immediate operational or security escalation?"))}},
	).WithInstruction("Judge operational urgency from the supplied ticket. Routine account and billing requests are not urgent merely because the customer uses time-sensitive language.")
	return decide.New(client, signature, decide.Noul("urgent"))
}

func collectResults(ctx context.Context, module *decide.Decide, tickets []labeledTicket) ([]*decide.Result, error) {
	results := make([]*decide.Result, len(tickets))
	for index, ticket := range tickets {
		result, err := module.ProcessDecision(ctx, map[string]any{"ticket": ticket.Text})
		if err != nil {
			return nil, fmt.Errorf("ticket %d: %w", index+1, err)
		}
		results[index] = result
	}
	return results, nil
}

func evaluateBand(module *decide.Decide, tickets []labeledTicket, results []*decide.Result, band thresholdBand) (bandMetrics, error) {
	if len(tickets) == 0 || len(tickets) != len(results) {
		return bandMetrics{}, fmt.Errorf("tickets and results must have the same non-zero length")
	}
	if !finiteProbability(band.Low) || !finiteProbability(band.High) || band.Low >= band.High {
		return bandMetrics{}, fmt.Errorf("thresholds must be finite probabilities with low below high")
	}
	if err := module.SetThreshold("urgent", band.High); err != nil {
		return bandMetrics{}, err
	}

	urgentCorrect := 0
	notUrgentCorrect := 0
	metrics := bandMetrics{Band: band}
	for index, result := range results {
		projected, err := module.Reinterpret(result)
		if err != nil {
			return bandMetrics{}, err
		}
		urgent, ok := decide.Get[decide.NoulDecision](projected, "urgent")
		if !ok {
			return bandMetrics{}, fmt.Errorf("urgent evidence is missing")
		}
		switch {
		case urgent.Value:
			metrics.AutoUrgent++
			if tickets[index].Urgent {
				urgentCorrect++
			}
		case urgent.Probability <= band.Low:
			metrics.AutoNotUrgent++
			if !tickets[index].Urgent {
				notUrgentCorrect++
			}
		default:
			metrics.HumanReview++
		}
	}
	metrics.UrgentPrecision = ratioOrNaN(urgentCorrect, metrics.AutoUrgent)
	metrics.NotUrgentPrecision = ratioOrNaN(notUrgentCorrect, metrics.AutoNotUrgent)
	metrics.Coverage = float64(metrics.AutoUrgent+metrics.AutoNotUrgent) / float64(len(results))
	return metrics, nil
}

func chooseBand(rows []bandMetrics, targetPrecision float64) (bandMetrics, bool) {
	var chosen bandMetrics
	found := false
	for _, row := range rows {
		if math.IsNaN(row.UrgentPrecision) || math.IsNaN(row.NotUrgentPrecision) ||
			row.UrgentPrecision < targetPrecision || row.NotUrgentPrecision < targetPrecision {
			continue
		}
		if !found || row.Coverage > chosen.Coverage {
			chosen = row
			found = true
		}
	}
	return chosen, found
}

func urgencyThreshold(parameters map[string]any) (float64, error) {
	thresholds, ok := parameters["thresholds"].(map[string]float64)
	if !ok {
		return 0, fmt.Errorf("thresholds have type %T, want map[string]float64", parameters["thresholds"])
	}
	threshold, ok := thresholds["urgent"]
	if !ok {
		return 0, fmt.Errorf("urgent threshold is missing")
	}
	return threshold, nil
}

func finiteProbability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func ratioOrNaN(numerator, denominator int) float64 {
	if denominator == 0 {
		return math.NaN()
	}
	return float64(numerator) / float64(denominator)
}

func formatRate(value float64) string {
	if math.IsNaN(value) {
		return "       -"
	}
	return fmt.Sprintf("%7.1f%%", value*100)
}
