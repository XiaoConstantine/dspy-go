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
)

func main() {
	replay := flag.Bool("replay", false, "run entirely offline with recorded System One responses and a deterministic LLM")
	ticket := flag.String("ticket", "", "process one ticket instead of the built-in batch (live mode, or an exact replay ticket)")
	model := flag.String("model", "", "TypeSafe model override (otherwise TYPESAFE_DEFAULT_MODEL or jev-latest)")
	llmModel := flag.String("llm-model", string(core.ModelGoogleGeminiFlash), "generative model used for drafted replies")
	llmAPIKey := flag.String("llm-api-key", "", "generative provider key (defaults to DSPY_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY)")
	timeout := flag.Duration("timeout", 45*time.Second, "total example deadline")
	flag.Parse()
	log.SetFlags(0)

	systemOne, closeReplay, err := newSystemOneClient(*replay, *model)
	if err != nil {
		log.Fatalf("create System One client: %v", err)
	}
	defer closeReplay()
	countedSystemOne := &countingSystemOneClient{inner: systemOne}

	program, err := newCheapGateProgram(countedSystemOne)
	if err != nil {
		log.Fatalf("create cheap-gate program: %v", err)
	}

	var generative core.LLM
	if *replay {
		generative = newReplayLLM()
	} else {
		generative, err = newLiveLLM(*llmAPIKey, *llmModel)
		if err != nil {
			log.Fatalf("create generative LLM: %v", err)
		}
	}
	countedLLM := &countingLLM{LLM: generative}
	configureProgramLLM(&program, countedLLM)

	tickets := replayTickets
	if strings.TrimSpace(*ticket) != "" {
		tickets = []sampleTicket{{Name: "custom ticket", Text: strings.TrimSpace(*ticket)}}
	}

	interruptContext, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(interruptContext, *timeout)
	defer cancel()

	mode := "live"
	if *replay {
		mode = "offline replay"
	}
	fmt.Printf("TypeSafe cheap-gate program (%s)\n", mode)
	fmt.Println("Decide triages every ticket; ChainOfThought runs only on the draft route.")

	routeCounts := map[string]int{}
	for index, item := range tickets {
		outputs, err := program.Execute(ctx, map[string]any{"ticket": item.Text})
		if err != nil {
			log.Fatalf("process %q: %v", item.Name, err)
		}
		route, _ := outputs["route"].(string)
		routeCounts[route]++

		fmt.Printf("\n%d. %s\n", index+1, item.Name)
		fmt.Printf("   Ticket: %s\n", item.Text)
		fmt.Printf("   Decision: category=%v (confidence %.2f), answerable=%v (P %.2f), needs_human=%v (P %.2f)\n",
			outputs["category"], floatOutput(outputs, "category_confidence"),
			outputs["answerable"], floatOutput(outputs, "answerable_probability"),
			outputs["needs_human"], floatOutput(outputs, "needs_human_probability"),
		)
		fmt.Printf("   Route: %s\n", route)
		if reply, ok := outputs["reply"].(string); ok {
			fmt.Printf("   Reply: %s\n", reply)
		} else {
			fmt.Printf("   Result: %v\n", outputs["message"])
		}
		fmt.Printf("   Evidence: model=%v request_id=%v\n", outputs["decision_model"], outputs["decision_request_id"])
	}

	llmCalls := int(countedLLM.Calls())
	avoided := len(tickets) - routeCounts[routeDraft]
	fmt.Println("\nBatch summary")
	fmt.Printf("  tickets: %d\n", len(tickets))
	fmt.Printf("  routes: draft=%d tools=%d human=%d\n", routeCounts[routeDraft], routeCounts[routeTools], routeCounts[routeHuman])
	fmt.Printf("  System One calls: %d\n", countedSystemOne.Calls())
	fmt.Printf("  generative LLM calls: %d\n", llmCalls)
	fmt.Printf("  one-call-per-ticket generations avoided by the gate: %d of %d\n", avoided, len(tickets))
	fmt.Println("  SetLLM was applied to every program module; Decide retained its explicit System One client.")
}

func floatOutput(outputs map[string]any, name string) float64 {
	value, _ := outputs[name].(float64)
	return value
}
