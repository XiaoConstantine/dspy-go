// Package main demonstrates basic ACE (Agentic Context Engineering) usage.
// This example shows how to use ACE components independently for trajectory
// recording, learning management, and insight extraction.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
)

func main() {
	// Create a temporary directory for learnings
	tmpDir, err := os.MkdirTemp("", "ace-demo-*")
	if err != nil {
		fmt.Printf("Failed to create temp dir: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmpDir)

	learningsPath := filepath.Join(tmpDir, "learnings.md")

	fmt.Println("=== ACE Basic Demo ===")
	fmt.Printf("Learnings file: %s\n\n", learningsPath)

	// --- Part 1: Configure and Create ACE Manager ---
	fmt.Println("1. Creating ACE Manager")
	fmt.Println("------------------------")

	config := ace.Config{
		Enabled:             true,
		LearningsPath:       learningsPath,
		AsyncReflection:     false, // Synchronous for demo clarity
		CurationFrequency:   5,
		MinConfidence:       0.6,
		MaxTokens:           80000,
		PruneMinRatio:       0.3,
		PruneMinUsage:       3,
		SimilarityThreshold: 0.85,
	}

	// Create a simple reflector for extracting insights
	reflector := ace.NewUnifiedReflector(nil, ace.NewSimpleReflector())

	manager, err := ace.NewManager(config, reflector)
	if err != nil {
		fmt.Printf("Failed to create manager: %v\n", err)
		os.Exit(1)
	}
	defer manager.Close()

	fmt.Println("   Manager created with SimpleReflector")
	fmt.Println()

	// --- Part 2: Record Simulated Agent Trajectories ---
	fmt.Println("2. Recording Agent Trajectories")
	fmt.Println("--------------------------------")

	ctx := context.Background()

	// Simulate a successful task execution
	simulateSuccessfulTask(ctx, manager)

	// Simulate a failed task execution
	simulateFailedTask(ctx, manager)

	// Simulate another successful task that cites a learning
	simulateTaskWithCitation(ctx, manager)

	// --- Part 3: Examine Learnings ---
	fmt.Println("\n3. Examining Learnings")
	fmt.Println("-----------------------")

	learnings := manager.Learnings()
	fmt.Printf("   Total learnings: %d\n", len(learnings))

	for _, l := range learnings {
		fmt.Printf("   - [%s] %s (helpful=%d, harmful=%d, rate=%.1f%%)\n",
			l.ShortCode(), l.Content, l.Helpful, l.Harmful, l.SuccessRate()*100)
	}

	// --- Part 4: Get Context for Injection ---
	fmt.Println("\n4. Context for LLM Injection")
	fmt.Println("-----------------------------")

	contextStr := manager.LearningsContext()
	if contextStr != "" {
		fmt.Println(contextStr)
	} else {
		fmt.Println("   (No learnings to inject yet)")
	}

	// --- Part 5: Check Metrics ---
	fmt.Println("5. ACE Metrics")
	fmt.Println("---------------")

	metrics := manager.Metrics()
	fmt.Printf("   Trajectories processed: %d\n", metrics["trajectories_processed"])
	fmt.Printf("   Insights extracted: %d\n", metrics["insights_extracted"])
	fmt.Printf("   Learnings added: %d\n", metrics["learnings_added"])
	fmt.Printf("   Learnings pruned: %d\n", metrics["learnings_pruned"])

	// --- Part 6: Demonstrate Storage Operations ---
	fmt.Println("\n6. Direct Storage Operations")
	fmt.Println("-----------------------------")

	file := ace.NewLearningsFile(learningsPath)

	// Load and display raw content
	content, err := file.LoadContent()
	if err != nil {
		fmt.Printf("   Error loading content: %v\n", err)
	} else {
		fmt.Println("   Raw learnings file content:")
		fmt.Println("   ---")
		if content != "" {
			fmt.Print("   " + content)
		} else {
			fmt.Println("   (empty)")
		}
		fmt.Println("   ---")
	}

	// Estimate tokens
	tokens, _ := file.EstimateTokens()
	fmt.Printf("   Estimated tokens: %d\n", tokens)

	fmt.Println("\n=== Demo Complete ===")
}

// simulateSuccessfulTask records a successful 2-step task execution.
func simulateSuccessfulTask(ctx context.Context, manager *ace.Manager) {
	fmt.Println("   Recording successful task: 'Find weather in NYC'")

	recorder := manager.StartTrajectory("agent-1", "research", "What is the weather in New York?")

	// Step 1: Search for weather
	recorder.RecordStep(
		"search",                       // action
		"web_search",                   // tool
		"I need to search for current weather data for New York City", // reasoning
		map[string]any{"query": "NYC weather today"},                  // input
		map[string]any{"result": "Sunny, 72F"},                        // output
		nil,                                                           // error
	)

	// Step 2: Format response
	recorder.RecordStep(
		"respond",   // action
		"",          // tool (no tool for final response)
		"The search returned clear weather data, I can now respond", // reasoning
		nil, // input
		map[string]any{"answer": "The weather in NYC is sunny, 72F"}, // output
		nil, // error
	)

	manager.EndTrajectory(ctx, recorder, ace.OutcomeSuccess)
	fmt.Println("   -> Recorded as SUCCESS")
}

// simulateFailedTask records a failed task with an error.
func simulateFailedTask(ctx context.Context, manager *ace.Manager) {
	fmt.Println("   Recording failed task: 'Query database'")

	recorder := manager.StartTrajectory("agent-1", "data", "Get user count from database")

	// Step 1: Attempt database query (fails)
	recorder.RecordStep(
		"query",                          // action
		"database",                       // tool
		"I need to query the users table", // reasoning
		map[string]any{"sql": "SELECT COUNT(*) FROM users"}, // input
		nil, // output (none due to error)
		fmt.Errorf("connection timeout: database unreachable"), // error
	)

	manager.EndTrajectory(ctx, recorder, ace.OutcomeFailure)
	fmt.Println("   -> Recorded as FAILURE")
}

// simulateTaskWithCitation records a task that cites a previous learning.
func simulateTaskWithCitation(ctx context.Context, manager *ace.Manager) {
	fmt.Println("   Recording task with learning citation")

	recorder := manager.StartTrajectory("agent-1", "research", "Find population of Tokyo")

	// Step 1: Search with citation in reasoning
	recorder.RecordStep(
		"search",     // action
		"web_search", // tool
		"Using [L001] efficient search strategy, I'll search for Tokyo population", // reasoning (cites L001!)
		map[string]any{"query": "Tokyo population 2024"}, // input
		map[string]any{"result": "13.96 million"},        // output
		nil, // error
	)

	manager.EndTrajectory(ctx, recorder, ace.OutcomeSuccess)
	fmt.Println("   -> Recorded as SUCCESS (cited L001)")
}
