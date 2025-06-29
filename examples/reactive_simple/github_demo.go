package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
)

// SimpleAgent demonstrates basic reactive agent functionality.
type SimpleAgent struct {
	ID       string
	reactive *workflows.ReactiveWorkflow
}

// NewSimpleAgent creates a new simple reactive agent.
func NewSimpleAgent(id string) *SimpleAgent {
	memory := agents.NewInMemoryStore()
	reactive := workflows.NewReactiveWorkflow(memory)

	agent := &SimpleAgent{
		ID:       id,
		reactive: reactive,
	}

	return agent
}

// OnEvent registers an event handler.
func (a *SimpleAgent) OnEvent(eventType string, handler workflows.EventHandler) {
	_ = a.reactive.GetEventBus().Subscribe(eventType, handler)
}

// EmitEvent sends an event.
func (a *SimpleAgent) EmitEvent(event workflows.Event) error {
	return a.reactive.Emit(event)
}

// Start begins reactive processing.
func (a *SimpleAgent) Start(ctx context.Context) error {
	// Don't start if using shared event bus (already started)
	if a.reactive.GetEventBus() != nil {
		// Event bus is external, don't start it
		return nil
	}
	return a.reactive.Start(ctx)
}

// Stop terminates processing.
func (a *SimpleAgent) Stop() error {
	return a.reactive.Stop()
}

// GitHubPREvent represents a simplified GitHub PR event.
type GitHubPREvent struct {
	Action     string   `json:"action"`
	PRNumber   int      `json:"pr_number"`
	Repository string   `json:"repository"`
	Author     string   `json:"author"`
	Title      string   `json:"title"`
	Files      []string `json:"files"`
}

// ReviewResult represents a code review result.
type ReviewResult struct {
	AgentID    string  `json:"agent_id"`
	PRNumber   int     `json:"pr_number"`
	Approved   bool    `json:"approved"`
	Issues     int     `json:"issues_found"`
	Confidence float64 `json:"confidence"`
}

// GitHubReviewDemo demonstrates reactive agent-based code review.
func GitHubReviewDemo() {
	fmt.Println("\n🔍 GitHub PR Review Agent Demo")
	fmt.Println("==============================")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a shared event bus for agent communication
	sharedEventBus := workflows.NewEventBus(workflows.DefaultEventBusConfig())

	// Create specialized review agents
	orchestrator := NewSimpleAgent("orchestrator")
	primaryReviewer := NewSimpleAgent("primary_reviewer")
	securityReviewer := NewSimpleAgent("security_reviewer")

	// Connect all agents to the shared event bus
	orchestrator.reactive.WithEventBus(sharedEventBus)
	primaryReviewer.reactive.WithEventBus(sharedEventBus)
	securityReviewer.reactive.WithEventBus(sharedEventBus)

	// Set up orchestrator - coordinates the review process
	orchestrator.OnEvent("pr_opened", func(ctx context.Context, event workflows.Event) error {
		prEvent := event.Data.(GitHubPREvent)
		fmt.Printf("🎯 Orchestrator: New PR #%d by %s\n", prEvent.PRNumber, prEvent.Author)
		fmt.Printf("   Title: %s\n", prEvent.Title)
		fmt.Printf("   Files: %v\n", prEvent.Files)

		// Determine what types of review are needed
		needsSecurity := false
		for _, file := range prEvent.Files {
			if contains(file, "auth") || contains(file, "crypto") || contains(file, "security") {
				needsSecurity = true
				break
			}
		}

		// Assign primary review
		primaryEvent := workflows.Event{
			ID:   "primary_" + fmt.Sprint(prEvent.PRNumber),
			Type: "review_assigned",
			Data: prEvent,
			Context: map[string]interface{}{
				"review_type": "primary",
				"pr_number":   prEvent.PRNumber,
			},
		}
		fmt.Printf("   → Assigning primary review\n")
		_ = orchestrator.EmitEvent(primaryEvent)

		// Assign security review if needed
		if needsSecurity {
			securityEvent := workflows.Event{
				ID:   "security_" + fmt.Sprint(prEvent.PRNumber),
				Type: "security_review_assigned",
				Data: prEvent,
				Context: map[string]interface{}{
					"review_type": "security",
					"pr_number":   prEvent.PRNumber,
				},
			}
			fmt.Printf("   → Assigning security review\n")
			_ = orchestrator.EmitEvent(securityEvent)
		}

		return nil
	})

	// Handle review completion aggregation
	orchestrator.OnEvent("review_completed", func(ctx context.Context, event workflows.Event) error {
		result := event.Data.(ReviewResult)
		fmt.Printf("📊 Orchestrator: Review completed by %s for PR #%d\n", result.AgentID, result.PRNumber)
		fmt.Printf("   Approved: %v, Issues: %d, Confidence: %.2f\n",
			result.Approved, result.Issues, result.Confidence)
		return nil
	})

	// Set up primary reviewer
	primaryReviewer.OnEvent("review_assigned", func(ctx context.Context, event workflows.Event) error {
		prEvent := event.Data.(GitHubPREvent)
		fmt.Printf("📝 Primary Reviewer: Analyzing PR #%d\n", prEvent.PRNumber)

		// Simulate review processing
		time.Sleep(1 * time.Second)

		// Generate review result
		result := ReviewResult{
			AgentID:    "primary_reviewer",
			PRNumber:   prEvent.PRNumber,
			Approved:   true,
			Issues:     2, // Found some minor issues
			Confidence: 0.85,
		}

		fmt.Printf("   ✅ Primary review completed: %d issues found\n", result.Issues)

		// Report completion
		completionEvent := workflows.Event{
			ID:   "completion_primary_" + fmt.Sprint(prEvent.PRNumber),
			Type: "review_completed",
			Data: result,
		}

		return primaryReviewer.EmitEvent(completionEvent)
	})

	// Set up security reviewer
	securityReviewer.OnEvent("security_review_assigned", func(ctx context.Context, event workflows.Event) error {
		prEvent := event.Data.(GitHubPREvent)
		fmt.Printf("🔒 Security Reviewer: Scanning PR #%d for vulnerabilities\n", prEvent.PRNumber)

		// Simulate security analysis (takes longer)
		time.Sleep(2 * time.Second)

		// Check for security issues
		hasSecurityIssues := false
		for _, file := range prEvent.Files {
			if contains(file, "auth") && contains(prEvent.Title, "hardcoded") {
				hasSecurityIssues = true
				break
			}
		}

		result := ReviewResult{
			AgentID:    "security_reviewer",
			PRNumber:   prEvent.PRNumber,
			Approved:   !hasSecurityIssues,
			Issues:     map[bool]int{true: 1, false: 0}[hasSecurityIssues],
			Confidence: 0.95,
		}

		if hasSecurityIssues {
			fmt.Printf("   🚨 Security review: CRITICAL security issue found!\n")
		} else {
			fmt.Printf("   ✅ Security review: No security issues detected\n")
		}

		// Report completion
		completionEvent := workflows.Event{
			ID:   "completion_security_" + fmt.Sprint(prEvent.PRNumber),
			Type: "review_completed",
			Data: result,
		}

		return securityReviewer.EmitEvent(completionEvent)
	})

	// Start shared event bus first
	err := sharedEventBus.Start(ctx)
	if err != nil {
		log.Fatalf("Failed to start shared event bus: %v", err)
	}
	defer func() { _ = sharedEventBus.Stop() }()

	// Agents are ready (using shared event bus, no need to start individually)
	fmt.Println("\n📡 Review agents ready...")
	agents := []*SimpleAgent{orchestrator, primaryReviewer, securityReviewer}

	// Set up cleanup
	defer func() {
		for _, agent := range agents {
			_ = agent.Stop()
		}
	}()

	// Wait for agents to initialize
	time.Sleep(100 * time.Millisecond)

	fmt.Println("\n🎭 Simulating GitHub PR events...")

	// Simulate different types of PRs
	testPRs := []GitHubPREvent{
		{
			Action:     "opened",
			PRNumber:   123,
			Repository: "company/secure-app",
			Author:     "developer1",
			Title:      "Add user authentication system",
			Files:      []string{"auth.go", "middleware.go", "main.go"},
		},
		{
			Action:     "opened",
			PRNumber:   124,
			Repository: "company/secure-app",
			Author:     "junior-dev",
			Title:      "Fix typo in README",
			Files:      []string{"README.md"},
		},
		{
			Action:     "opened",
			PRNumber:   125,
			Repository: "company/secure-app",
			Author:     "bad-actor",
			Title:      "Add hardcoded API keys for auth",
			Files:      []string{"auth.go", "config.go"},
		},
	}

	for i, pr := range testPRs {
		fmt.Printf("\n📨 PR Event %d: %s opened PR #%d\n", i+1, pr.Author, pr.PRNumber)

		event := workflows.Event{
			ID:       fmt.Sprintf("github_pr_%d", pr.PRNumber),
			Type:     "pr_opened",
			Data:     pr,
			Priority: 5,
			Context: map[string]interface{}{
				"repository": pr.Repository,
				"author":     pr.Author,
			},
		}

		err := orchestrator.EmitEvent(event)
		if err != nil {
			log.Printf("Failed to emit PR event: %v", err)
		}

		// Wait between PRs to see the review flow
		time.Sleep(4 * time.Second)
	}

	fmt.Println("\n🏁 GitHub review demo completed!")
	fmt.Println("\nDemo showed:")
	fmt.Println("✅ Event-driven PR review orchestration")
	fmt.Println("✅ Specialized agents (primary, security reviewers)")
	fmt.Println("✅ Intelligent review assignment based on file content")
	fmt.Println("✅ Parallel review processing")
	fmt.Println("✅ Centralized result aggregation")
	fmt.Println("✅ Real-time reactive agent communication")
}

// Helper function for string contains check (case insensitive).
func contains(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

func main() {
	// Run the basic reactive demo first
	fmt.Println("🚀 Reactive Workflow Demo")
	fmt.Println("========================")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create shared event bus for communication between agents
	sharedEventBus := workflows.NewEventBus(workflows.DefaultEventBusConfig())
	err := sharedEventBus.Start(ctx)
	if err != nil {
		log.Fatalf("Failed to start shared event bus: %v", err)
	}
	defer func() { _ = sharedEventBus.Stop() }()

	// Create two simple agents with shared event bus
	agentA := NewSimpleAgent("agent_a")
	agentB := NewSimpleAgent("agent_b")
	
	// Configure agents to use the shared event bus
	agentA.reactive.WithEventBus(sharedEventBus)
	agentB.reactive.WithEventBus(sharedEventBus)

	// Set up event handlers
	agentA.OnEvent("user_message", func(ctx context.Context, event workflows.Event) error {
		fmt.Printf("🤖 Agent A received: %v\n", event.Data)

		// Agent A responds by emitting a processed event
		response := workflows.Event{
			ID:   "response_" + event.ID,
			Type: "message_processed",
			Data: fmt.Sprintf("Agent A processed: %v", event.Data),
		}

		return agentA.EmitEvent(response)
	})

	agentB.OnEvent("message_processed", func(ctx context.Context, event workflows.Event) error {
		fmt.Printf("✅ Agent B confirmed: %v\n", event.Data)
		return nil
	})

	// Agents don't need to start their own event bus since they use the shared one
	// The shared event bus is already started above

	time.Sleep(100 * time.Millisecond)

	// Send a test message
	event := workflows.Event{
		ID:   "test_1",
		Type: "user_message",
		Data: "Hello reactive world!",
	}

	_ = agentA.EmitEvent(event)
	time.Sleep(500 * time.Millisecond)

	// Now run the GitHub demo
	GitHubReviewDemo()
}
