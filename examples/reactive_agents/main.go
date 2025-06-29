package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/communication"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// GitHubPREvent represents a GitHub pull request event.
type GitHubPREvent struct {
	Action     string       `json:"action"` // opened, closed, synchronize
	PRNumber   int          `json:"pr_number"`
	Repository string       `json:"repository"`
	Author     string       `json:"author"`
	Title      string       `json:"title"`
	Body       string       `json:"body"`
	Files      []GitHubFile `json:"files"`
	Branch     string       `json:"branch"`
}

// GitHubFile represents a file in a PR.
type GitHubFile struct {
	Filename string `json:"filename"`
	Status   string `json:"status"` // added, modified, deleted
	Changes  int    `json:"changes"`
	Content  string `json:"content"`
}

// ReviewResult represents the output of a code review.
type ReviewResult struct {
	AgentID     string             `json:"agent_id"`
	PRNumber    int                `json:"pr_number"`
	Approved    bool               `json:"approved"`
	Issues      []ReviewIssue      `json:"issues"`
	Suggestions []ReviewSuggestion `json:"suggestions"`
	Confidence  float64            `json:"confidence"`
}

// ReviewIssue represents a problem found during review.
type ReviewIssue struct {
	Severity    string `json:"severity"` // critical, major, minor
	Category    string `json:"category"` // security, performance, style, etc.
	File        string `json:"file"`
	Line        int    `json:"line"`
	Description string `json:"description"`
}

// ReviewSuggestion represents an improvement suggestion.
type ReviewSuggestion struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Description string `json:"description"`
	Code        string `json:"suggested_code"`
}

// PRReviewOrchestrator coordinates the entire review process.
func main() {
	fmt.Println("🚀 Starting GitHub PR Review Agent Team...")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create agent network
	network := communication.NewAgentNetwork()
	err := network.Start(ctx)
	if err != nil {
		log.Fatalf("Failed to start agent network: %v", err)
	}
	defer func() { _ = network.Stop() }()

	// Create specialized review agents
	primaryReviewer := createPrimaryReviewAgent()
	securityAgent := createSecurityReviewAgent()
	performanceAgent := createPerformanceReviewAgent()
	codeStyleAgent := createCodeStyleReviewAgent()
	orchestrator := createOrchestratorAgent()

	// Connect agents to network
	agents := []*communication.Agent{
		orchestrator, primaryReviewer, securityAgent,
		performanceAgent, codeStyleAgent,
	}

	for _, agent := range agents {
		err := agent.ConnectToNetwork(network)
		if err != nil {
			log.Fatalf("Failed to connect agent %s: %v", agent.GetID(), err)
		}
		defer func() { _ = agent.Stop() }()
	}

	fmt.Printf("✅ Connected %d review agents to network\n", len(agents))

	// Simulate GitHub PR events
	simulateGitHubEvents(orchestrator, ctx)

	fmt.Println("🔄 PR review simulation completed")

	fmt.Println("🏁 Demo completed!")
}

// createOrchestratorAgent creates the main coordination agent.
func createOrchestratorAgent() *communication.Agent {
	memory := agents.NewInMemoryStore()
	orchestrator := communication.NewAgent("orchestrator", "PR Review Orchestrator", memory)

	// In a real implementation, we would register event handlers here
	// orchestrator.OnModule("pr_opened", &PROpenedHandler{})
	// orchestrator.OnModule("review_completed", &ReviewCompletedHandler{})
	// orchestrator.OnModule("all_reviews_completed", &FinalReviewHandler{})

	orchestrator.UpdateStatus(communication.AgentStatus{
		State:        communication.AgentStateIdle,
		Load:         0.1,
		Capabilities: []string{"orchestration", "aggregation"},
	})

	return orchestrator
}

// createPrimaryReviewAgent creates the main code review agent.
func createPrimaryReviewAgent() *communication.Agent {
	memory := agents.NewInMemoryStore()
	reviewer := communication.NewAgent("primary_reviewer", "Primary Code Reviewer", memory)

	// In a real implementation, we would register handlers here
	// reviewer.OnModule("review_assigned", &PrimaryReviewHandler{})

	reviewer.UpdateStatus(communication.AgentStatus{
		State:        communication.AgentStateIdle,
		Load:         0.0,
		Capabilities: []string{"code_review", "general_analysis"},
	})

	return reviewer
}

// createSecurityReviewAgent creates the security-focused review agent.
func createSecurityReviewAgent() *communication.Agent {
	memory := agents.NewInMemoryStore()
	security := communication.NewAgent("security_reviewer", "Security Review Specialist", memory)

	// In a real implementation, we would register handlers here
	// security.OnModule("security_review_assigned", &SecurityReviewHandler{})

	security.UpdateStatus(communication.AgentStatus{
		State:        communication.AgentStateIdle,
		Load:         0.0,
		Capabilities: []string{"security_review", "vulnerability_analysis"},
	})

	return security
}

// createPerformanceReviewAgent creates the performance-focused review agent.
func createPerformanceReviewAgent() *communication.Agent {
	memory := agents.NewInMemoryStore()
	performance := communication.NewAgent("performance_reviewer", "Performance Review Specialist", memory)

	// In a real implementation, we would register handlers here
	// performance.OnModule("performance_review_assigned", &PerformanceReviewHandler{})

	performance.UpdateStatus(communication.AgentStatus{
		State:        communication.AgentStateIdle,
		Load:         0.0,
		Capabilities: []string{"performance_review", "optimization_analysis"},
	})

	return performance
}

// createCodeStyleAgent creates the code style review agent.
func createCodeStyleReviewAgent() *communication.Agent {
	memory := agents.NewInMemoryStore()
	style := communication.NewAgent("style_reviewer", "Code Style Reviewer", memory)

	// In a real implementation, we would register handlers here
	// style.OnModule("style_review_assigned", &StyleReviewHandler{})

	style.UpdateStatus(communication.AgentStatus{
		State:        communication.AgentStateIdle,
		Load:         0.0,
		Capabilities: []string{"style_review", "formatting_analysis"},
	})

	return style
}

// Handler modules for different review stages

// PROpenedHandler orchestrates the review process when a PR is opened.
type PROpenedHandler struct{}

func (h *PROpenedHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert inputs for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := h.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (h *PROpenedHandler) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	prEvent, ok := event.Data.(GitHubPREvent)
	if !ok {
		return nil, fmt.Errorf("invalid PR event data")
	}

	fmt.Printf("🔍 Orchestrator: New PR #%d opened by %s\n", prEvent.PRNumber, prEvent.Author)
	fmt.Printf("   Title: %s\n", prEvent.Title)
	fmt.Printf("   Files changed: %d\n", len(prEvent.Files))

	// Analyze PR to determine which specialists are needed
	specialists := determineRequiredSpecialists(prEvent)

	fmt.Printf("   Required specialists: %v\n", specialists)

	// Create review context (would be used in a real implementation)
	_ = map[string]interface{}{
		"pr_event":    prEvent,
		"review_id":   fmt.Sprintf("review_%d_%d", prEvent.PRNumber, time.Now().Unix()),
		"specialists": specialists,
		"status":      "in_progress",
	}

	// Delegate to specialist agents
	for _, specialist := range specialists {
		// In a real implementation, we would emit assignment events here
		// assignmentEvent := workflows.Event{
		//	ID:   fmt.Sprintf("assignment_%s_%d", specialist, prEvent.PRNumber),
		//	Type: specialist + "_review_assigned",
		//	Data: reviewContext,
		//	Context: map[string]interface{}{
		//		"pr_number": prEvent.PRNumber,
		//		"specialist": specialist,
		//	},
		// }

		// Emit assignment event (would be sent via agent network in real implementation)
		fmt.Printf("   → Assigning %s review to %s agent\n", specialist, specialist)
	}

	return map[string]interface{}{
		"processed":            true,
		"pr_number":            prEvent.PRNumber,
		"specialists_assigned": len(specialists),
	}, nil
}

func (h *PROpenedHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *PROpenedHandler) SetSignature(signature core.Signature) {}
func (h *PROpenedHandler) SetLLM(llm core.LLM)                   {}

func (h *PROpenedHandler) Clone() core.Module {
	return &PROpenedHandler{}
}

// PrimaryReviewHandler handles general code review.
type PrimaryReviewHandler struct{}

func (h *PrimaryReviewHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert inputs for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := h.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (h *PrimaryReviewHandler) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	reviewContext, ok := event.Data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid review context")
	}

	prEvent := reviewContext["pr_event"].(GitHubPREvent)

	fmt.Printf("📝 Primary Reviewer: Analyzing PR #%d\n", prEvent.PRNumber)

	// Simulate primary code review
	time.Sleep(2 * time.Second) // Simulate processing time

	// Generate mock review results
	result := ReviewResult{
		AgentID:    "primary_reviewer",
		PRNumber:   prEvent.PRNumber,
		Approved:   true,
		Confidence: 0.85,
		Issues: []ReviewIssue{
			{
				Severity:    "minor",
				Category:    "readability",
				File:        "main.go",
				Line:        42,
				Description: "Consider adding a comment to explain the algorithm",
			},
		},
		Suggestions: []ReviewSuggestion{
			{
				File:        "main.go",
				Line:        15,
				Description: "Consider using a more descriptive variable name",
				Code:        "userAuthToken := getToken()",
			},
		},
	}

	fmt.Printf("   ✅ Primary review completed: %d issues, %d suggestions\n",
		len(result.Issues), len(result.Suggestions))

	// Would emit review_completed event in real implementation

	return map[string]interface{}{
		"processed": true,
		"result":    result,
	}, nil
}

func (h *PrimaryReviewHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *PrimaryReviewHandler) SetSignature(signature core.Signature) {}
func (h *PrimaryReviewHandler) SetLLM(llm core.LLM)                   {}

func (h *PrimaryReviewHandler) Clone() core.Module {
	return &PrimaryReviewHandler{}
}

// SecurityReviewHandler handles security-focused review.
type SecurityReviewHandler struct{}

func (h *SecurityReviewHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert inputs for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := h.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (h *SecurityReviewHandler) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	reviewContext, ok := event.Data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid review context")
	}

	prEvent := reviewContext["pr_event"].(GitHubPREvent)

	fmt.Printf("🔒 Security Reviewer: Scanning PR #%d for vulnerabilities\n", prEvent.PRNumber)

	// Simulate security analysis
	time.Sleep(3 * time.Second) // Security scans take longer

	// Generate mock security review
	result := ReviewResult{
		AgentID:    "security_reviewer",
		PRNumber:   prEvent.PRNumber,
		Approved:   false, // Found security issues
		Confidence: 0.95,
		Issues: []ReviewIssue{
			{
				Severity:    "critical",
				Category:    "security",
				File:        "auth.go",
				Line:        28,
				Description: "Potential SQL injection vulnerability in user input handling",
			},
			{
				Severity:    "major",
				Category:    "security",
				File:        "config.go",
				Line:        5,
				Description: "Hardcoded API key detected",
			},
		},
	}

	fmt.Printf("   🚨 Security review completed: %d security issues found!\n", len(result.Issues))

	return map[string]interface{}{
		"processed": true,
		"result":    result,
	}, nil
}

func (h *SecurityReviewHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *SecurityReviewHandler) SetSignature(signature core.Signature) {}
func (h *SecurityReviewHandler) SetLLM(llm core.LLM)                   {}

func (h *SecurityReviewHandler) Clone() core.Module {
	return &SecurityReviewHandler{}
}

// PerformanceReviewHandler handles performance-focused review.
type PerformanceReviewHandler struct{}

func (h *PerformanceReviewHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert inputs for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := h.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (h *PerformanceReviewHandler) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	reviewContext, ok := event.Data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid review context")
	}

	prEvent := reviewContext["pr_event"].(GitHubPREvent)

	fmt.Printf("⚡ Performance Reviewer: Analyzing PR #%d for optimization opportunities\n", prEvent.PRNumber)

	// Simulate performance analysis
	time.Sleep(2500 * time.Millisecond)

	result := ReviewResult{
		AgentID:    "performance_reviewer",
		PRNumber:   prEvent.PRNumber,
		Approved:   true,
		Confidence: 0.78,
		Issues: []ReviewIssue{
			{
				Severity:    "minor",
				Category:    "performance",
				File:        "processor.go",
				Line:        134,
				Description: "Nested loop could be optimized with better algorithm",
			},
		},
		Suggestions: []ReviewSuggestion{
			{
				File:        "cache.go",
				Line:        67,
				Description: "Consider using sync.Pool for object reuse",
				Code:        "pool := &sync.Pool{New: func() interface{} { return &Object{} }}",
			},
		},
	}

	fmt.Printf("   ⚡ Performance review completed: %d optimization opportunities\n", len(result.Issues))

	return map[string]interface{}{
		"processed": true,
		"result":    result,
	}, nil
}

func (h *PerformanceReviewHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *PerformanceReviewHandler) SetSignature(signature core.Signature) {}
func (h *PerformanceReviewHandler) SetLLM(llm core.LLM)                   {}

func (h *PerformanceReviewHandler) Clone() core.Module {
	return &PerformanceReviewHandler{}
}

// StyleReviewHandler handles code style review.
type StyleReviewHandler struct{}

func (h *StyleReviewHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert inputs for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := h.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (h *StyleReviewHandler) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	reviewContext, ok := event.Data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid review context")
	}

	prEvent := reviewContext["pr_event"].(GitHubPREvent)

	fmt.Printf("✨ Style Reviewer: Checking PR #%d for style compliance\n", prEvent.PRNumber)

	// Simulate style checking
	time.Sleep(1 * time.Second) // Style checks are usually fast

	result := ReviewResult{
		AgentID:    "style_reviewer",
		PRNumber:   prEvent.PRNumber,
		Approved:   true,
		Confidence: 0.92,
		Issues: []ReviewIssue{
			{
				Severity:    "minor",
				Category:    "style",
				File:        "utils.go",
				Line:        89,
				Description: "Line exceeds 120 character limit",
			},
		},
		Suggestions: []ReviewSuggestion{
			{
				File:        "main.go",
				Line:        1,
				Description: "Add package documentation",
				Code:        "// Package main provides GitHub PR review automation",
			},
		},
	}

	fmt.Printf("   ✨ Style review completed: %d style issues\n", len(result.Issues))

	return map[string]interface{}{
		"processed": true,
		"result":    result,
	}, nil
}

func (h *StyleReviewHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *StyleReviewHandler) SetSignature(signature core.Signature) {}
func (h *StyleReviewHandler) SetLLM(llm core.LLM)                   {}

func (h *StyleReviewHandler) Clone() core.Module {
	return &StyleReviewHandler{}
}

// ReviewCompletedHandler aggregates individual review results.
type ReviewCompletedHandler struct{}

func (h *ReviewCompletedHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// In a real implementation, this would aggregate results from all specialists
	// and determine the final review status
	return map[string]any{"processed": true}, nil
}

func (h *ReviewCompletedHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *ReviewCompletedHandler) SetSignature(signature core.Signature) {}
func (h *ReviewCompletedHandler) SetLLM(llm core.LLM)                   {}

func (h *ReviewCompletedHandler) Clone() core.Module {
	return &ReviewCompletedHandler{}
}

// FinalReviewHandler creates the final review summary.
type FinalReviewHandler struct{}

func (h *FinalReviewHandler) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// In a real implementation, this would create the final PR review comment
	// and update the PR status
	return map[string]any{"processed": true}, nil
}

func (h *FinalReviewHandler) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (h *FinalReviewHandler) SetSignature(signature core.Signature) {}
func (h *FinalReviewHandler) SetLLM(llm core.LLM)                   {}

func (h *FinalReviewHandler) Clone() core.Module {
	return &FinalReviewHandler{}
}

// determineRequiredSpecialists analyzes a PR to determine which specialists are needed.
func determineRequiredSpecialists(pr GitHubPREvent) []string {
	specialists := []string{"primary"} // Always need primary review

	// Check for security-sensitive files
	for _, file := range pr.Files {
		if containsSecuritySensitiveCode(file) {
			specialists = append(specialists, "security")
			break
		}
	}

	// Check for performance-critical files
	for _, file := range pr.Files {
		if isPerformanceCritical(file) {
			specialists = append(specialists, "performance")
			break
		}
	}

	// Always check style for Go files
	for _, file := range pr.Files {
		if strings.HasSuffix(file.Filename, ".go") {
			specialists = append(specialists, "style")
			break
		}
	}

	return specialists
}

// containsSecuritySensitiveCode checks if a file contains security-sensitive code.
func containsSecuritySensitiveCode(file GitHubFile) bool {
	securityKeywords := []string{"auth", "password", "token", "crypto", "security", "sql"}

	for _, keyword := range securityKeywords {
		if contains(file.Filename, keyword) || contains(file.Content, keyword) {
			return true
		}
	}

	return false
}

// isPerformanceCritical checks if a file is performance-critical.
func isPerformanceCritical(file GitHubFile) bool {
	performanceKeywords := []string{"loop", "algorithm", "cache", "optimization", "benchmark"}

	for _, keyword := range performanceKeywords {
		if contains(file.Content, keyword) {
			return true
		}
	}

	// Large files might need performance review
	return file.Changes > 100
}

// contains checks if a string contains a substring (case insensitive).
func contains(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

// simulateGitHubEvents creates mock GitHub PR events for demonstration.
func simulateGitHubEvents(orchestrator *communication.Agent, ctx context.Context) {
	time.Sleep(2 * time.Second) // Wait for agents to start

	// Simulate first PR
	fmt.Println("\n🎭 Simulating GitHub PR events...")

	pr1 := GitHubPREvent{
		Action:     "opened",
		PRNumber:   123,
		Repository: "company/awesome-service",
		Author:     "developer123",
		Title:      "Add user authentication system",
		Body:       "This PR adds JWT-based authentication with role-based access control.",
		Branch:     "feature/auth-system",
		Files: []GitHubFile{
			{
				Filename: "auth.go",
				Status:   "added",
				Changes:  85,
				Content:  "package auth\n\nimport \"crypto/jwt\"\n\nfunc authenticateUser(token string) {...}",
			},
			{
				Filename: "config.go",
				Status:   "modified",
				Changes:  12,
				Content:  "const API_KEY = \"hardcoded-secret-key-123\"\n",
			},
			{
				Filename: "main.go",
				Status:   "modified",
				Changes:  8,
				Content:  "func main() {\n    // TODO: Add authentication\n}",
			},
		},
	}

	// Emit PR opened event
	event1 := workflows.Event{
		ID:        "github_pr_123",
		Type:      "pr_opened",
		Data:      pr1,
		Priority:  5,
		Timestamp: time.Now(),
		Context: map[string]interface{}{
			"repository": pr1.Repository,
			"author":     pr1.Author,
		},
	}

	// Simulate PR processing by the orchestrator
	fmt.Printf("📨 Processing PR event: %s\n", event1.ID)

	// Wait and simulate another PR
	time.Sleep(8 * time.Second)

	pr2 := GitHubPREvent{
		Action:     "opened",
		PRNumber:   124,
		Repository: "company/awesome-service",
		Author:     "junior-dev",
		Title:      "Fix typo in README",
		Body:       "Corrects a spelling mistake in the installation instructions.",
		Branch:     "fix/readme-typo",
		Files: []GitHubFile{
			{
				Filename: "README.md",
				Status:   "modified",
				Changes:  2,
				Content:  "# Awesome Service\n\nFixed: instalation -> installation",
			},
		},
	}

	event2 := workflows.Event{
		ID:        "github_pr_124",
		Type:      "pr_opened",
		Data:      pr2,
		Priority:  3,
		Timestamp: time.Now(),
		Context: map[string]interface{}{
			"repository": pr2.Repository,
			"author":     pr2.Author,
		},
	}

	// Simulate PR processing by the orchestrator
	fmt.Printf("📨 Processing PR event: %s\n", event2.ID)
}
