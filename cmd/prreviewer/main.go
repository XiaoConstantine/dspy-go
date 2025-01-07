package main

import (
	"context"
	"encoding/xml"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// PRReviewTask represents a single file review task.
type PRReviewTask struct {
	FilePath    string
	FileContent string
	Changes     string // Git diff content
}

// PRReviewComment represents a review comment.
type PRReviewComment struct {
	FilePath   string
	LineNumber int
	Content    string
	Severity   string
	Suggestion string
	Category   string // e.g., "security", "performance", "style"
}

// PRReviewAgent handles code review using dspy-go.
type PRReviewAgent struct {
	orchestrator *agents.FlexibleOrchestrator
	memory       agents.Memory
}
type ReviewMetadata struct {
	FilePath    string
	FileContent string
	Changes     string
	Category    string
	ReviewType  string
}

type XMLTaskParser struct {
	// Configuration for XML parsing
	RequiredFields []string
}

type XMLTask struct {
	XMLName       xml.Name    `xml:"task"`
	ID            string      `xml:"id,attr"`
	Type          string      `xml:"type,attr"`      // Make sure this maps to the type attribute
	ProcessorType string      `xml:"processor,attr"` // Make sure this maps to the processor attribute
	Priority      int         `xml:"priority,attr"`
	Description   string      `xml:"description"`
	Dependencies  []string    `xml:"dependencies>dep"` // This maps to the <dependencies><dep>...</dep></dependencies> structure
	Metadata      XMLMetadata `xml:"metadata"`
}

type XMLMetadata struct {
	Items []XMLMetadataItem `xml:"item"`
}

type XMLMetadataItem struct {
	Key   string `xml:"key,attr"`
	Value string `xml:",chardata"`
}

func (p *XMLTaskParser) Parse(analyzerOutput map[string]interface{}) ([]agents.Task, error) {
	tasksXML, ok := analyzerOutput["tasks"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid tasks format in analyzer output")
	}

	// Parse XML tasks
	var xmlTasks struct {
		Tasks []XMLTask `xml:"task"`
	}

	xmlStart := strings.Index(tasksXML, "<tasks>")
	if xmlStart == -1 {
		return nil, fmt.Errorf("no valid XML tasks found in output")
	}
	tasksXML = tasksXML[xmlStart:]

	if err := xml.Unmarshal([]byte(tasksXML), &xmlTasks); err != nil {
		return nil, fmt.Errorf("failed to parse XML tasks: %w", err)
	}

	// Convert to Task objects
	tasks := make([]agents.Task, len(xmlTasks.Tasks))
	for i, xmlTask := range xmlTasks.Tasks {
		// Validate required fields
		if err := p.validateTask(xmlTask); err != nil {
			return nil, fmt.Errorf("invalid task %s: %w", xmlTask.ID, err)
		}

		// Convert metadata to map
		metadata := make(map[string]interface{})
		for _, item := range xmlTask.Metadata.Items {
			metadata[item.Key] = item.Value
		}

		tasks[i] = agents.Task{
			ID:            xmlTask.ID,
			Type:          xmlTask.Type,
			ProcessorType: xmlTask.ProcessorType,
			Dependencies:  xmlTask.Dependencies,
			Priority:      xmlTask.Priority,
			Metadata:      metadata,
		}
	}

	return tasks, nil
}

func (p *XMLTaskParser) validateTask(task XMLTask) error {
	if task.ID == "" {
		return fmt.Errorf("missing task ID")
	}
	if task.Type == "" {
		return fmt.Errorf("missing task type")
	}
	if task.ProcessorType == "" {
		return fmt.Errorf("missing processor type")
	}
	return nil
}

// DependencyPlanCreator creates execution plans based on task dependencies.
type DependencyPlanCreator struct {
	// Optional configuration for planning
	MaxTasksPerPhase int
}

func NewDependencyPlanCreator(maxTasksPerPhase int) *DependencyPlanCreator {
	if maxTasksPerPhase <= 0 {
		maxTasksPerPhase = 10 // Default value
	}
	return &DependencyPlanCreator{
		MaxTasksPerPhase: maxTasksPerPhase,
	}
}

func (p *DependencyPlanCreator) CreatePlan(tasks []agents.Task) ([][]agents.Task, error) {
	// Build dependency graph
	graph := buildDependencyGraph(tasks)

	// Detect cycles
	if err := detectCycles(graph); err != nil {
		return nil, fmt.Errorf("invalid task dependencies: %w", err)
	}

	// Create phases based on dependencies
	phases := [][]agents.Task{}
	remaining := make(map[string]agents.Task)
	completed := make(map[string]bool)

	// Initialize remaining tasks
	for _, task := range tasks {
		remaining[task.ID] = task
	}

	// Create phases until all tasks are allocated
	for len(remaining) > 0 {
		phase := []agents.Task{}

		// Find tasks with satisfied dependencies
		for _, task := range remaining {
			if canExecute(task, completed) {
				phase = append(phase, task)
				delete(remaining, task.ID)

				// Respect max tasks per phase
				if len(phase) >= p.MaxTasksPerPhase {
					break
				}
			}
		}

		// If no tasks can be executed, we have a problem
		if len(phase) == 0 {
			return nil, fmt.Errorf("circular dependency or missing dependency detected")
		}

		// Sort phase by priority
		sort.Slice(phase, func(i, j int) bool {
			return phase[i].Priority < phase[j].Priority
		})

		phases = append(phases, phase)

		// Mark phase tasks as completed
		for _, task := range phase {
			completed[task.ID] = true
		}
	}

	return phases, nil
}

// Helper function to build dependency graph.
func buildDependencyGraph(tasks []agents.Task) map[string][]string {
	graph := make(map[string][]string)
	for _, task := range tasks {
		graph[task.ID] = task.Dependencies
	}
	return graph
}

// Helper function to detect cycles in the dependency graph.
func detectCycles(graph map[string][]string) error {
	visited := make(map[string]bool)
	path := make(map[string]bool)

	var checkCycle func(string) error
	checkCycle = func(node string) error {
		visited[node] = true
		path[node] = true

		for _, dep := range graph[node] {
			if !visited[dep] {
				if err := checkCycle(dep); err != nil {
					return err
				}
			} else if path[dep] {
				return fmt.Errorf("cycle detected involving task %s", node)
			}
		}

		path[node] = false
		return nil
	}

	for node := range graph {
		if !visited[node] {
			if err := checkCycle(node); err != nil {
				return err
			}
		}
	}

	return nil
}

// Helper function to check if a task can be executed.
func canExecute(task agents.Task, completed map[string]bool) bool {
	for _, dep := range task.Dependencies {
		if !completed[dep] {
			return false
		}
	}
	return true
}

// NewPRReviewAgent creates a new PR review agent.
func NewPRReviewAgent() (*PRReviewAgent, error) {
	memory := agents.NewInMemoryStore()

	// Configure the analyzer for PR review tasks
	analyzerConfig := agents.AnalyzerConfig{
		BaseInstruction: `Analyze the PR changes and break down the review into focused tasks.
		Consider:
		- Code quality and style
		- Potential bugs and edge cases
		- Security implications
		- Performance considerations
		- Testing coverage`,
		FormatInstructions: `Format tasks in XML with clear dependencies and metadata:
		<tasks>
			<task id="task_1" type="review" processor="code_review" priority="1">
				<description>Review file for code quality and style</description>
				<metadata>
					<item key="file_path">{file_path}</item>
					<item key="category">style</item>
				</metadata>
			</task>
		</tasks>`,
	}

	config := agents.OrchestrationConfig{
		MaxConcurrent:  5,
		TaskParser:     &XMLTaskParser{},
		PlanCreator:    &DependencyPlanCreator{},
		AnalyzerConfig: analyzerConfig,
		CustomProcessors: map[string]agents.TaskProcessor{
			"code_review": &CodeReviewProcessor{},
		},
	}

	orchestrator := agents.NewFlexibleOrchestrator(memory, config)

	return &PRReviewAgent{
		orchestrator: orchestrator,
		memory:       memory,
	}, nil
}

// ReviewPR reviews a complete pull request.
func (a *PRReviewAgent) ReviewPR(ctx context.Context, tasks []PRReviewTask) ([]PRReviewComment, error) {
	// Create review context
	reviewContext := map[string]interface{}{
		"tasks":       tasks,
		"review_type": "pull_request",
	}

	// Execute orchestrated review
	result, err := a.orchestrator.Process(ctx, "Review pull request changes", reviewContext)
	if err != nil {
		return nil, fmt.Errorf("failed to process PR review: %w", err)
	}

	// Collect and format comments
	comments := make([]PRReviewComment, 0)
	for taskID, taskResult := range result.CompletedTasks {
		logging.GetLogger().Info(ctx, "Processing task: %s", taskID)

		if reviewComments, ok := taskResult.([]PRReviewComment); ok {
			comments = append(comments, reviewComments...)
		}
	}

	return comments, nil
}

// CodeReviewProcessor implements the core review logic.
type CodeReviewProcessor struct{}

func (p *CodeReviewProcessor) Process(ctx context.Context, task agents.Task, context map[string]interface{}) (interface{}, error) {
	logger := logging.GetLogger()
	// Create signature for code review
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "file_content"}},
			{Field: core.Field{Name: "changes"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "comments"}},
			{Field: core.Field{Name: "summary"}},
		},
	).WithInstruction(`Review the code changes and provide specific, actionable feedback.
	Consider:
	- Code style and formatting
	- Function and variable naming
	- Code organization and structure
	- Error handling
	- Documentation completeness
	`)

	// Create predict module for review
	predict := modules.NewPredict(signature)

	metadata, err := extractReviewMetadata(task.Metadata)
	if err != nil {
		return nil, fmt.Errorf("task %s: %w", task.ID, err)
	}

	logger.Debug(ctx, "Extracted metadata for task %s: file_path=%s, content_length=%d",
		task.ID, metadata.FilePath, len(metadata.FileContent))
	// Process the review
	result, err := predict.Process(ctx, map[string]interface{}{
		"file_content": metadata.FileContent,
		"changes":      metadata.Changes,
	})
	if err != nil {
		return nil, err
	}

	// Parse and format comments
	comments, err := extractComments(result, metadata.FilePath)

	if err != nil {
		return nil, fmt.Errorf("failed to parse comments for task %s: %w", task.ID, err)
	}

	logger.Debug(ctx, "Successfully processed review for task %s with %d comments",
		task.ID, len(comments))

	return comments, nil
}

// Helper functions.
func parseReviewComments(filePath string, commentsStr string) ([]PRReviewComment, error) {
	// Parse the comments string into structured comments
	// This is a placeholder - actual implementation would parse the
	// LLM's output format into PRReviewComment structs
	return []PRReviewComment{}, nil
}

func extractComments(result map[string]interface{}, filePath string) ([]PRReviewComment, error) {
	commentsRaw, exists := result["comments"]
	if !exists {
		return nil, fmt.Errorf("prediction result missing 'comments' field")
	}

	commentsStr, ok := commentsRaw.(string)
	if !ok {
		return nil, fmt.Errorf("comments must be string, got %T", commentsRaw)
	}

	return parseReviewComments(filePath, commentsStr)
}
func determineReviewType(category string) string {
	// Categories that require file-specific review
	fileReviewCategories := map[string]bool{
		"style": true,
		"bugs":  true,
	}

	if fileReviewCategories[category] {
		return "file"
	}
	return "general"
}

func extractReviewMetadata(metadata map[string]interface{}) (*ReviewMetadata, error) {
	rm := &ReviewMetadata{}

	// Extract category (always required)
	categoryRaw, exists := metadata["category"]
	if !exists {
		return nil, fmt.Errorf("missing required field 'category' in metadata")
	}
	category, ok := categoryRaw.(string)
	if !ok {
		return nil, fmt.Errorf("field 'category' must be string, got %T", categoryRaw)
	}
	rm.Category = category

	// Determine review type based on category
	rm.ReviewType = determineReviewType(category)

	// For file-specific reviews
	if rm.ReviewType == "file" {
		// Extract file path (required for file reviews)
		filePathRaw, exists := metadata["file_path"]
		if !exists {
			return nil, fmt.Errorf("missing required field 'file_path' for file review")
		}
		filePath, ok := filePathRaw.(string)
		if !ok {
			return nil, fmt.Errorf("field 'file_path' must be string, got %T", filePathRaw)
		}
		rm.FilePath = filePath

		// Extract changes (required for file reviews)
		changesRaw, exists := metadata["changes"]
		if !exists {
			return nil, fmt.Errorf("missing required field 'changes' for file review")
		}
		changes, ok := changesRaw.(string)
		if !ok {
			return nil, fmt.Errorf("field 'changes' must be string, got %T", changesRaw)
		}
		rm.Changes = changes

		// Extract file content (optional but recommended for file reviews)
		if fileContent, ok := metadata["file_content"]; ok {
			if str, ok := fileContent.(string); ok {
				rm.FileContent = str
			}
		}
	}

	return rm, nil
}

func main() {
	ctx := core.WithExecutionState(context.Background())
	apiKey := flag.String("api-key", "", "Anthropic API Key")
	githubToken := flag.String("github-token", os.Getenv("GITHUB_TOKEN"), "GitHub Token")
	owner := flag.String("owner", "", "Repository owner")
	repo := flag.String("repo", "", "Repository name")
	prNumber := flag.Int("pr", 0, "Pull Request number")
	debug := flag.Bool("debug", false, "Enable debug logging")
	verifyOnly := flag.Bool("verify-only", false, "Only verify token permissions without running review")

	flag.Parse()

	if *apiKey == "" || *githubToken == "" || *owner == "" || *repo == "" || *prNumber == 0 {
		fmt.Println("Missing required flags. Please provide:")
		fmt.Println("  -api-key or set ANTHROPIC_API_KEY")
		fmt.Println("  -github-token or set GITHUB_TOKEN")
		fmt.Println("  -owner (repository owner)")
		fmt.Println("  -repo (repository name)")
		fmt.Println("  -pr (pull request number)")
		os.Exit(1)
	}
	logLevel := logging.INFO
	if *debug {
		logLevel = logging.DEBUG
	}

	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logLevel,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	// err := VerifyTokenPermissions(ctx, *githubToken, *owner, *repo)
	// if err != nil {
	// 	logger.Error(ctx, "Token permission verification failed: %v", err)
	// 	os.Exit(1)
	// }

	if *verifyOnly {
		os.Exit(0)
	}
	err := config.ConfigureDefaultLLM(*apiKey, core.ModelAnthropicSonnet)
	if err != nil {
		logger.Error(ctx, "Failed to configure LLM: %v", err)
	}

	agent, err := NewPRReviewAgent()
	if err != nil {
		panic(err)
	}

	githubTools := NewGitHubTools(*githubToken, *owner, *repo)
	logger.Info(ctx, "Fetching changes for PR #%d", *prNumber)
	changes, err := githubTools.GetPullRequestChanges(ctx, *prNumber)
	tasks := make([]PRReviewTask, 0, len(changes.Files))
	for _, file := range changes.Files {
		// Log file being processed
		logger.Info(ctx, "Processing file: %s (+%d/-%d lines)",
			file.FilePath,
			file.Additions,
			file.Deletions,
		)

		tasks = append(tasks, PRReviewTask{
			FilePath:    file.FilePath,
			FileContent: file.FileContent,
			Changes:     file.Patch,
		})
	}
	if err != nil {
		logger.Error(ctx, "Failed to get PR changes: %v", err)
		os.Exit(1)
	}

	logger.Info(ctx, "Starting code review for %d files", len(tasks))
	comments, err := agent.ReviewPR(ctx, tasks)
	if err != nil {
		logger.Error(ctx, "Failed to review PR: %v", err)
		os.Exit(1)
	}
	categoryCounts := make(map[string]int)
	severityCounts := make(map[string]int)
	for _, comment := range comments {
		categoryCounts[comment.Category]++
		severityCounts[comment.Severity]++
	}

	logger.Info(ctx, "Review completed: %d comments generated", len(comments))
	logger.Info(ctx, "Categories:")
	for category, count := range categoryCounts {
		logger.Info(ctx, "  - %s: %d", category, count)
	}
	logger.Info(ctx, "Severities:")
	for severity, count := range severityCounts {
		logger.Info(ctx, "  - %s: %d", severity, count)
	}

	// Post review comments
	// Post comments back to GitHub
	for _, comment := range comments {
		// Use GitHub API to post review comments
		fmt.Printf("Review comment for %s:%d: %s\n",
			comment.FilePath,
			comment.LineNumber,
			comment.Content)
	}

	logger.Info(ctx, "Posting review comments to GitHub")
	// err = githubTools.CreateReviewComments(ctx, *prNumber, comments)
	// if err != nil {
	// 	logger.Error(ctx, "Failed to post review comments: %v", err)
	// 	os.Exit(1)
	// }

	logger.Info(ctx, "Successfully completed PR review")
}
