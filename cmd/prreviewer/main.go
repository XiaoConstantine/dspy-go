package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
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

// NewPRReviewAgent creates a new PR review agent.
func NewPRReviewAgent() (*PRReviewAgent, error) {
	memory := agents.NewInMemoryStore()

	// Configure the analyzer for PR review tasks
	analyzerConfig := agents.AnalyzerConfig{
		BaseInstruction: `Analyze the PR changes and break down the code review into focused tasks.
		For each file that needs review, create a separate task that examines:

		IMPORTANT FORMAT RULES:

		1. Start fields exactly with 'analysis:' or 'tasks:' (no markdown formatting)
		2. Provide raw XML directly after 'tasks:' without any wrapping
		3. Keep the exact field prefix format - no decorations or modifications
		4. Ensure proper indentation and structure in the XML

		Consider:
		- Code style and readability
		- Function and variable naming
		- Code organization
		- Error handling patterns
		- Documentation quality

		Each file should be reviewed independently, with higher priority given to files
    with more changes or core functionality.`,
		FormatInstructions: `Format tasks in XML with one task per file:
    <tasks>
        <task id="review_{file_path}" type="code_review" processor="code_review" priority="1">
            <description>Review {file_path} for code quality</description>
            <metadata>
                <item key="file_path">{file_path}</item>
                <item key="file_content">{file_content}</item>
                <item key="changes">{changes}</item>
                <item key="category">code_review</item>
            </metadata>
        </task>
    </tasks>`,
	}

	config := agents.OrchestrationConfig{
		MaxConcurrent:  5,
		TaskParser:     &agents.XMLTaskParser{},
		PlanCreator:    &agents.DependencyPlanCreator{},
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
	fileData := make(map[string]map[string]interface{})
	for _, task := range tasks {
		fileData[task.FilePath] = map[string]interface{}{
			"file_content": task.FileContent,
			"changes":      task.Changes,
		}
	}
	reviewContext := map[string]interface{}{
		"files":       fileData,
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
    Analyze the code for:
    1. Code Style and Readability:
       - Clear and consistent formatting
       - Meaningful variable and function names
       - Code complexity and readability
    
    2. Code Structure:
       - Function size and responsibility
       - Code organization and modularity
       - Interface design and abstraction
    
    3. Error Handling:
       - Comprehensive error cases
       - Proper error propagation
       - Meaningful error messages
    
    4. Documentation:
       - Function and type documentation
       - Important logic explanation
       - Usage examples where needed

    Provide comments in this format:
    - Each comment should be specific and actionable
    - Include line numbers where applicable
    - Suggest improvements with example code when helpful
    - Prioritize major issues over minor style concerns
	`)

	// Create predict module for review
	predict := modules.NewPredict(signature)

	metadata, err := extractReviewMetadata(task.Metadata)
	if err != nil {
		return nil, fmt.Errorf("task %s: %w", task.ID, err)
	}
	if metadata.FileContent == "" && metadata.Changes == "" {
		return nil, fmt.Errorf("both file content and changes cannot be empty for file %s", metadata.FilePath)
	}
	logger.Debug(ctx, "Extracted metadata for task %s: file_path=%s, content_length=%d",
		task.ID, metadata.FilePath, len(metadata.FileContent))
	// Process the review
	result, err := predict.Process(ctx, map[string]interface{}{
		"file_content": metadata.FileContent,
		"changes":      metadata.Changes,
	})
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
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

func extractReviewMetadata(metadata map[string]interface{}) (*ReviewMetadata, error) {
	logger := logging.GetLogger()
	rm := &ReviewMetadata{}
	logger.Info(context.Background(), "Meta: %v", metadata)

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

	if fileContent, ok := metadata["file_content"]; ok {
		if str, ok := fileContent.(string); ok {

			logger.Info(context.Background(), "file content: %v", str)
			rm.FileContent = str
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

	if *githubToken == "" || *owner == "" || *repo == "" || *prNumber == 0 {
		fmt.Println("Missing required flags. Please provide:")
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

	err := VerifyTokenPermissions(ctx, *githubToken, *owner, *repo)
	if err != nil {
		logger.Error(ctx, "Token permission verification failed: %v", err)
		os.Exit(1)
	}

	if *verifyOnly {
		os.Exit(0)
	}
	llms.EnsureFactory()

	err = core.ConfigureDefaultLLM(*apiKey, "ollama:deepseek-r1:14b-qwen-distill-q4_K_M")
	//err = core.ConfigureDefaultLLM(*apiKey, "ollama:")
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
	if err != nil {
		logger.Error(ctx, "Failed to get PR changes: %v", err)
		os.Exit(1)
	}
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
	//logger.Info(ctx, "tasks: %v", tasks)

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
