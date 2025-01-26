package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/google/go-github/v68/github"
	"golang.org/x/oauth2"
)

// GitHubTools handles interactions with GitHub API.
type GitHubTools struct {
	client *github.Client
	owner  string
	repo   string
}

// NewGitHubTools creates a new GitHub tools instance.
func NewGitHubTools(token, owner, repo string) *GitHubTools {
	// Create an authenticated GitHub client
	ctx := context.Background()
	ts := oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: token},
	)
	tc := oauth2.NewClient(ctx, ts)
	client := github.NewClient(tc)

	return &GitHubTools{
		client: client,
		owner:  owner,
		repo:   repo,
	}
}

// PRChanges contains changes made in a pull request.
type PRChanges struct {
	Files []PRFileChange
}

// PRFileChange represents changes to a single file.
type PRFileChange struct {
	FilePath    string
	FileContent string // The complete file content
	Patch       string // The diff/patch content
	Additions   int
	Deletions   int
}

// GetPullRequestChanges retrieves the changes from a pull request.
func (g *GitHubTools) GetPullRequestChanges(ctx context.Context, prNumber int) (*PRChanges, error) {
	// Get the list of files changed in the PR
	logger := logging.GetLogger()
	files, _, err := g.client.PullRequests.ListFiles(ctx, g.owner, g.repo, prNumber, &github.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list PR files: %w", err)
	}

	logger.Debug(ctx, "Retrieved %d files from PR", len(files))
	changes := &PRChanges{
		Files: make([]PRFileChange, 0, len(files)),
	}

	for _, file := range files {
		// Skip files we don't want to review (like dependencies or generated files)
		filename := file.GetFilename()
		logger.Debug(ctx, "Processing file: %s", filename)

		if shouldSkipFile(filename) {
			logger.Debug(ctx, "Skipping file: %s (matched skip criteria)", filename)

			continue
		}
		var fileContent string

		if file.GetStatus() != "removed" {
			opts := &github.RepositoryContentGetOptions{
				Ref: fmt.Sprintf("pull/%d/head", prNumber), // This is crucial!
			}
			// Get the file content
			content, _, resp, err := g.client.Repositories.GetContents(
				ctx,
				g.owner,
				g.repo,
				file.GetFilename(),
				opts,
			)

			if err != nil {
				if resp != nil && resp.StatusCode == 404 {
					// File might have been deleted or moved
					continue
				}
				// For other errors, log but continue
				fileContent = fmt.Sprintf("Error getting content: %v", err)
			} else if content != nil {
				// Only try to get content if the content object is not nil
				if fc, err := content.GetContent(); err == nil {
					fileContent = fc
				}
			}

		}

		changes.Files = append(changes.Files, PRFileChange{
			FilePath:    file.GetFilename(),
			FileContent: fileContent,
			Patch:       file.GetPatch(),
			Additions:   file.GetAdditions(),
			Deletions:   file.GetDeletions(),
		})
	}

	if len(changes.Files) == 0 {
		return nil, fmt.Errorf("no reviewable files found in PR #%d", prNumber)
	}
	return changes, nil
}

// CreateReviewComments posts review comments back to GitHub.
func (g *GitHubTools) CreateReviewComments(ctx context.Context, prNumber int, comments []PRReviewComment) error {
	// Convert our comments into GitHub review comments
	ghComments := make([]*github.DraftReviewComment, 0, len(comments))

	for _, comment := range comments {
		// Find the line number in the diff for this comment
		position, err := g.findDiffPosition(ctx, prNumber, comment)
		if err != nil {
			continue // Skip comments we can't place in the diff
		}

		// Format the comment body with severity and category
		body := formatCommentBody(comment)

		ghComments = append(ghComments, &github.DraftReviewComment{
			Path:     &comment.FilePath,
			Position: github.Ptr(position),
			Body:     &body,
		})
	}

	// Create the review
	review := &github.PullRequestReviewRequest{
		CommitID: nil, // Will use the latest commit
		Body:     github.Ptr("Code Review Comments"),
		Event:    github.Ptr("COMMENT"),
		Comments: ghComments,
	}

	_, _, err := g.client.PullRequests.CreateReview(ctx, g.owner, g.repo, prNumber, review)
	if err != nil {
		return fmt.Errorf("failed to create review: %w", err)
	}

	return nil
}

// Helper functions

func shouldSkipFile(filename string) bool {
	// Skip common files we don't want to review
	skippedPaths := []string{
		"go.mod",
		"go.sum",
		"vendor/",
		"generated/",
		".git",
	}

	for _, path := range skippedPaths {
		if strings.Contains(filename, path) {
			return true
		}
	}

	// Skip based on file extension
	skippedExtensions := []string{
		".pb.go",  // Generated protobuf
		".gen.go", // Other generated files
		".md",     // Documentation
		".txt",    // Text files
		".yaml",   // Config files
		".yml",    // Config files
		".json",   // Config files
	}

	for _, ext := range skippedExtensions {
		if strings.HasSuffix(filename, ext) {
			return true
		}
	}

	return false
}

func (g *GitHubTools) findDiffPosition(ctx context.Context, prNumber int, comment PRReviewComment) (int, error) {
	// Get the file changes
	files, _, err := g.client.PullRequests.ListFiles(ctx, g.owner, g.repo, prNumber, &github.ListOptions{})
	if err != nil {
		return 0, err
	}

	// Find the right file
	var targetFile *github.CommitFile
	for _, file := range files {
		if file.GetFilename() == comment.FilePath {
			targetFile = file
			break
		}
	}

	if targetFile == nil {
		return 0, fmt.Errorf("file not found in PR: %s", comment.FilePath)
	}

	// Convert line number to diff position
	// This is a simplified version - in practice, you'd need to parse the patch
	// to find the exact position in the diff
	position := comment.LineNumber
	return position, nil
}

func formatCommentBody(comment PRReviewComment) string {
	var sb strings.Builder

	// Add severity indicator
	sb.WriteString(fmt.Sprintf("**%s**: ", strings.ToUpper(comment.Severity)))

	// Add the main comment
	sb.WriteString(comment.Content)

	// Add suggestion if present
	if comment.Suggestion != "" {
		sb.WriteString("\n\n**Suggestion:**\n")
		sb.WriteString(comment.Suggestion)
	}

	// Add category tag
	sb.WriteString(fmt.Sprintf("\n\n_Category: %s_", comment.Category))

	return sb.String()
}
func VerifyTokenPermissions(ctx context.Context, token, owner, repo string) error {
	// Create an authenticated client
	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token})
	tc := oauth2.NewClient(ctx, ts)
	client := github.NewClient(tc)

	// First, let's check the token's basic information
	fmt.Println("Checking token permissions...")

	// Check token validity and scopes
	user, resp, err := client.Users.Get(ctx, "") // Empty string gets authenticated user
	if err != nil {
		if resp != nil && resp.StatusCode == 401 {
			return fmt.Errorf("invalid token or token has expired")
		}
		return fmt.Errorf("error checking token: %w", err)
	}

	fmt.Printf("\nToken belongs to user: %s\n", user.GetLogin())
	fmt.Printf("Token scopes: %s\n", resp.Header.Get("X-OAuth-Scopes"))

	fmt.Printf("Checking access to repository: %s/%s\n", owner, repo)

	// Now let's check specific permissions we need
	permissionChecks := []struct {
		name  string
		check func() error
	}{
		{
			name: "Repository read access",
			check: func() error {
				_, resp, err := client.Repositories.Get(ctx, owner, repo)
				if err != nil {
					if resp != nil && resp.StatusCode == 404 {
						return fmt.Errorf("repository not found or no access")
					}
					return err
				}
				return nil
			},
		},
		{
			name: "Pull request read access",
			check: func() error {
				_, resp, err := client.PullRequests.List(ctx, owner, repo, &github.PullRequestListOptions{
					ListOptions: github.ListOptions{PerPage: 1},
				})
				if err != nil {
					if resp != nil && resp.StatusCode == 403 {
						return fmt.Errorf("no access to pull requests")
					}
					return err
				}
				return nil
			},
		},
		//		{
		// 	name: "Pull request write access (comment creation)",
		// 	check: func() error {
		// 		// Try to create a draft review to check write permissions
		// 		// We'll delete it right after
		// 		_, _, err := client.PullRequests.CreateReview(ctx, owner, repo, 1,
		// 			&github.PullRequestReviewRequest{
		// 				Body:  github.Ptr("Permission check - please ignore"),
		// 				Event: github.Ptr("COMMENT"),
		// 			})
		// 		if err != nil {
		// 			if strings.Contains(err.Error(), "403") {
		// 				return fmt.Errorf("no permission to create reviews")
		// 			}
		// 			// Don't return error if PR #1 doesn't exist
		// 			if !strings.Contains(err.Error(), "404") {
		// 				return err
		// 			}
		// 		}
		//
		// 		return nil
		// 	},
		// },
	}

	// Run all permission checks
	fmt.Println("\nPermission Check Results:")
	fmt.Println("------------------------")
	allPassed := true
	for _, check := range permissionChecks {
		fmt.Printf("%-30s: ", check.name)
		if err := check.check(); err != nil {
			fmt.Printf("❌ Failed - %v\n", err)
			allPassed = false
		} else {
			fmt.Printf("✅ Passed\n")
		}
	}

	if !allPassed {
		return fmt.Errorf("\nsome permission checks failed - token may not have sufficient access")
	}

	fmt.Println("\n✅ Token has all required permissions for PR review functionality")
	return nil
}
