package rlm

import (
	"regexp"
	"strings"
)

var (
	// goCodeBlockRe matches ```go or ```repl code blocks.
	// (?s) enables DOTALL mode so . matches newlines.
	goCodeBlockRe = regexp.MustCompile("(?s)```(?:go|repl)\\s*\\n(.*?)\\n```")

	// finalVarRe matches FINAL_VAR(identifier) at start of line.
	finalVarRe = regexp.MustCompile(`(?m)^\s*FINAL_VAR\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)`)

	// finalRe matches FINAL(...) at start of line.
	finalRe = regexp.MustCompile(`(?m)^\s*FINAL\((.+?)\)\s*$`)
)

// FinalAnswerType indicates whether the answer is direct or a variable reference.
type FinalAnswerType string

const (
	// FinalTypeDirect indicates a direct value like FINAL(42).
	FinalTypeDirect FinalAnswerType = "FINAL"
	// FinalTypeVariable indicates a variable reference like FINAL_VAR(answer).
	FinalTypeVariable FinalAnswerType = "FINAL_VAR"
)

// FinalAnswer represents a detected FINAL or FINAL_VAR signal.
type FinalAnswer struct {
	Type    FinalAnswerType
	Content string
}

// FindCodeBlocks extracts all ```go or ```repl code blocks from the LLM response.
func FindCodeBlocks(text string) []string {
	matches := goCodeBlockRe.FindAllStringSubmatch(text, -1)
	if matches == nil {
		return nil
	}

	results := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			code := strings.TrimSpace(match[1])
			if code != "" {
				results = append(results, code)
			}
		}
	}
	return results
}

// FindFinalAnswer detects FINAL() or FINAL_VAR() signals in the LLM response.
// Returns nil if no final answer is found.
func FindFinalAnswer(text string) *FinalAnswer {
	// Check FINAL_VAR first (more specific pattern)
	if match := finalVarRe.FindStringSubmatch(text); match != nil {
		return &FinalAnswer{
			Type:    FinalTypeVariable,
			Content: strings.TrimSpace(match[1]),
		}
	}

	// Check FINAL
	if match := finalRe.FindStringSubmatch(text); match != nil {
		content := strings.TrimSpace(match[1])
		content = stripQuotes(content)
		return &FinalAnswer{
			Type:    FinalTypeDirect,
			Content: content,
		}
	}

	return nil
}

// stripQuotes removes surrounding quotes from a string if present.
func stripQuotes(s string) string {
	if len(s) >= 2 {
		first, last := s[0], s[len(s)-1]
		if (first == '"' && last == '"') ||
			(first == '\'' && last == '\'') ||
			(first == '`' && last == '`') {
			return s[1 : len(s)-1]
		}
	}
	return s
}
