// Package main provides an interactive CLI for OAuth login to AI provider subscriptions.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/llms/oauth"
)

func main() {
	fmt.Println("AI Provider OAuth Login")
	fmt.Println("=======================")
	fmt.Println()
	fmt.Println("This utility helps you obtain OAuth tokens for AI provider subscriptions.")
	fmt.Println()
	fmt.Println("Select provider:")
	fmt.Println("  1. Anthropic (Claude Max/Pro)")
	fmt.Println("  2. OpenAI (ChatGPT Plus/Pro)")
	fmt.Println()
	fmt.Print("Enter choice (1 or 2): ")

	reader := bufio.NewReader(os.Stdin)
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)

	switch choice {
	case "1":
		loginAnthropic()
	case "2":
		loginOpenAI()
	default:
		fmt.Println("Invalid choice. Please enter 1 or 2.")
		os.Exit(1)
	}
}

func loginAnthropic() {
	fmt.Println()
	fmt.Println("Anthropic Claude Max/Pro OAuth Login")
	fmt.Println("=====================================")
	fmt.Println()
	fmt.Println("The token can be used with dspy-go by setting ANTHROPIC_OAUTH_TOKEN.")
	fmt.Println()

	if _, err := oauth.LoginAnthropic(); err != nil {
		fmt.Fprintf(os.Stderr, "Login failed: %v\n", err)
		os.Exit(1)
	}
}

func loginOpenAI() {
	fmt.Println()
	fmt.Println("OpenAI ChatGPT Plus/Pro OAuth Login")
	fmt.Println("====================================")
	fmt.Println()
	fmt.Println("The token can be used with dspy-go by setting OPENAI_OAUTH_TOKEN.")
	fmt.Println()

	if _, err := oauth.LoginOpenAI(); err != nil {
		fmt.Fprintf(os.Stderr, "Login failed: %v\n", err)
		os.Exit(1)
	}
}
