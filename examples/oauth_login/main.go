// Package main provides an interactive CLI for OAuth login to Claude Max/Pro.
package main

import (
	"fmt"
	"os"

	"github.com/XiaoConstantine/dspy-go/pkg/llms/oauth"
)

func main() {
	fmt.Println("Claude Max/Pro OAuth Login")
	fmt.Println("==========================")
	fmt.Println()
	fmt.Println("This utility will help you obtain an OAuth token for Claude Max/Pro subscriptions.")
	fmt.Println("The token can be used with dspy-go by setting ANTHROPIC_OAUTH_TOKEN.")
	fmt.Println()

	if _, err := oauth.LoginAnthropic(); err != nil {
		fmt.Fprintf(os.Stderr, "Login failed: %v\n", err)
		os.Exit(1)
	}
}
