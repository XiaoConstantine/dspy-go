package oauth

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// LoginAnthropic performs interactive OAuth login for Claude Max/Pro subscriptions.
// It opens a browser for authentication and returns the access token.
func LoginAnthropic() (*TokenResponse, error) {
	// Generate PKCE
	verifier, challenge, err := GeneratePKCE()
	if err != nil {
		return nil, fmt.Errorf("failed to generate PKCE: %w", err)
	}

	// Get authorization URL
	authURL := GetAuthorizationURL(verifier, challenge)

	// Open browser
	fmt.Println("Opening browser for Claude authentication...")
	fmt.Printf("If browser doesn't open, visit:\n%s\n\n", authURL)
	openBrowser(authURL)

	// Prompt for code
	fmt.Print("After authorizing, paste the code from the redirect URL: ")
	reader := bufio.NewReader(os.Stdin)
	input, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read input: %w", err)
	}
	code := strings.TrimSpace(input)

	// Extract code if user pasted full URL or code#state format
	code = extractCode(code)

	// Exchange for token
	fmt.Println("Exchanging code for token...")
	tokens, err := ExchangeCode(code, verifier)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	fmt.Println("\nSuccess! Your OAuth token:")
	fmt.Printf("export ANTHROPIC_OAUTH_TOKEN=\"%s\"\n", tokens.AccessToken)
	fmt.Printf("\nRefresh token (save for later):\n%s\n", tokens.RefreshToken)

	return tokens, nil
}

// extractCode extracts the authorization code from various input formats.
// Supports: bare code, code#state, and full URL with code parameter.
func extractCode(input string) string {
	// If it contains #, take the part before it
	if idx := strings.Index(input, "#"); idx != -1 {
		input = input[:idx]
	}

	// If it contains code=, extract the code parameter
	if strings.Contains(input, "code=") {
		parts := strings.Split(input, "code=")
		if len(parts) > 1 {
			code := parts[1]
			// Take until next & if present
			if idx := strings.Index(code, "&"); idx != -1 {
				code = code[:idx]
			}
			return code
		}
	}

	return input
}

// LoginOpenAI performs interactive OAuth login for ChatGPT Plus/Pro subscriptions.
// It opens a browser for authentication and returns the access token.
func LoginOpenAI() (*OpenAITokenResponse, error) {
	// Generate PKCE
	verifier, challenge, err := GeneratePKCE()
	if err != nil {
		return nil, fmt.Errorf("failed to generate PKCE: %w", err)
	}

	// Get authorization URL
	authURL := GetOpenAIAuthorizationURL(verifier, challenge)

	// Open browser
	fmt.Println("Opening browser for OpenAI authentication...")
	fmt.Printf("If browser doesn't open, visit:\n%s\n\n", authURL)
	openBrowser(authURL)

	// Prompt for code
	fmt.Print("After authorizing, paste the code from the redirect URL: ")
	reader := bufio.NewReader(os.Stdin)
	input, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read input: %w", err)
	}
	code := strings.TrimSpace(input)

	// Extract code if user pasted full URL
	code = extractCode(code)

	// Exchange for token
	fmt.Println("Exchanging code for token...")
	tokens, err := ExchangeOpenAICode(code, verifier)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	fmt.Println("\nSuccess! Your OAuth token:")
	fmt.Printf("export OPENAI_OAUTH_TOKEN=\"%s\"\n", tokens.AccessToken)
	fmt.Printf("\nRefresh token (save for later):\n%s\n", tokens.RefreshToken)

	return tokens, nil
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	}
	if cmd != nil {
		_ = cmd.Start()
	}
}
