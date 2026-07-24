package oauth

import (
	"bufio"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

var (
	pkceGenerator                    = GeneratePKCE
	oauthStateGenerator              = GenerateState
	anthropicAuthURL                 = GetAuthorizationURL
	anthropicCodeExchanger           = ExchangeCode
	openAIAuthURL                    = GetOpenAIAuthorizationURLWithState
	openAICodeExchanger              = ExchangeOpenAICode
	browserOpener                    = openBrowser
	currentGOOS                      = runtime.GOOS
	execCommand                      = exec.Command
	commandStarter                   = func(cmd *exec.Cmd) error { return cmd.Start() }
	loginInputReader       io.Reader = os.Stdin
)

// LoginAnthropic performs interactive OAuth login for Claude Max/Pro subscriptions.
// It opens a browser for authentication and returns the access token.
func LoginAnthropic() (*TokenResponse, error) {
	// Generate PKCE
	verifier, challenge, err := pkceGenerator()
	if err != nil {
		return nil, fmt.Errorf("failed to generate PKCE: %w", err)
	}

	// Get authorization URL
	authURL := anthropicAuthURL(verifier, challenge)

	// Open browser
	fmt.Println("Opening browser for Claude authentication...")
	fmt.Printf("If browser doesn't open, visit:\n%s\n\n", authURL)
	browserOpener(authURL)

	// Prompt for code
	fmt.Print("After authorizing, paste the code from the redirect URL: ")
	reader := bufio.NewReader(loginInputReader)
	input, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read input: %w", err)
	}
	code := strings.TrimSpace(input)

	// Extract code if user pasted full URL or code#state format
	code = extractCode(code)

	// Exchange for token
	fmt.Println("Exchanging code for token...")
	tokens, err := anthropicCodeExchanger(code, verifier)
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
func extractCodeAndState(input string) (code, state string) {
	if parsed, err := url.Parse(input); err == nil && parsed.Scheme != "" && parsed.Host != "" {
		return parsed.Query().Get("code"), parsed.Query().Get("state")
	}
	if strings.Contains(input, "#") {
		parts := strings.SplitN(input, "#", 2)
		return parts[0], parts[1]
	}
	if values, err := url.ParseQuery(input); err == nil && values.Has("code") {
		return values.Get("code"), values.Get("state")
	}
	return input, ""
}

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
	// Generate independent PKCE and CSRF values. The verifier must never be
	// placed in the authorization URL or browser history.
	verifier, challenge, err := pkceGenerator()
	if err != nil {
		return nil, fmt.Errorf("failed to generate PKCE: %w", err)
	}
	state, err := oauthStateGenerator()
	if err != nil {
		return nil, fmt.Errorf("failed to generate OAuth state: %w", err)
	}

	// Get authorization URL
	authURL := openAIAuthURL(challenge, state)

	// Open browser
	fmt.Println("Opening browser for OpenAI authentication...")
	fmt.Printf("If browser doesn't open, visit:\n%s\n\n", authURL)
	browserOpener(authURL)

	// Prompt for code
	fmt.Print("After authorizing, paste the code from the redirect URL: ")
	reader := bufio.NewReader(loginInputReader)
	input, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read input: %w", err)
	}
	code, returnedState := extractCodeAndState(strings.TrimSpace(input))
	if returnedState != state {
		return nil, fmt.Errorf("OpenAI OAuth state mismatch")
	}
	if code == "" {
		return nil, fmt.Errorf("OpenAI OAuth callback did not include a code")
	}

	// Exchange for token
	fmt.Println("Exchanging code for token...")
	tokens, err := openAICodeExchanger(code, verifier)
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
	switch currentGOOS {
	case "darwin":
		cmd = execCommand("open", url)
	case "linux":
		cmd = execCommand("xdg-open", url)
	case "windows":
		cmd = execCommand("rundll32", "url.dll,FileProtocolHandler", url)
	}
	if cmd != nil {
		_ = commandStarter(cmd)
	}
}
