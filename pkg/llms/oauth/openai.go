package oauth

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

const (
	// OpenAI OAuth endpoints.
	OpenAIAuthorizationURL = "https://auth.openai.com/oauth/authorize"
	OpenAITokenURL         = "https://auth.openai.com/oauth/token"
	OpenAIRedirectURI      = "http://localhost:1455/auth/callback"
	// OpenAI Codex CLI public client ID.
	OpenAIClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	// Scopes for ChatGPT Plus/Pro subscription access.
	OpenAIScopes = "openid profile email offline_access"
)

// OpenAITokenResponse represents the OAuth token response from OpenAI.
type OpenAITokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
	Scope        string `json:"scope"`
}

// GetOpenAIAuthorizationURL returns the URL for user to authorize with OpenAI.
func GetOpenAIAuthorizationURL(verifier, challenge string) string {
	params := url.Values{
		"response_type":         {"code"},
		"client_id":             {OpenAIClientID},
		"redirect_uri":          {OpenAIRedirectURI},
		"scope":                 {OpenAIScopes},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
		"state":                 {verifier},
	}
	return OpenAIAuthorizationURL + "?" + params.Encode()
}

// ExchangeOpenAICode exchanges an authorization code for tokens.
func ExchangeOpenAICode(code, verifier string) (*OpenAITokenResponse, error) {
	payload := map[string]string{
		"grant_type":    "authorization_code",
		"client_id":     OpenAIClientID,
		"code":          code,
		"redirect_uri":  OpenAIRedirectURI,
		"code_verifier": verifier,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := http.Post(OpenAITokenURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token exchange failed: %s", resp.Status)
	}

	var result OpenAITokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// RefreshOpenAIAccessToken refreshes an expired access token.
func RefreshOpenAIAccessToken(refreshToken string) (*OpenAITokenResponse, error) {
	payload := map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     OpenAIClientID,
		"refresh_token": refreshToken,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := http.Post(OpenAITokenURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to refresh token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token refresh failed: %s", resp.Status)
	}

	var result OpenAITokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}
