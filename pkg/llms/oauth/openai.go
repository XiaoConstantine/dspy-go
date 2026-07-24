package oauth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
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
	IDToken      string `json:"id_token,omitempty"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
	Scope        string `json:"scope"`
}

// GetOpenAIAuthorizationURL returns the URL for user to authorize with OpenAI.
// Deprecated: use GetOpenAIAuthorizationURLWithState with an independently
// generated state value. The challenge is used as a non-secret compatibility
// state here so this legacy API never exposes the PKCE verifier.
func GetOpenAIAuthorizationURL(_verifier, challenge string) string {
	return GetOpenAIAuthorizationURLWithState(challenge, challenge)
}

// GetOpenAIAuthorizationURLWithState returns an OpenAI Codex authorization URL.
// State must be independently generated and validated by the caller.
func GetOpenAIAuthorizationURLWithState(challenge, state string) string {
	params := url.Values{
		"response_type":              {"code"},
		"client_id":                  {OpenAIClientID},
		"redirect_uri":               {OpenAIRedirectURI},
		"scope":                      {OpenAIScopes},
		"code_challenge":             {challenge},
		"code_challenge_method":      {"S256"},
		"state":                      {state},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
		"originator":                 {"dspy-go"},
	}
	return OpenAIAuthorizationURL + "?" + params.Encode()
}

// ExchangeOpenAICode exchanges an authorization code for tokens.
func ExchangeOpenAICode(code, verifier string) (*OpenAITokenResponse, error) {
	payload := url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {OpenAIClientID},
		"code":          {code},
		"redirect_uri":  {OpenAIRedirectURI},
		"code_verifier": {verifier},
	}

	resp, err := httpPost(OpenAITokenURL, "application/x-www-form-urlencoded", strings.NewReader(payload.Encode()))
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

// ExchangeOpenAICodeContext exchanges an authorization code with cancellation.
func ExchangeOpenAICodeContext(ctx context.Context, code, verifier string) (*OpenAITokenResponse, error) {
	payload := url.Values{
		"grant_type": {"authorization_code"}, "client_id": {OpenAIClientID},
		"code": {code}, "redirect_uri": {OpenAIRedirectURI}, "code_verifier": {verifier},
	}
	return postOpenAITokenContext(ctx, payload, "exchange")
}

// RefreshOpenAIAccessTokenContext refreshes an access token with cancellation.
func RefreshOpenAIAccessTokenContext(ctx context.Context, refreshToken string) (*OpenAITokenResponse, error) {
	payload := url.Values{
		"grant_type": {"refresh_token"}, "client_id": {OpenAIClientID}, "refresh_token": {refreshToken},
	}
	return postOpenAITokenContext(ctx, payload, "refresh")
}

func postOpenAITokenContext(ctx context.Context, payload url.Values, action string) (*OpenAITokenResponse, error) {
	resp, err := postFormContext(ctx, OpenAITokenURL, payload)
	if err != nil {
		return nil, fmt.Errorf("failed to %s token: %w", action, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token %s failed: %s", action, resp.Status)
	}
	var result OpenAITokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	return &result, nil
}

// RefreshOpenAIAccessToken refreshes an expired access token.
func RefreshOpenAIAccessToken(refreshToken string) (*OpenAITokenResponse, error) {
	payload := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {OpenAIClientID},
		"refresh_token": {refreshToken},
	}

	resp, err := httpPost(OpenAITokenURL, "application/x-www-form-urlencoded", strings.NewReader(payload.Encode()))
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
