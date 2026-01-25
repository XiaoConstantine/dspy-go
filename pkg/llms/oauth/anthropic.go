package oauth

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

const (
	// AuthorizationURL is the Anthropic OAuth authorization endpoint.
	AuthorizationURL = "https://claude.ai/oauth/authorize"
	// TokenURL is the Anthropic OAuth token endpoint.
	TokenURL = "https://console.anthropic.com/v1/oauth/token"
	// RedirectURI is the callback URL for the OAuth flow.
	RedirectURI = "https://console.anthropic.com/oauth/code/callback"
	// ClientID is the OAuth client ID for Claude Code.
	ClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	// Scopes defines the permissions requested.
	Scopes = "org:create_api_key user:profile user:inference"
)

// TokenResponse represents the OAuth token response from Anthropic.
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
}

// GetAuthorizationURL returns the URL for user to authorize.
func GetAuthorizationURL(verifier, challenge string) string {
	params := url.Values{
		"response_type":         {"code"},
		"client_id":             {ClientID},
		"redirect_uri":          {RedirectURI},
		"scope":                 {Scopes},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
		"state":                 {verifier},
	}
	return AuthorizationURL + "?" + params.Encode()
}

// ExchangeCode exchanges an authorization code for tokens.
func ExchangeCode(code, verifier string) (*TokenResponse, error) {
	payload := map[string]string{
		"grant_type":    "authorization_code",
		"client_id":     ClientID,
		"code":          code,
		"state":         verifier,
		"redirect_uri":  RedirectURI,
		"code_verifier": verifier,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := http.Post(TokenURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token exchange failed: %s", resp.Status)
	}

	var result TokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// RefreshAccessToken refreshes an expired access token.
func RefreshAccessToken(refreshToken string) (*TokenResponse, error) {
	payload := map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     ClientID,
		"refresh_token": refreshToken,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	resp, err := http.Post(TokenURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to refresh token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token refresh failed: %s", resp.Status)
	}

	var result TokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}
