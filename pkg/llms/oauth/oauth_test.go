package oauth

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type errReader struct {
	err error
}

func (e errReader) Read(_ []byte) (int, error) {
	return 0, e.err
}

func withOAuthSeams(t *testing.T) {
	t.Helper()

	origRandRead := randRead
	origHTTPPost := httpPost
	origPKCEGenerator := pkceGenerator
	origAnthropicAuthURL := anthropicAuthURL
	origAnthropicExchanger := anthropicCodeExchanger
	origOpenAIAuthURL := openAIAuthURL
	origOpenAIExchanger := openAICodeExchanger
	origBrowserOpener := browserOpener
	origLoginInput := loginInputReader

	t.Cleanup(func() {
		randRead = origRandRead
		httpPost = origHTTPPost
		pkceGenerator = origPKCEGenerator
		anthropicAuthURL = origAnthropicAuthURL
		anthropicCodeExchanger = origAnthropicExchanger
		openAIAuthURL = origOpenAIAuthURL
		openAICodeExchanger = origOpenAIExchanger
		browserOpener = origBrowserOpener
		loginInputReader = origLoginInput
	})
}

func httpResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Status:     fmt.Sprintf("%d %s", statusCode, http.StatusText(statusCode)),
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

func TestGeneratePKCE(t *testing.T) {
	withOAuthSeams(t)

	verifier, challenge, err := GeneratePKCE()
	require.NoError(t, err)
	assert.NotEmpty(t, verifier)
	assert.NotEmpty(t, challenge)
	assert.NotContains(t, verifier, "=")
	assert.NotContains(t, challenge, "=")

	hash := sha256.Sum256([]byte(verifier))
	expectedChallenge := base64.RawURLEncoding.EncodeToString(hash[:])
	assert.Equal(t, expectedChallenge, challenge)

	verifier2, challenge2, err := GeneratePKCE()
	require.NoError(t, err)
	assert.NotEqual(t, verifier, verifier2)
	assert.NotEqual(t, challenge, challenge2)
}

func TestGeneratePKCE_RandError(t *testing.T) {
	withOAuthSeams(t)

	randRead = func(_ []byte) (int, error) {
		return 0, errors.New("entropy unavailable")
	}

	verifier, challenge, err := GeneratePKCE()
	require.Error(t, err)
	assert.Empty(t, verifier)
	assert.Empty(t, challenge)
	assert.Contains(t, err.Error(), "entropy unavailable")
}

func TestBase64URLEncode(t *testing.T) {
	assert.Equal(t, "", base64URLEncode(nil))
	assert.Equal(t, "Pz8", base64URLEncode([]byte("??")))
	assert.NotContains(t, base64URLEncode([]byte{0xfb, 0xff, 0xef}), "+")
	assert.NotContains(t, base64URLEncode([]byte{0xfb, 0xff, 0xef}), "/")
	assert.NotContains(t, base64URLEncode([]byte{0xfb, 0xff, 0xef}), "=")
}

func TestExtractCode(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "bare code", input: "abc123", want: "abc123"},
		{name: "code with fragment", input: "abc123#state=xyz", want: "abc123"},
		{name: "full url", input: "http://localhost/callback?code=abc123", want: "abc123"},
		{name: "full url with extra params", input: "http://localhost/callback?foo=1&code=abc123&bar=2", want: "abc123"},
		{name: "empty code value", input: "http://localhost/callback?code=", want: ""},
		{name: "no code param", input: "http://localhost/callback?state=xyz", want: "http://localhost/callback?state=xyz"},
		{name: "multiple code segments", input: "prefixcode=firstcode=second", want: "first"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, extractCode(tc.input))
		})
	}
}

func TestGetAuthorizationURL(t *testing.T) {
	u, err := url.Parse(GetAuthorizationURL("verifier", "challenge"))
	require.NoError(t, err)
	assert.Equal(t, AuthorizationURL, u.Scheme+"://"+u.Host+u.Path)
	assert.Equal(t, "code", u.Query().Get("response_type"))
	assert.Equal(t, ClientID, u.Query().Get("client_id"))
	assert.Equal(t, RedirectURI, u.Query().Get("redirect_uri"))
	assert.Equal(t, Scopes, u.Query().Get("scope"))
	assert.Equal(t, "challenge", u.Query().Get("code_challenge"))
	assert.Equal(t, "S256", u.Query().Get("code_challenge_method"))
	assert.Equal(t, "verifier", u.Query().Get("state"))
}

func TestGetOpenAIAuthorizationURL(t *testing.T) {
	u, err := url.Parse(GetOpenAIAuthorizationURL("verifier", "challenge"))
	require.NoError(t, err)
	assert.Equal(t, OpenAIAuthorizationURL, u.Scheme+"://"+u.Host+u.Path)
	assert.Equal(t, "code", u.Query().Get("response_type"))
	assert.Equal(t, OpenAIClientID, u.Query().Get("client_id"))
	assert.Equal(t, OpenAIRedirectURI, u.Query().Get("redirect_uri"))
	assert.Equal(t, OpenAIScopes, u.Query().Get("scope"))
	assert.Equal(t, "challenge", u.Query().Get("code_challenge"))
	assert.Equal(t, "S256", u.Query().Get("code_challenge_method"))
	assert.Equal(t, "verifier", u.Query().Get("state"))
}

func TestExchangeCode(t *testing.T) {
	withOAuthSeams(t)

	t.Run("success", func(t *testing.T) {
		httpPost = func(url, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, TokenURL, url)
			assert.Equal(t, "application/json", contentType)

			var payload map[string]string
			require.NoError(t, json.NewDecoder(body).Decode(&payload))
			assert.Equal(t, "authorization_code", payload["grant_type"])
			assert.Equal(t, ClientID, payload["client_id"])
			assert.Equal(t, "auth-code", payload["code"])
			assert.Equal(t, "verifier", payload["state"])
			assert.Equal(t, RedirectURI, payload["redirect_uri"])
			assert.Equal(t, "verifier", payload["code_verifier"])

			return httpResponse(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","expires_in":3600}`), nil
		}

		resp, err := ExchangeCode("auth-code", "verifier")
		require.NoError(t, err)
		assert.Equal(t, "access", resp.AccessToken)
		assert.Equal(t, "refresh", resp.RefreshToken)
		assert.Equal(t, 3600, resp.ExpiresIn)
	})

	t.Run("transport error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return nil, errors.New("network down")
		}

		resp, err := ExchangeCode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to exchange code")
	})

	t.Run("status error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusBadRequest, `{"error":"nope"}`), nil
		}

		resp, err := ExchangeCode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "token exchange failed")
	})

	t.Run("decode error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusOK, `{`), nil
		}

		resp, err := ExchangeCode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestRefreshAccessToken(t *testing.T) {
	withOAuthSeams(t)

	t.Run("success", func(t *testing.T) {
		httpPost = func(url, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, TokenURL, url)
			assert.Equal(t, "application/json", contentType)

			var payload map[string]string
			require.NoError(t, json.NewDecoder(body).Decode(&payload))
			assert.Equal(t, "refresh_token", payload["grant_type"])
			assert.Equal(t, ClientID, payload["client_id"])
			assert.Equal(t, "refresh-token", payload["refresh_token"])

			return httpResponse(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","expires_in":3600}`), nil
		}

		resp, err := RefreshAccessToken("refresh-token")
		require.NoError(t, err)
		assert.Equal(t, "access", resp.AccessToken)
	})

	t.Run("transport error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return nil, errors.New("network down")
		}

		resp, err := RefreshAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to refresh token")
	})

	t.Run("status error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusUnauthorized, `{"error":"nope"}`), nil
		}

		resp, err := RefreshAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "token refresh failed")
	})

	t.Run("decode error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusOK, `{`), nil
		}

		resp, err := RefreshAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestExchangeOpenAICode(t *testing.T) {
	withOAuthSeams(t)

	t.Run("success", func(t *testing.T) {
		httpPost = func(url, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, OpenAITokenURL, url)
			assert.Equal(t, "application/json", contentType)

			var payload map[string]string
			require.NoError(t, json.NewDecoder(body).Decode(&payload))
			assert.Equal(t, "authorization_code", payload["grant_type"])
			assert.Equal(t, OpenAIClientID, payload["client_id"])
			assert.Equal(t, "auth-code", payload["code"])
			assert.Equal(t, OpenAIRedirectURI, payload["redirect_uri"])
			assert.Equal(t, "verifier", payload["code_verifier"])

			return httpResponse(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","expires_in":3600,"token_type":"Bearer","scope":"openid"}`), nil
		}

		resp, err := ExchangeOpenAICode("auth-code", "verifier")
		require.NoError(t, err)
		assert.Equal(t, "access", resp.AccessToken)
		assert.Equal(t, "refresh", resp.RefreshToken)
		assert.Equal(t, "Bearer", resp.TokenType)
	})

	t.Run("transport error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return nil, errors.New("network down")
		}

		resp, err := ExchangeOpenAICode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to exchange code")
	})

	t.Run("status error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusBadRequest, `{"error":"nope"}`), nil
		}

		resp, err := ExchangeOpenAICode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "token exchange failed")
	})

	t.Run("decode error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusOK, `{`), nil
		}

		resp, err := ExchangeOpenAICode("auth-code", "verifier")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestRefreshOpenAIAccessToken(t *testing.T) {
	withOAuthSeams(t)

	t.Run("success", func(t *testing.T) {
		httpPost = func(url, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, OpenAITokenURL, url)
			assert.Equal(t, "application/json", contentType)

			var payload map[string]string
			require.NoError(t, json.NewDecoder(body).Decode(&payload))
			assert.Equal(t, "refresh_token", payload["grant_type"])
			assert.Equal(t, OpenAIClientID, payload["client_id"])
			assert.Equal(t, "refresh-token", payload["refresh_token"])

			return httpResponse(http.StatusOK, `{"access_token":"access","refresh_token":"refresh","expires_in":3600,"token_type":"Bearer","scope":"openid"}`), nil
		}

		resp, err := RefreshOpenAIAccessToken("refresh-token")
		require.NoError(t, err)
		assert.Equal(t, "access", resp.AccessToken)
	})

	t.Run("transport error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return nil, errors.New("network down")
		}

		resp, err := RefreshOpenAIAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to refresh token")
	})

	t.Run("status error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusUnauthorized, `{"error":"nope"}`), nil
		}

		resp, err := RefreshOpenAIAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "token refresh failed")
	})

	t.Run("decode error", func(t *testing.T) {
		httpPost = func(string, string, io.Reader) (*http.Response, error) {
			return httpResponse(http.StatusOK, `{`), nil
		}

		resp, err := RefreshOpenAIAccessToken("refresh-token")
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestLoginAnthropic(t *testing.T) {
	withOAuthSeams(t)

	t.Run("pkce failure", func(t *testing.T) {
		pkceGenerator = func() (string, string, error) {
			return "", "", errors.New("pkce failed")
		}

		resp, err := LoginAnthropic()
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to generate PKCE")
	})

	t.Run("read input failure", func(t *testing.T) {
		pkceGenerator = func() (string, string, error) { return "verifier", "challenge", nil }
		anthropicAuthURL = func(verifier, challenge string) string {
			assert.Equal(t, "verifier", verifier)
			assert.Equal(t, "challenge", challenge)
			return "https://example.com/authorize"
		}
		browserOpener = func(string) {}
		loginInputReader = errReader{err: errors.New("stdin failed")}

		resp, err := LoginAnthropic()
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to read input")
	})

	t.Run("success", func(t *testing.T) {
		var openedURL string
		pkceGenerator = func() (string, string, error) { return "verifier", "challenge", nil }
		anthropicAuthURL = func(verifier, challenge string) string {
			assert.Equal(t, "verifier", verifier)
			assert.Equal(t, "challenge", challenge)
			return "https://example.com/authorize"
		}
		browserOpener = func(url string) { openedURL = url }
		loginInputReader = strings.NewReader("http://localhost/callback?code=abc123&state=ignored\n")
		anthropicCodeExchanger = func(code, verifier string) (*TokenResponse, error) {
			assert.Equal(t, "abc123", code)
			assert.Equal(t, "verifier", verifier)
			return &TokenResponse{AccessToken: "access", RefreshToken: "refresh", ExpiresIn: 3600}, nil
		}

		resp, err := LoginAnthropic()
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "access", resp.AccessToken)
		assert.Equal(t, "refresh", resp.RefreshToken)
		assert.Equal(t, "https://example.com/authorize", openedURL)
	})
}

func TestLoginOpenAI(t *testing.T) {
	withOAuthSeams(t)

	t.Run("exchange failure", func(t *testing.T) {
		pkceGenerator = func() (string, string, error) { return "verifier", "challenge", nil }
		openAIAuthURL = func(verifier, challenge string) string {
			assert.Equal(t, "verifier", verifier)
			assert.Equal(t, "challenge", challenge)
			return "https://example.com/authorize"
		}
		browserOpener = func(string) {}
		loginInputReader = strings.NewReader("code123\n")
		openAICodeExchanger = func(code, verifier string) (*OpenAITokenResponse, error) {
			assert.Equal(t, "code123", code)
			assert.Equal(t, "verifier", verifier)
			return nil, errors.New("exchange failed")
		}

		resp, err := LoginOpenAI()
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "failed to exchange code")
	})

	t.Run("success", func(t *testing.T) {
		var openedURL string
		pkceGenerator = func() (string, string, error) { return "verifier", "challenge", nil }
		openAIAuthURL = func(verifier, challenge string) string {
			assert.Equal(t, "verifier", verifier)
			assert.Equal(t, "challenge", challenge)
			return "https://example.com/openai"
		}
		browserOpener = func(url string) { openedURL = url }
		loginInputReader = strings.NewReader("http://localhost/callback?code=xyz789\n")
		openAICodeExchanger = func(code, verifier string) (*OpenAITokenResponse, error) {
			assert.Equal(t, "xyz789", code)
			assert.Equal(t, "verifier", verifier)
			return &OpenAITokenResponse{
				AccessToken:  "access",
				RefreshToken: "refresh",
				ExpiresIn:    3600,
				TokenType:    "Bearer",
				Scope:        "openid",
			}, nil
		}

		resp, err := LoginOpenAI()
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "access", resp.AccessToken)
		assert.Equal(t, "refresh", resp.RefreshToken)
		assert.Equal(t, "https://example.com/openai", openedURL)
	})
}

func TestOpenBrowser(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skip command-path interception on windows")
	}

	tempDir := t.TempDir()
	capturePath := filepath.Join(tempDir, "args.txt")

	commandName := "xdg-open"
	if runtime.GOOS == "darwin" {
		commandName = "open"
	}

	scriptPath := filepath.Join(tempDir, commandName)
	script := fmt.Sprintf("#!/bin/sh\nprintf '%%s' \"$1\" > %s\n", capturePath)
	require.NoError(t, os.WriteFile(scriptPath, []byte(script), 0o755))

	origPath := os.Getenv("PATH")
	require.NoError(t, os.Setenv("PATH", tempDir+string(os.PathListSeparator)+origPath))
	t.Cleanup(func() {
		_ = os.Setenv("PATH", origPath)
	})

	openBrowser("https://example.com/login")

	require.Eventually(t, func() bool {
		data, err := os.ReadFile(capturePath)
		return err == nil && string(data) == "https://example.com/login"
	}, time.Second, 25*time.Millisecond)
}
