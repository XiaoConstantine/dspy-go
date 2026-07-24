package oauth

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"strings"
	"testing"

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
	origHTTPDo := httpDo
	origPKCEGenerator := pkceGenerator
	origStateGenerator := oauthStateGenerator
	origAnthropicAuthURL := anthropicAuthURL
	origAnthropicExchanger := anthropicCodeExchanger
	origOpenAIAuthURL := openAIAuthURL
	origOpenAIExchanger := openAICodeExchanger
	origBrowserOpener := browserOpener
	origCurrentGOOS := currentGOOS
	origExecCommand := execCommand
	origCommandStarter := commandStarter
	origLoginInput := loginInputReader

	t.Cleanup(func() {
		randRead = origRandRead
		httpPost = origHTTPPost
		httpDo = origHTTPDo
		pkceGenerator = origPKCEGenerator
		oauthStateGenerator = origStateGenerator
		anthropicAuthURL = origAnthropicAuthURL
		anthropicCodeExchanger = origAnthropicExchanger
		openAIAuthURL = origOpenAIAuthURL
		openAICodeExchanger = origOpenAIExchanger
		browserOpener = origBrowserOpener
		currentGOOS = origCurrentGOOS
		execCommand = origExecCommand
		commandStarter = origCommandStarter
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
	assert.Equal(t, "challenge", u.Query().Get("state"))

	secure, err := url.Parse(GetOpenAIAuthorizationURLWithState("challenge", "independent-state"))
	require.NoError(t, err)
	assert.Equal(t, "independent-state", secure.Query().Get("state"))
	assert.NotEqual(t, "verifier", secure.Query().Get("state"))
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

func TestOpenAITokenContextCancellation(t *testing.T) {
	withOAuthSeams(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	httpDo = func(req *http.Request) (*http.Response, error) {
		return nil, req.Context().Err()
	}

	response, err := RefreshOpenAIAccessTokenContext(ctx, "refresh-token")
	require.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, response)
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
		httpPost = func(endpoint, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, OpenAITokenURL, endpoint)
			assert.Equal(t, "application/x-www-form-urlencoded", contentType)

			encoded, err := io.ReadAll(body)
			require.NoError(t, err)
			payload, err := url.ParseQuery(string(encoded))
			require.NoError(t, err)
			assert.Equal(t, "authorization_code", payload.Get("grant_type"))
			assert.Equal(t, OpenAIClientID, payload.Get("client_id"))
			assert.Equal(t, "auth-code", payload.Get("code"))
			assert.Equal(t, OpenAIRedirectURI, payload.Get("redirect_uri"))
			assert.Equal(t, "verifier", payload.Get("code_verifier"))

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
		httpPost = func(endpoint, contentType string, body io.Reader) (*http.Response, error) {
			assert.Equal(t, OpenAITokenURL, endpoint)
			assert.Equal(t, "application/x-www-form-urlencoded", contentType)

			encoded, err := io.ReadAll(body)
			require.NoError(t, err)
			payload, err := url.ParseQuery(string(encoded))
			require.NoError(t, err)
			assert.Equal(t, "refresh_token", payload.Get("grant_type"))
			assert.Equal(t, OpenAIClientID, payload.Get("client_id"))
			assert.Equal(t, "refresh-token", payload.Get("refresh_token"))

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

	configure := func(input string) {
		pkceGenerator = func() (string, string, error) { return "verifier", "challenge", nil }
		oauthStateGenerator = func() (string, error) { return "csrf-state", nil }
		openAIAuthURL = func(challenge, state string) string {
			assert.Equal(t, "challenge", challenge)
			assert.Equal(t, "csrf-state", state)
			return "https://example.com/authorize"
		}
		browserOpener = func(string) {}
		loginInputReader = strings.NewReader(input + "\n")
	}

	t.Run("state mismatch", func(t *testing.T) {
		configure("http://localhost/callback?code=code123&state=wrong")
		resp, err := LoginOpenAI()
		require.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "state mismatch")
	})

	t.Run("exchange failure", func(t *testing.T) {
		configure("code123#csrf-state")
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
		configure("http://localhost/callback?code=xyz789&state=csrf-state")
		openAICodeExchanger = func(code, verifier string) (*OpenAITokenResponse, error) {
			assert.Equal(t, "xyz789", code)
			assert.Equal(t, "verifier", verifier)
			return &OpenAITokenResponse{AccessToken: "access", RefreshToken: "refresh", ExpiresIn: 3600}, nil
		}

		resp, err := LoginOpenAI()
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "access", resp.AccessToken)
		assert.Equal(t, "refresh", resp.RefreshToken)
	})
}

func TestOpenBrowser(t *testing.T) {
	withOAuthSeams(t)

	tests := []struct {
		name        string
		goos        string
		wantCommand string
		wantArgs    []string
		wantStarted bool
	}{
		{
			name:        "darwin uses open",
			goos:        "darwin",
			wantCommand: "open",
			wantArgs:    []string{"https://example.com/login"},
			wantStarted: true,
		},
		{
			name:        "linux uses xdg-open",
			goos:        "linux",
			wantCommand: "xdg-open",
			wantArgs:    []string{"https://example.com/login"},
			wantStarted: true,
		},
		{
			name:        "windows uses rundll32",
			goos:        "windows",
			wantCommand: "rundll32",
			wantArgs:    []string{"url.dll,FileProtocolHandler", "https://example.com/login"},
			wantStarted: true,
		},
		{
			name:        "unsupported OS is a no-op",
			goos:        "plan9",
			wantStarted: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			currentGOOS = tc.goos

			var capturedName string
			var capturedArgs []string
			var started bool

			execCommand = func(name string, args ...string) *exec.Cmd {
				capturedName = name
				capturedArgs = append([]string{}, args...)
				return &exec.Cmd{}
			}
			commandStarter = func(cmd *exec.Cmd) error {
				started = true
				return nil
			}

			openBrowser("https://example.com/login")

			assert.Equal(t, tc.wantStarted, started)
			assert.Equal(t, tc.wantCommand, capturedName)
			assert.Equal(t, tc.wantArgs, capturedArgs)
		})
	}
}
