package oauth

import (
	"context"
	"net/http"
	"net/url"
	"strings"
)

var (
	httpPost = http.Post
	httpDo   = http.DefaultClient.Do
)

func postFormContext(ctx context.Context, endpoint string, values url.Values) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(values.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return httpDo(req)
}
