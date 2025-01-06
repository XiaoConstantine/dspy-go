package logging

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestContextValues(t *testing.T) {
	ctx := context.Background()

	// Test ModelID
	modelID := core.ModelID("test-model")
	ctxWithModel := WithModelID(ctx, modelID)
	retrievedModelID, ok := GetModelID(ctxWithModel)
	assert.True(t, ok)
	assert.Equal(t, modelID, retrievedModelID)

	// Test TokenInfo
	tokenInfo := &core.TokenInfo{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}
	ctxWithToken := WithTokenInfo(ctx, tokenInfo)
	retrievedTokenInfo, ok := GetTokenInfo(ctxWithToken)
	assert.True(t, ok)
	assert.Equal(t, tokenInfo, retrievedTokenInfo)

	// Test invalid context values
	_, ok = GetModelID(ctx)
	assert.False(t, ok)
	_, ok = GetTokenInfo(ctx)
	assert.False(t, ok)
}
