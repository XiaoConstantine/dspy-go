package llms

import (
	"context"
	"fmt"
	"testing"

	stdErr "errors"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestAnthropicLLM_Generate(t *testing.T) {
	prompt := "example prompt"
	ctx := context.Background()
	tests := []struct {
		name        string
		setupMock   func(*testutil.MockLLM)
		options     []core.GenerateOption
		wantResp    *core.LLMResponse
		wantErr     bool
		expectedErr string
	}{
		{
			name: "Successful generation",
			setupMock: func(mockLLM *testutil.MockLLM) {
				mockLLM.On("Generate", mock.Anything, prompt, mock.Anything).Return(&core.LLMResponse{Content: "Generated response"}, nil)
			},
			options:  []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
			wantResp: &core.LLMResponse{Content: "Generated response"},
			wantErr:  false,
		},
		{
			name: "API error",
			setupMock: func(mockLLM *testutil.MockLLM) {
				mockLLM.On("Generate", mock.Anything, prompt, mock.Anything).Return(nil, errors.WithFields(
					errors.Wrap(stdErr.New("API error"), errors.LLMGenerationFailed, "failed to generate response"),
					errors.Fields{},
				))
			},
			wantErr:     true,
			expectedErr: "failed to generate response: API error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			mockLLM := new(testutil.MockLLM)
			tt.setupMock(mockLLM)

			resp, err := mockLLM.Generate(ctx, prompt, tt.options...)

			if tt.wantErr {
				assert.Error(t, err)
				if customErr, ok := err.(*errors.Error); ok {
					assert.Equal(t, tt.expectedErr, customErr.Error())
				} else {
					t.Errorf("expected error of type *errors.Error, got %T", err)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantResp, resp)
			}
			mockLLM.AssertExpectations(t)

		})
	}
}
