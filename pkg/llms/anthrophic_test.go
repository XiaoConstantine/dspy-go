package llms

import (
	"context"
	"errors"
	"testing"

	mock_anthropic "github.com/XiaoConstantine/dspy-go/mocks"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func TestAnthropicLLM_Generate(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockLLM := mock_anthropic.NewMockLLM(ctrl)

	ctx := context.Background()
	prompt := "Test prompt"

	tests := []struct {
		name        string
		setupMock   func()
		options     []core.GenerateOption
		wantResp    string
		wantErr     bool
		expectedErr string
	}{
		{
			name: "Successful generation",
			setupMock: func() {
				mockLLM.EXPECT().Generate(gomock.Any(), prompt, gomock.Any()).Return("Generated response", nil)
			},
			options:  []core.GenerateOption{core.WithMaxTokens(100), core.WithTemperature(0.7)},
			wantResp: "Generated response",
			wantErr:  false,
		},
		{
			name: "API error",
			setupMock: func() {
				mockLLM.EXPECT().Generate(gomock.Any(), prompt, gomock.Any()).Return("", errors.New("API error"))
			},
			wantErr:     true,
			expectedErr: "API error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupMock()

			resp, err := mockLLM.Generate(ctx, prompt, tt.options...)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, tt.expectedErr, err.Error())
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.wantResp, resp)
			}
		})
	}
}
