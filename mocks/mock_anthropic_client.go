package mock_anthropic

import (
	"context"
	"reflect"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/golang/mock/gomock"
)

// MockLLM is a mock of LLM interface.
type MockLLM struct {
	ctrl     *gomock.Controller
	recorder *MockLLMMockRecorder
}

// MockLLMMockRecorder is the mock recorder for MockLLM.
type MockLLMMockRecorder struct {
	mock *MockLLM
}

// NewMockLLM creates a new mock instance.
func NewMockLLM(ctrl *gomock.Controller) *MockLLM {
	mock := &MockLLM{ctrl: ctrl}
	mock.recorder = &MockLLMMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockLLM) EXPECT() *MockLLMMockRecorder {
	return m.recorder
}

// Generate mocks base method.
func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
	m.ctrl.T.Helper()
	varargs := []interface{}{ctx, prompt}
	for _, a := range options {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "Generate", varargs...)
	ret0, _ := ret[0].(string)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Generate indicates an expected call of Generate.
func (mr *MockLLMMockRecorder) Generate(ctx, prompt interface{}, options ...interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]interface{}{ctx, prompt}, options...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Generate", reflect.TypeOf((*MockLLM)(nil).Generate), varargs...)
}

// GenerateWithJSON mocks base method.
func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.ctrl.T.Helper()
	varargs := []interface{}{ctx, prompt}
	for _, a := range options {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GenerateWithJSON", varargs...)
	ret0, _ := ret[0].(map[string]interface{})
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GenerateWithJSON indicates an expected call of GenerateWithJSON.
func (mr *MockLLMMockRecorder) GenerateWithJSON(ctx, prompt interface{}, options ...interface{}) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]interface{}{ctx, prompt}, options...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GenerateWithJSON", reflect.TypeOf((*MockLLM)(nil).GenerateWithJSON), varargs...)
}

// GetModelID mocks base method.
func (m *MockLLM) GetModelID() string {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetModelID")
	ret0, _ := ret[0].(string)
	return ret0
}

// GetModelID indicates an expected call of GetModelID.
func (mr *MockLLMMockRecorder) GetModelID() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetModelID", reflect.TypeOf((*MockLLM)(nil).GetModelID))
}

// GetProviderName mocks base method.
func (m *MockLLM) GetProviderName() string {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetProviderName")
	ret0, _ := ret[0].(string)
	return ret0
}

// GetProviderName indicates an expected call of GetProviderName.
func (mr *MockLLMMockRecorder) GetProviderName() *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetProviderName", reflect.TypeOf((*MockLLM)(nil).GetProviderName))
}
