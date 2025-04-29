package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// StateTestMockModule is a simple mock implementing Module, DemoProvider, and DemoConsumer for state tests.
type StateTestMockModule struct {
	BaseModule
	MyDemos []Example
}

func NewStateTestMockModule(name string) *StateTestMockModule {
	return &StateTestMockModule{
		BaseModule: BaseModule{
			Signature: NewSignature(
				[]InputField{{Field: Field{Name: name + "_in"}}},
				[]OutputField{{Field: Field{Name: name + "_out"}}},
			),
		},
		MyDemos: []Example{},
	}
}

// Process is a dummy implementation.
func (m *StateTestMockModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	// Not needed for state tests
	return map[string]any{m.Signature.Outputs[0].Name: "mock_output"}, nil
}

// GetDemos implements DemoProvider.
func (m *StateTestMockModule) GetDemos() []Example {
	return m.MyDemos
}

// SetDemos implements DemoConsumer.
func (m *StateTestMockModule) SetDemos(demos []Example) {
	m.MyDemos = demos
}

// Clone implements Module.
func (m *StateTestMockModule) Clone() Module {
	cloned := NewStateTestMockModule("cloned")         // Name doesn't really matter here
	cloned.Signature = m.Signature                     // Copy signature
	cloned.MyDemos = append([]Example{}, m.MyDemos...) // Deep copy demos
	return cloned
}

// Another mock type for mismatch testing.
type AnotherMockModule struct {
	BaseModule
	// No Demos field
}

func NewAnotherMockModule(name string) *AnotherMockModule {
	return &AnotherMockModule{
		BaseModule: BaseModule{
			Signature: NewSignature(
				[]InputField{{Field: Field{Name: name + "_in"}}},
				[]OutputField{{Field: Field{Name: name + "_out"}}},
			),
		},
	}
}

func (m *AnotherMockModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return map[string]any{m.Signature.Outputs[0].Name: "another_mock_output"}, nil
}

func (m *AnotherMockModule) Clone() Module {
	cloned := NewAnotherMockModule("cloned")
	cloned.Signature = m.Signature
	return cloned
}

// --- Test Functions Start Here ---

func TestSaveLoadProgram_Success(t *testing.T) {
	// --- Setup Original Program ---
	mockModuleA := NewStateTestMockModule("moduleA")
	mockModuleB := NewStateTestMockModule("moduleB")

	originalDemosA := []Example{
		{Inputs: map[string]interface{}{"a_in": "q1"}, Outputs: map[string]interface{}{"a_out": "a1"}},
		{Inputs: map[string]interface{}{"a_in": "q2"}, Outputs: map[string]interface{}{"a_out": "a2"}},
	}
	originalDemosB := []Example{
		{Inputs: map[string]interface{}{"b_in": "q3"}, Outputs: map[string]interface{}{"b_out": "a3"}},
	}

	mockModuleA.SetDemos(originalDemosA)
	mockModuleB.SetDemos(originalDemosB)

	originalProgram := NewProgram(
		map[string]Module{"modA": mockModuleA, "modB": mockModuleB},
		nil, // Forward func not needed for state tests
	)

	// --- Save Program ---
	tempFile := tempFilePath(t, "success_state.json")
	err := SaveProgram(&originalProgram, tempFile)
	require.NoError(t, err, "SaveProgram should not return an error")

	// --- Verify Saved File Content ---
	jsonData, err := os.ReadFile(tempFile)
	require.NoError(t, err, "Failed to read saved file")

	var savedState SavedProgramState
	err = json.Unmarshal(jsonData, &savedState)
	require.NoError(t, err, "Failed to unmarshal saved JSON")

	// Check metadata
	assert.Equal(t, Version, savedState.Metadata["dspy_go_version"])

	// Check modules map
	require.Len(t, savedState.Modules, 2, "Should have saved state for 2 modules")
	assert.Contains(t, savedState.Modules, "modA")
	assert.Contains(t, savedState.Modules, "modB")

	// Check module A state
	stateA := savedState.Modules["modA"]
	assert.Equal(t, "StateTestMockModule", stateA.ModuleType)
	assert.Equal(t, mockModuleA.GetSignature().String(), stateA.Signature)
	require.Len(t, stateA.Demos, 2)
	assert.Equal(t, originalDemosA[0].Inputs, stateA.Demos[0].Inputs)
	assert.Equal(t, originalDemosA[0].Outputs, stateA.Demos[0].Outputs)
	assert.Equal(t, originalDemosA[1].Inputs, stateA.Demos[1].Inputs)
	assert.Equal(t, originalDemosA[1].Outputs, stateA.Demos[1].Outputs)

	// Check module B state
	stateB := savedState.Modules["modB"]
	assert.Equal(t, "StateTestMockModule", stateB.ModuleType)
	assert.Equal(t, mockModuleB.GetSignature().String(), stateB.Signature)
	require.Len(t, stateB.Demos, 1)
	assert.Equal(t, originalDemosB[0].Inputs, stateB.Demos[0].Inputs)
	assert.Equal(t, originalDemosB[0].Outputs, stateB.Demos[0].Outputs)

	// --- Load Program ---
	// Create a NEW program instance with the same structure
	newMockModuleA := NewStateTestMockModule("moduleA")
	newMockModuleB := NewStateTestMockModule("moduleB")
	loadedProgram := NewProgram(
		map[string]Module{"modA": newMockModuleA, "modB": newMockModuleB},
		nil,
	)

	err = LoadProgram(&loadedProgram, tempFile)
	require.NoError(t, err, "LoadProgram should not return an error")

	// --- Verify Loaded State ---
	loadedDemosA := newMockModuleA.GetDemos()
	loadedDemosB := newMockModuleB.GetDemos()

	// Use assert.Equal which handles slices and maps correctly
	assert.Equal(t, originalDemosA, loadedDemosA, "Demos for module A should match original")
	assert.Equal(t, originalDemosB, loadedDemosB, "Demos for module B should match original")
}

func TestLoadProgram_VersionMismatch(t *testing.T) {
	// --- Setup Saved State with different version ---
	mismatchedState := SavedProgramState{
		Modules: make(map[string]SavedModuleState),
		Metadata: map[string]string{
			"dspy_go_version": "ancient-version-0.0.1",
		},
	}
	jsonData, err := json.Marshal(mismatchedState)
	require.NoError(t, err)

	tempFile := tempFilePath(t, "version_mismatch.json")
	err = os.WriteFile(tempFile, jsonData, 0644)
	require.NoError(t, err)

	// --- Setup Program to Load Into ---
	mockModule := NewStateTestMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": mockModule}, nil)

	// --- Load Program ---
	// We expect a warning printed to stdout, but no error returned
	err = LoadProgram(&program, tempFile)
	assert.NoError(t, err, "LoadProgram should not return error on version mismatch, only warn")

	// TODO: Optionally capture stdout/stderr to verify the warning message
	// This often requires more complex test setup (e.g., redirecting os.Stdout)
}

func TestLoadProgram_ModuleTypeMismatch(t *testing.T) {
	// --- Setup Saved State for StateTestMockModule ---
	originalDemos := []SavedExample{
		{Inputs: map[string]interface{}{"a_in": "q1"}, Outputs: map[string]interface{}{"a_out": "a1"}},
	}
	savedState := SavedProgramState{
		Modules: map[string]SavedModuleState{
			"modA": {
				ModuleType: "StateTestMockModule", // Save as this type
				Signature:  "a_in -> a_out",       // Signature doesn't matter much for loading
				Demos:      originalDemos,
			},
		},
		Metadata: map[string]string{"dspy_go_version": Version},
	}
	jsonData, err := json.Marshal(savedState)
	require.NoError(t, err)

	tempFile := tempFilePath(t, "type_mismatch.json")
	err = os.WriteFile(tempFile, jsonData, 0644)
	require.NoError(t, err)

	// --- Setup Program with a DIFFERENT module type ---
	mismatchedModule := NewAnotherMockModule("moduleA") // Use the other mock type
	program := NewProgram(map[string]Module{"modA": mismatchedModule}, nil)

	// --- Load Program ---
	// We expect a warning, no error, and state NOT loaded
	err = LoadProgram(&program, tempFile)
	assert.NoError(t, err, "LoadProgram should not return error on module type mismatch")

	// Verify state was NOT loaded (since AnotherMockModule doesn't have MyDemos)
	// If it had a Demos field, we would assert it's empty/nil.
	// The primary check is the lack of error and the warning (checked conceptually).
}

func TestLoadProgram_MissingModuleInProgram(t *testing.T) {
	// --- Setup Saved State for modules A and B ---
	demosA := []SavedExample{
		{Inputs: map[string]interface{}{"a_in": "q1"}, Outputs: map[string]interface{}{"a_out": "a1"}},
	}
	savedState := SavedProgramState{
		Modules: map[string]SavedModuleState{
			"modA": {
				ModuleType: "StateTestMockModule",
				Signature:  "a_in -> a_out",
				Demos:      demosA,
			},
			"modB": { // State for module B exists
				ModuleType: "StateTestMockModule",
				Signature:  "b_in -> b_out",
				Demos:      []SavedExample{},
			},
		},
		Metadata: map[string]string{"dspy_go_version": Version},
	}
	jsonData, err := json.Marshal(savedState)
	require.NoError(t, err)

	tempFile := tempFilePath(t, "missing_in_prog.json")
	err = os.WriteFile(tempFile, jsonData, 0644)
	require.NoError(t, err)

	// --- Setup Program with ONLY module A ---
	mockModuleA := NewStateTestMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": mockModuleA}, nil)

	// --- Load Program ---
	// We expect a warning about modB, no error, and modA loaded
	err = LoadProgram(&program, tempFile)
	assert.NoError(t, err, "LoadProgram should not return error when state exists for non-existent program module")

	// Verify state WAS loaded for modA
	assert.Len(t, mockModuleA.GetDemos(), 1, "Demos for module A should have been loaded")
	assert.Equal(t, demosA[0].Inputs, mockModuleA.GetDemos()[0].Inputs)

	// TODO: Optionally capture stdout/stderr to verify the warning about modB.
}

func TestLoadProgram_MissingModuleInState(t *testing.T) {
	// --- Setup Saved State for ONLY module A ---
	demosA := []SavedExample{
		{Inputs: map[string]interface{}{"a_in": "q1"}, Outputs: map[string]interface{}{"a_out": "a1"}},
	}
	savedState := SavedProgramState{
		Modules: map[string]SavedModuleState{
			"modA": {
				ModuleType: "StateTestMockModule",
				Signature:  "a_in -> a_out",
				Demos:      demosA,
			},
			// No state for modB
		},
		Metadata: map[string]string{"dspy_go_version": Version},
	}
	jsonData, err := json.Marshal(savedState)
	require.NoError(t, err)

	tempFile := tempFilePath(t, "missing_in_state.json")
	err = os.WriteFile(tempFile, jsonData, 0644)
	require.NoError(t, err)

	// --- Setup Program with modules A and B ---
	mockModuleA := NewStateTestMockModule("moduleA")
	mockModuleB := NewStateTestMockModule("moduleB") // Module B exists in program
	program := NewProgram(map[string]Module{"modA": mockModuleA, "modB": mockModuleB}, nil)

	// --- Load Program ---
	// We expect a warning about modB, no error, and modA loaded
	err = LoadProgram(&program, tempFile)
	assert.NoError(t, err, "LoadProgram should not return error when program module has no corresponding saved state")

	// Verify state WAS loaded for modA
	assert.Len(t, mockModuleA.GetDemos(), 1, "Demos for module A should have been loaded")
	assert.Equal(t, demosA[0].Inputs, mockModuleA.GetDemos()[0].Inputs)

	// Verify state was NOT loaded for modB (it starts empty)
	assert.Empty(t, mockModuleB.GetDemos(), "Demos for module B should remain empty")

	// TODO: Optionally capture stdout/stderr to verify the warning about modB.
}

func TestLoadProgram_NoDemoConsumer(t *testing.T) {
	// --- Setup Saved State with demos for a module ---
	demosA := []SavedExample{
		{Inputs: map[string]interface{}{"a_in": "q1"}, Outputs: map[string]interface{}{"a_out": "a1"}},
	}
	savedState := SavedProgramState{
		Modules: map[string]SavedModuleState{
			"modA": {
				ModuleType: "AnotherMockModule", // Match the type we will use in program
				Signature:  "a_in -> a_out",
				Demos:      demosA, // State includes demos
			},
		},
		Metadata: map[string]string{"dspy_go_version": Version},
	}
	jsonData, err := json.Marshal(savedState)
	require.NoError(t, err)

	tempFile := tempFilePath(t, "no_consumer.json")
	err = os.WriteFile(tempFile, jsonData, 0644)
	require.NoError(t, err)

	// --- Setup Program with a module that DOES NOT implement DemoConsumer ---
	noConsumerModule := NewAnotherMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": noConsumerModule}, nil)

	// --- Load Program ---
	// We expect a warning, no error, and demos NOT loaded
	err = LoadProgram(&program, tempFile)
	assert.NoError(t, err, "LoadProgram should not return error when module does not consume demos")

	// Since AnotherMockModule doesn't have a GetDemos method or Demos field,
	// we primarily rely on the conceptual check of the warning and the lack of error.

	// TODO: Optionally capture stdout/stderr to verify the warning.
}

func TestSaveProgram_FileError(t *testing.T) {
	// --- Setup Program ---
	mockModuleA := NewStateTestMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": mockModuleA}, nil)

	// --- Attempt to Save to an invalid path (e.g., a directory) ---
	invalidPath := t.TempDir() // Use the temporary directory itself as the file path
	err := SaveProgram(&program, invalidPath)

	// --- Verify Error ---
	require.Error(t, err, "SaveProgram should return an error for invalid file path")
	assert.ErrorContains(t, err, "failed to write program state to file") // Check for specific error text if desired
}

func TestLoadProgram_FileError(t *testing.T) {
	// --- Setup Program ---
	mockModuleA := NewStateTestMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": mockModuleA}, nil)

	// --- Attempt to Load from a non-existent file ---
	nonExistentPath := filepath.Join(t.TempDir(), "does_not_exist.json")
	err := LoadProgram(&program, nonExistentPath)

	// --- Verify Error ---
	require.Error(t, err, "LoadProgram should return an error for non-existent file")
	assert.ErrorContains(t, err, "failed to read program state file")
}

func TestLoadProgram_JsonError(t *testing.T) {
	// --- Setup Program ---
	mockModuleA := NewStateTestMockModule("moduleA")
	program := NewProgram(map[string]Module{"modA": mockModuleA}, nil)

	// --- Create invalid JSON file ---
	invalidJson := "{ this is not valid json, }"
	tempFile := tempFilePath(t, "invalid.json")
	err := os.WriteFile(tempFile, []byte(invalidJson), 0644)
	require.NoError(t, err)

	// --- Attempt to Load invalid JSON ---
	err = LoadProgram(&program, tempFile)

	// --- Verify Error ---
	require.Error(t, err, "LoadProgram should return an error for invalid JSON")
	assert.ErrorContains(t, err, "failed to unmarshal program state from JSON")
}

// Helper function to create a temporary file path.
func tempFilePath(t *testing.T, pattern string) string {
	t.Helper()
	tempDir := t.TempDir()
	return filepath.Join(tempDir, pattern)
}
