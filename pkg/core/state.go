package core

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
)

// Version is a placeholder for the dspy-go version.
const Version = "0.0.0-dev"

// DemoProvider is an interface that modules can optionally implement
// to allow their demos to be retrieved.
type DemoProvider interface {
	GetDemos() []Example
}

// DemoConsumer is an interface that modules can optionally implement
// to allow their demos to be loaded.
type DemoConsumer interface {
	SetDemos([]Example)
}

// SavedExample represents a serializable dspy.Example.
type SavedExample struct {
	Inputs  map[string]interface{} `json:"inputs"`
	Outputs map[string]interface{} `json:"outputs"`
	// TODO: Add any other relevant metadata from the Example struct if needed.
}

// SavedModuleState represents the serializable state of a single DSPy module.
// Specific fields might vary depending on the module type (Predict, CoT, etc.).
type SavedModuleState struct {
	Signature  string         `json:"signature"`           // Module's signature string representation (READ-ONLY during load)
	Demos      []SavedExample `json:"demos,omitempty"`     // Saved demos, if the module provides them
	LMConfig   interface{}    `json:"lm_config,omitempty"` // Placeholder for LM config, omitted if nil (READ-ONLY during load)
	ModuleType string         `json:"module_type"`         // Concrete type name (e.g., "Predict")
	// TODO: Add other potential module-specific tuned parameters (e.g., k for retrievers).
}

// SavedProgramState represents the serializable state of an entire DSPy program.
type SavedProgramState struct {
	Modules  map[string]SavedModuleState `json:"modules"`  // Map module ID/name to its state
	Metadata map[string]string           `json:"metadata"` // e.g., {"dspy_go_version": "..."}
}

// SaveProgram serializes the current state of the Program's modules to a JSON file.
func SaveProgram(p *Program, filepath string) error {
	// 1. Create SavedProgramState instance
	state := SavedProgramState{
		Modules: make(map[string]SavedModuleState),
		Metadata: map[string]string{
			"dspy_go_version": Version,
			// Add other metadata like optimizer used, timestamp, etc. if needed
		},
	}

	// 2. Populate it by iterating through p.Modules, extracting state
	for name, module := range p.Modules {
		var demos []Example
		// Check if the module provides demos
		if demoProvider, ok := module.(DemoProvider); ok {
			demos = demoProvider.GetDemos()
		}

		savedDemos := make([]SavedExample, len(demos))
		for i, demo := range demos {
			savedDemos[i] = SavedExample(demo)
		}

		// Placeholder for LM config extraction - currently not saved
		var lmConfig interface{}

		moduleState := SavedModuleState{
			Signature:  module.GetSignature().String(), // Use String() for representation
			Demos:      savedDemos,
			LMConfig:   lmConfig,                             // Store extracted LM config (currently nil)
			ModuleType: reflect.TypeOf(module).Elem().Name(), // Get concrete type name
		}
		state.Modules[name] = moduleState
	}

	// 3. Marshal to JSON
	jsonData, err := json.MarshalIndent(state, "", "  ") // Use MarshalIndent for readability
	if err != nil {
		return fmt.Errorf("failed to marshal program state to JSON: %w", err)
	}

	// 4. Write to file
	err = os.WriteFile(filepath, jsonData, 0644) // Default permissions rw-r--r--
	if err != nil {
		return fmt.Errorf("failed to write program state to file '%s': %w", filepath, err)
	}

	return nil // Success
}

// LoadProgram loads program state (currently only demos) from a JSON file
// into an existing Program instance.
// It assumes the Program `p` has already been constructed with the correct
// architecture (modules and signatures).
func LoadProgram(p *Program, filepath string) error {
	// logger := logging.GetLogger() // Removed to break import cycle

	// 1. Read file
	jsonData, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read program state file '%s': %w", filepath, err)
	}

	// 2. Unmarshal JSON into SavedProgramState
	var loadedState SavedProgramState
	err = json.Unmarshal(jsonData, &loadedState)
	if err != nil {
		return fmt.Errorf("failed to unmarshal program state from JSON: %w", err)
	}

	// 3. Version Check
	if savedVersion, ok := loadedState.Metadata["dspy_go_version"]; ok {
		if savedVersion != Version {
			// logger.Warn(nil, "Loading state saved with dspy-go version '%s' but current version is '%s'. Compatibility not guaranteed.", savedVersion, Version)
			fmt.Printf("[WARN] Loading state saved with dspy-go version '%s' but current version is '%s'. Compatibility not guaranteed.\n", savedVersion, Version)
		}
	} else {
		// logger.Warn(nil, "Saved state file does not contain dspy-go version information.")
		fmt.Println("[WARN] Saved state file does not contain dspy-go version information.")
	}

	loadedModuleNames := make(map[string]bool)

	// 4. Iterate through the program's modules and apply loaded state
	for name, module := range p.Modules {
		savedModuleState, ok := loadedState.Modules[name]
		if !ok {
			// logger.Warn(nil, "No saved state found for module '%s' in program.", name)
			fmt.Printf("[WARN] No saved state found for module '%s' in program.\n", name)
			continue // Skip this module if no state was saved for it
		}
		loadedModuleNames[name] = true // Mark this saved state as used

		// 5. Module Type Check
		currentModuleType := reflect.TypeOf(module).Elem().Name()
		if currentModuleType != savedModuleState.ModuleType {
			// logger.Warn(nil, "Type mismatch for module '%s': Program has type '%s', saved state has type '%s'. Skipping loading state for this module.", name, currentModuleType, savedModuleState.ModuleType)
			fmt.Printf("[WARN] Type mismatch for module '%s': Program has type '%s', saved state has type '%s'. Skipping loading state for this module.\n", name, currentModuleType, savedModuleState.ModuleType)
			continue
		}

		// 6. Load Demos if module supports it
		if demoConsumer, implementsDemoConsumer := module.(DemoConsumer); implementsDemoConsumer {
			if len(savedModuleState.Demos) > 0 {
				demosToLoad := make([]Example, len(savedModuleState.Demos))
				for i, savedDemo := range savedModuleState.Demos {
					// Convert SavedExample back to Example
					// Note: This assumes Example struct has matching field names/types
					demosToLoad[i] = Example(savedDemo)
				}
				demoConsumer.SetDemos(demosToLoad)
			}
		} else if len(savedModuleState.Demos) > 0 {
			// Saved state has demos, but the current module doesn't consume them
			// logger.Warn(nil, "Saved state for module '%s' contains demos, but the module does not implement DemoConsumer. Demos not loaded.", name)
			fmt.Printf("[WARN] Saved state for module '%s' contains demos, but the module does not implement DemoConsumer. Demos not loaded.\n", name)
		}

		// NOTE: Signature and LMConfig are not loaded from state in this version.
		// The program structure (including signatures) must be defined before loading.
	}

	// 7. Check for unused saved state
	for savedName := range loadedState.Modules {
		if !loadedModuleNames[savedName] {
			// logger.Warn(nil, "Saved state contains data for module '%s', but this module was not found in the current program.", savedName)
			fmt.Printf("[WARN] Saved state contains data for module '%s', but this module was not found in the current program.\n", savedName)
		}
	}

	return nil // Success
}
