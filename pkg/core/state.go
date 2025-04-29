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

// LMConfigProvider is an interface that modules can optionally implement
// to allow their LM configuration identifiers to be saved and checked.
type LMConfigProvider interface {
	GetLLMIdentifier() map[string]string // Returns map like {"provider": "OpenAI", "model": "gpt-4"}
}

// ParameterProvider is an interface that modules can optionally implement
// to allow their tuned parameters to be saved.
type ParameterProvider interface {
	GetTunedParameters() map[string]interface{}
}

// ParameterConsumer is an interface that modules can optionally implement
// to allow their tuned parameters to be loaded.
type ParameterConsumer interface {
	SetTunedParameters(params map[string]interface{}) error // Return error for validation
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
	Signature       string                 `json:"signature"`                  // Module's signature string representation (READ-ONLY during load)
	Demos           []SavedExample         `json:"demos,omitempty"`            // Saved demos, if the module provides them
	LMIdentifier    map[string]string      `json:"lm_identifier,omitempty"`    // Identifying info for the LM used (e.g., provider, model)
	TunedParameters map[string]interface{} `json:"tuned_parameters,omitempty"` // Module-specific tuned parameters (e.g., k for retriever)
	ModuleType      string                 `json:"module_type"`                // Concrete type name (e.g., "Predict")
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

		// Get LM Identifier if module provides it
		var lmIdentifier map[string]string
		if lmProvider, ok := module.(LMConfigProvider); ok {
			lmIdentifier = lmProvider.GetLLMIdentifier()
		}

		// Get Tuned Parameters
		var tunedParams map[string]interface{}
		if paramProvider, ok := module.(ParameterProvider); ok {
			tunedParams = paramProvider.GetTunedParameters()
		}

		moduleState := SavedModuleState{
			Signature:       module.GetSignature().String(),
			Demos:           savedDemos,
			LMIdentifier:    lmIdentifier,
			TunedParameters: tunedParams,
			ModuleType:      reflect.TypeOf(module).Elem().Name(),
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

// LoadProgram loads program state (demos and tuned parameters) from a JSON file
// into an existing Program instance.
// It assumes the Program `p` has already been constructed with the correct
// architecture (modules and signatures) and necessary LLMs configured.
func LoadProgram(p *Program, filepath string) error {
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
			fmt.Printf("[WARN] Loading state saved with dspy-go version '%s' but current version is '%s'. Compatibility not guaranteed.\n", savedVersion, Version)
		}
	} else {
		fmt.Println("[WARN] Saved state file does not contain dspy-go version information.")
	}

	loadedModuleNames := make(map[string]bool)

	// 4. Iterate through the program's modules and apply loaded state
	for name, module := range p.Modules {
		savedModuleState, ok := loadedState.Modules[name]
		if !ok {
			fmt.Printf("[WARN] No saved state found for module '%s' in program.\n", name)
			continue // Skip this module if no state was saved for it
		}
		loadedModuleNames[name] = true // Mark this saved state as used

		// 5. Module Type Check
		currentModuleType := reflect.TypeOf(module).Elem().Name()
		if currentModuleType != savedModuleState.ModuleType {
			fmt.Printf("[WARN] Type mismatch for module '%s': Program has type '%s', saved state has type '%s'. Skipping loading state for this module.\n", name, currentModuleType, savedModuleState.ModuleType)
			continue
		}

		// 6. LM Identifier Check (Warning only)
		if len(savedModuleState.LMIdentifier) > 0 {
			if lmProvider, ok := module.(LMConfigProvider); ok {
				currentIdentifier := lmProvider.GetLLMIdentifier()
				if !reflect.DeepEqual(savedModuleState.LMIdentifier, currentIdentifier) {
					fmt.Printf("[WARN] LM mismatch for module '%s': Saved state used %v, current module uses %v. Loaded demos/params may behave differently.\n", name, savedModuleState.LMIdentifier, currentIdentifier)
				}
			} else {
				// Saved state has LM info, but current module doesn't provide it for comparison
				fmt.Printf("[WARN] Cannot verify LM config for module '%s': Module does not implement LMConfigProvider, but saved state has LM info: %v\n", name, savedModuleState.LMIdentifier)
			}
		}

		// 7. Load Demos if module supports it
		if demoConsumer, implementsDemoConsumer := module.(DemoConsumer); implementsDemoConsumer {
			if len(savedModuleState.Demos) > 0 {
				demosToLoad := make([]Example, len(savedModuleState.Demos))
				for i, savedDemo := range savedModuleState.Demos {
					// Convert SavedExample back to Example.
					// This direct assignment is type-safe because both SavedExample and Example
					// use map[string]interface{} for Inputs and Outputs. json.Unmarshal has
					// already parsed the JSON into this structure. The responsibility for handling
					// the specific concrete types within the interface{} values lies with the
					// code that consumes the loaded Example later.
					demosToLoad[i] = Example(savedDemo)
				}
				demoConsumer.SetDemos(demosToLoad)
			}
		} else if len(savedModuleState.Demos) > 0 {
			// Saved state has demos, but the current module doesn't consume them
			fmt.Printf("[WARN] Saved state for module '%s' contains demos, but the module does not implement DemoConsumer. Demos not loaded.\n", name)
		}

		// 8. Load Tuned Parameters if module supports it
		if paramConsumer, implementsParamConsumer := module.(ParameterConsumer); implementsParamConsumer {
			if len(savedModuleState.TunedParameters) > 0 {
				err := paramConsumer.SetTunedParameters(savedModuleState.TunedParameters)
				if err != nil {
					// If SetTunedParameters returns an error (e.g., validation failed),
					// wrap it and return, as this indicates a problem loading state.
					return fmt.Errorf("error setting tuned parameters for module '%s': %w", name, err)
				}
			}
		} else if len(savedModuleState.TunedParameters) > 0 {
			// Saved state has tuned parameters, but the current module doesn't consume them
			fmt.Printf("[WARN] Saved state for module '%s' contains tuned parameters, but the module does not implement ParameterConsumer. Parameters not loaded.\n", name)
		}

		// NOTE: Signature and full LMConfig object are not loaded from state in this version.
		// The program structure (including signatures and LLM instances) must be defined before loading.
	}

	// 9. Check for unused saved state
	for savedName := range loadedState.Modules {
		if !loadedModuleNames[savedName] {
			fmt.Printf("[WARN] Saved state contains data for module '%s', but this module was not found in the current program.\n", savedName)
		}
	}

	return nil // Success
}
