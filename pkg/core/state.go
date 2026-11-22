package core

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strconv"
)

// Version is a placeholder for the dspy-go version.
const Version = "0.0.0-dev"

// DefaultMaxSavedMediaBytes caps total raw media bytes serialized across demos
// in a saved program state (10 MiB).
const DefaultMaxSavedMediaBytes = 10 * 1024 * 1024

// savedMediaEnvelopeType marks a demo value that holds an image or audio
// block with its bytes as base64. ContentBlock.Data itself stays json:"-".
const savedMediaEnvelopeType = "__dspy_media_v1__"

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
	GetTunedParameters() map[string]any
}

// ParameterConsumer is an interface that modules can optionally implement
// to allow their tuned parameters to be loaded.
type ParameterConsumer interface {
	SetTunedParameters(params map[string]any) error // Return error for validation
}

// SavedExample represents a serializable dspy.Example.
type SavedExample struct {
	Inputs  map[string]any `json:"inputs"`
	Outputs map[string]any `json:"outputs"`
	// TODO: Add any other relevant metadata from the Example struct if needed.
}

// SavedModuleState represents the serializable state of a single DSPy module.
// Specific fields might vary depending on the module type (Predict, CoT, etc.).
//
// A predictor is a module without sub-modules. A predictor's state carries its
// instruction and output prefixes, which optimizers tune. A module built from
// other modules, such as ChainOfThought or ReAct, carries the state of every
// predictor inside it in Predictors.
type SavedModuleState struct {
	Signature       string            `json:"signature"`                  // Module's signature string representation (READ-ONLY during load)
	Instruction     *string           `json:"instruction,omitempty"`      // Predictor's instruction; absent in older files, whose instructions are not loaded
	OutputPrefixes  map[string]string `json:"output_prefixes,omitempty"`  // Predictor's output field prefixes by field name
	Adapter         string            `json:"adapter,omitempty"`          // Structured-output adapter name (e.g. "xml", "json")
	AdapterVersion  string            `json:"adapter_version,omitempty"`  // Adapter format version
	Demos           []SavedExample    `json:"demos,omitempty"`            // Saved demos, if the module provides them
	LMIdentifier    map[string]string `json:"lm_identifier,omitempty"`    // Identifying info for the LM used (e.g., provider, model)
	TunedParameters map[string]any    `json:"tuned_parameters,omitempty"` // Module-specific tuned parameters (e.g., k for retriever)
	ModuleType      string            `json:"module_type"`                // Concrete type name (e.g., "Predict")
	// Predictors holds the state of each predictor nested in this module, keyed
	// by its path of GetSubModules indexes, such as "1.0" for the first
	// sub-module of the second sub-module.
	Predictors map[string]SavedModuleState `json:"predictors,omitempty"`
}

// SavedProgramState represents the serializable state of an entire DSPy program.
type SavedProgramState struct {
	Modules  map[string]SavedModuleState `json:"modules"`  // Map module ID/name to its state
	Metadata map[string]string           `json:"metadata"` // e.g., {"dspy_go_version": "..."}
}

// LoadProgramOptions controls optional behavior for program-state loading.
type LoadProgramOptions struct {
	// WarningHandler receives non-fatal compatibility or shape warnings encountered
	// while loading state. When nil, warnings are ignored.
	WarningHandler func(message string)
}

func (o LoadProgramOptions) warnf(format string, args ...any) {
	if o.WarningHandler != nil {
		o.WarningHandler(fmt.Sprintf(format, args...))
	}
}

// persistedModuleTypeName preserves the historical state-file behavior of
// storing concrete Go type names, while avoiding panics for non-pointer
// module implementations.
func persistedModuleTypeName(module Module) string {
	if module == nil {
		return ""
	}

	typ := reflect.TypeOf(module)
	if typ == nil {
		return ""
	}
	if typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.Name() != "" {
		return typ.Name()
	}

	if moduleType := module.GetModuleType(); moduleType != "" {
		return moduleType
	}

	return typ.String()
}

// namedPredictor is a predictor nested in a module, with its path of
// GetSubModules indexes.
type namedPredictor struct {
	path   string
	module Module
}

// nestedPredictors returns the predictors inside module, depth first, or nil
// when module has no sub-modules and so is a predictor itself.
func nestedPredictors(module Module) []namedPredictor {
	provider, ok := module.(SubModuleProvider)
	if !ok {
		return nil
	}
	var predictors []namedPredictor
	for i, sub := range provider.GetSubModules() {
		if sub == nil {
			continue
		}
		index := strconv.Itoa(i)
		nested := nestedPredictors(sub)
		if nested == nil {
			predictors = append(predictors, namedPredictor{path: index, module: sub})
			continue
		}
		for _, predictor := range nested {
			predictors = append(predictors, namedPredictor{path: index + "." + predictor.path, module: predictor.module})
		}
	}
	return predictors
}

// saveModuleState captures module's state and, for a module built from other
// modules, the state of every predictor inside it.
func saveModuleState(module Module, totalMediaBytes *int) SavedModuleState {
	state := SavedModuleState{
		Signature:  module.GetSignature().String(),
		ModuleType: persistedModuleTypeName(module),
	}
	if adapterProvider, ok := module.(AdapterProvider); ok {
		state.Adapter, state.AdapterVersion = adapterProvider.GetAdapterInfo()
	}
	if demoProvider, ok := module.(DemoProvider); ok {
		demos := demoProvider.GetDemos()
		state.Demos = make([]SavedExample, len(demos))
		for i, demo := range demos {
			state.Demos[i] = encodeSavedExample(demo, totalMediaBytes)
		}
	}
	if lmProvider, ok := module.(LMConfigProvider); ok {
		state.LMIdentifier = lmProvider.GetLLMIdentifier()
	}
	if paramProvider, ok := module.(ParameterProvider); ok {
		state.TunedParameters = paramProvider.GetTunedParameters()
	}

	predictors := nestedPredictors(module)
	if predictors == nil {
		signature := module.GetSignature()
		state.Instruction = &signature.Instruction
		if len(signature.Outputs) > 0 {
			state.OutputPrefixes = make(map[string]string, len(signature.Outputs))
			for _, output := range signature.Outputs {
				state.OutputPrefixes[output.Name] = output.Prefix
			}
		}
		return state
	}
	state.Predictors = make(map[string]SavedModuleState, len(predictors))
	for _, predictor := range predictors {
		state.Predictors[predictor.path] = saveModuleState(predictor.module, totalMediaBytes)
	}
	return state
}

func encodeSavedExample(demo Example, totalMediaBytes *int) SavedExample {
	return SavedExample{
		Inputs:  encodeSavedFields(demo.Inputs, totalMediaBytes),
		Outputs: encodeSavedFields(demo.Outputs, totalMediaBytes),
	}
}

// encodeSavedFields stores the bytes of image and audio blocks as base64 while
// the program's total stays within DefaultMaxSavedMediaBytes. A block with no
// bytes, or one that would exceed the cap, is saved as before: without bytes.
func encodeSavedFields(fields map[string]any, totalMediaBytes *int) map[string]any {
	if fields == nil {
		return nil
	}
	encoded := make(map[string]any, len(fields))
	for k, v := range fields {
		block, ok := v.(ContentBlock)
		if ok && block.Type != FieldTypeText && len(block.Data) > 0 &&
			*totalMediaBytes+len(block.Data) <= DefaultMaxSavedMediaBytes {
			*totalMediaBytes += len(block.Data)
			encoded[k] = map[string]any{
				"__kind":    savedMediaEnvelopeType,
				"type":      string(block.Type),
				"mime_type": block.MimeType,
				"data_b64":  base64.StdEncoding.EncodeToString(block.Data),
				"metadata":  block.Metadata,
			}
			continue
		}
		encoded[k] = v
	}
	return encoded
}

func decodeSavedExample(saved SavedExample) (Example, bool, string) {
	inputs, okIn, reasonIn := decodeSavedFields(saved.Inputs)
	if !okIn {
		return Example{}, false, reasonIn
	}
	outputs, okOut, reasonOut := decodeSavedFields(saved.Outputs)
	if !okOut {
		return Example{}, false, reasonOut
	}
	return Example{Inputs: inputs, Outputs: outputs}, true, ""
}

func decodeSavedFields(fields map[string]any) (map[string]any, bool, string) {
	if fields == nil {
		return nil, true, ""
	}
	decoded := make(map[string]any, len(fields))
	for k, v := range fields {
		m, isMap := v.(map[string]any)
		if !isMap {
			decoded[k] = v
			continue
		}
		if kind, _ := m["__kind"].(string); kind == savedMediaEnvelopeType {
			b64, _ := m["data_b64"].(string)
			data, err := base64.StdEncoding.DecodeString(b64)
			if err != nil || len(data) == 0 {
				return nil, false, fmt.Sprintf("invalid or empty media bytes in field %q", k)
			}
			typ, _ := m["type"].(string)
			mime, _ := m["mime_type"].(string)
			meta, _ := m["metadata"].(map[string]any)
			decoded[k] = ContentBlock{
				Type:     FieldType(typ),
				Data:     data,
				MimeType: mime,
				Metadata: meta,
			}
			continue
		}
		decoded[k] = v
	}
	return decoded, true, ""
}

// SaveProgram serializes the current state of the Program's modules to a JSON
// file: each predictor's instruction, output prefixes, adapter, and demos,
// including predictors nested in modules such as ChainOfThought and ReAct.
func SaveProgram(p *Program, filepath string) error {
	state := SavedProgramState{
		Modules: make(map[string]SavedModuleState),
		Metadata: map[string]string{
			"dspy_go_version": Version,
		},
	}
	totalMediaBytes := 0
	for name, module := range p.Modules {
		state.Modules[name] = saveModuleState(module, &totalMediaBytes)
	}

	jsonData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal program state to JSON: %w", err)
	}
	if err := os.WriteFile(filepath, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write program state to file '%s': %w", filepath, err)
	}
	return nil
}

// LoadProgram loads program state from a JSON file into an existing Program
// instance: each predictor's instruction, output prefixes, and demos, and
// modules' tuned parameters.
// It assumes the Program `p` has already been constructed with the correct
// architecture (modules and signatures) and necessary LLMs configured.
func LoadProgram(p *Program, filepath string) error {
	return LoadProgramWithOptions(p, filepath, LoadProgramOptions{})
}

// LoadProgramWithOptions loads program state and reports any non-fatal warnings
// through the provided options instead of writing to the process logger.
func LoadProgramWithOptions(p *Program, filepath string, opts LoadProgramOptions) error {
	jsonData, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read program state file '%s': %w", filepath, err)
	}

	var loadedState SavedProgramState
	if err := json.Unmarshal(jsonData, &loadedState); err != nil {
		return fmt.Errorf("failed to unmarshal program state from JSON: %w", err)
	}

	if savedVersion, ok := loadedState.Metadata["dspy_go_version"]; ok {
		if savedVersion != Version {
			opts.warnf("loading state saved with dspy-go version %q but current version is %q; compatibility is not guaranteed", savedVersion, Version)
		}
	} else {
		opts.warnf("saved state file does not contain dspy-go version information")
	}

	for name, module := range p.Modules {
		savedModuleState, ok := loadedState.Modules[name]
		if !ok {
			opts.warnf("no saved state found for module %q in program", name)
			continue
		}
		if err := loadModuleState(name, module, savedModuleState, opts); err != nil {
			return err
		}
	}

	for savedName := range loadedState.Modules {
		if _, ok := p.Modules[savedName]; !ok {
			opts.warnf("saved state contains data for module %q, but this module was not found in the current program", savedName)
		}
	}

	return nil
}

// loadModuleState applies saved to module, named name in warnings and errors,
// and then applies the saved state of each predictor nested in module.
// Signatures other than predictors' instructions and output prefixes, and LLM
// instances, are not loaded: the program must be constructed before loading.
func loadModuleState(name string, module Module, saved SavedModuleState, opts LoadProgramOptions) error {
	if currentModuleType := persistedModuleTypeName(module); currentModuleType != saved.ModuleType {
		opts.warnf("type mismatch for module %q: program has type %q, saved state has type %q; skipping load for this module", name, currentModuleType, saved.ModuleType)
		return nil
	}

	if len(saved.LMIdentifier) > 0 {
		if lmProvider, ok := module.(LMConfigProvider); ok {
			currentIdentifier := lmProvider.GetLLMIdentifier()
			if !reflect.DeepEqual(saved.LMIdentifier, currentIdentifier) {
				opts.warnf("llm mismatch for module %q: saved state used %v, current module uses %v; loaded demos or params may behave differently", name, saved.LMIdentifier, currentIdentifier)
			}
		} else {
			opts.warnf("cannot verify llm config for module %q: module does not implement LMConfigProvider, but saved state has lm info %v", name, saved.LMIdentifier)
		}
	}

	if demoConsumer, ok := module.(DemoConsumer); ok {
		// A predictor saved with its instruction saved its demos too, so its
		// empty demos clear the constructed ones. Older files omit empty demos.
		if len(saved.Demos) > 0 || saved.Instruction != nil {
			demos := make([]Example, 0, len(saved.Demos))
			for i, savedDemo := range saved.Demos {
				decoded, keep, reason := decodeSavedExample(savedDemo)
				if !keep {
					opts.warnf("dropping demo %d for module %q: %s", i, name, reason)
					continue
				}
				demos = append(demos, decoded)
			}
			demoConsumer.SetDemos(demos)
		}
	} else if len(saved.Demos) > 0 {
		opts.warnf("saved state for module %q contains demos, but the module does not implement DemoConsumer; demos were not loaded", name)
	}

	// A file that records no adapter was saved by a module without one, so the
	// module keeps the output mode it was built with.
	if adapterConsumer, ok := module.(AdapterConsumer); ok && saved.Adapter != "" {
		if err := adapterConsumer.SetAdapterByName(saved.Adapter, saved.AdapterVersion); err != nil {
			opts.warnf("could not restore adapter %q (%s) for module %q: %v; using current adapter", saved.Adapter, saved.AdapterVersion, name, err)
		}
	}

	if paramConsumer, ok := module.(ParameterConsumer); ok {
		if len(saved.TunedParameters) > 0 {
			if err := paramConsumer.SetTunedParameters(saved.TunedParameters); err != nil {
				return fmt.Errorf("error setting tuned parameters for module '%s': %w", name, err)
			}
		}
	} else if len(saved.TunedParameters) > 0 {
		opts.warnf("saved state for module %q contains tuned parameters, but the module does not implement ParameterConsumer; parameters were not loaded", name)
	}

	predictors := nestedPredictors(module)
	if predictors == nil {
		if len(saved.Predictors) > 0 {
			opts.warnf("saved state for module %q contains nested predictors, but the module has no sub-modules; they were not loaded", name)
		}
		loadPredictorSignature(module, saved)
		return nil
	}

	for _, predictor := range predictors {
		predictorName := name + "." + predictor.path
		savedPredictor, ok := saved.Predictors[predictor.path]
		if !ok {
			// Files saved before nested predictors were saved hold only the
			// top-level state applied above.
			if len(saved.Predictors) > 0 {
				opts.warnf("no saved state found for predictor %q in program", predictorName)
			}
			continue
		}
		if err := loadModuleState(predictorName, predictor.module, savedPredictor, opts); err != nil {
			return err
		}
	}
	for path := range saved.Predictors {
		if !slices.ContainsFunc(predictors, func(predictor namedPredictor) bool { return predictor.path == path }) {
			opts.warnf("saved state contains data for predictor %q, but this predictor was not found in the current program", name+"."+path)
		}
	}
	return nil
}

// loadPredictorSignature applies a predictor's saved instruction and output
// prefixes. Files saved before instructions were saved leave the signature as
// constructed.
func loadPredictorSignature(module Module, saved SavedModuleState) {
	if saved.Instruction == nil {
		return
	}
	signature := module.GetSignature()
	changed := signature.Instruction != *saved.Instruction
	signature.Instruction = *saved.Instruction
	signature.Outputs = slices.Clone(signature.Outputs)
	for i, output := range signature.Outputs {
		if prefix, ok := saved.OutputPrefixes[output.Name]; ok && prefix != output.Prefix {
			signature.Outputs[i].Prefix = prefix
			changed = true
		}
	}
	if changed {
		module.SetSignature(signature)
	}
}
