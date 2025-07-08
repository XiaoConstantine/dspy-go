package config

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/go-playground/validator/v10"
)

// ValidationError represents a configuration validation error.
type ValidationError struct {
	Field   string
	Tag     string
	Value   interface{}
	Message string
}

func (e *ValidationError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	
	// Generate custom message based on tag
	switch e.Tag {
	case "required":
		return fmt.Sprintf("%s is required", e.Field)
	case "min":
		return fmt.Sprintf("%s must be at least", e.Field)
	case "max":
		return fmt.Sprintf("%s must be at most", e.Field)
	case "oneof":
		return fmt.Sprintf("%s must be one of", e.Field)
	case "url":
		return fmt.Sprintf("%s must be a valid URL", e.Field)
	case "":
		return fmt.Sprintf("%s failed validation", e.Field)
	default:
		return fmt.Sprintf("%s failed validation", e.Field)
	}
}

// ValidationErrors represents multiple validation errors.
type ValidationErrors []ValidationError

func (e ValidationErrors) Error() string {
	if len(e) == 0 {
		return ""
	}
	var messages []string
	for _, err := range e {
		messages = append(messages, err.Error())
	}
	return fmt.Sprintf("validation failed: %s", strings.Join(messages, "; "))
}

// Validator provides configuration validation.
type Validator struct {
	validate *validator.Validate
}

// NewValidator creates a new configuration validator.
func NewValidator() (*Validator, error) {
	validate := validator.New()

	// Register custom validation functions
	if err := registerAllValidators(validate); err != nil {
		return nil, fmt.Errorf("failed to register validators: %w", err)
	}

	return &Validator{validate: validate}, nil
}

// ValidateConfig validates a configuration struct.
func (v *Validator) ValidateConfig(config *Config) error {
	// Check for nil config first
	if config == nil {
		return ValidationErrors{
			ValidationError{
				Field:   "config",
				Tag:     "required",
				Value:   nil,
				Message: "config is nil",
			},
		}
	}

	err := v.validate.Struct(config)
	if err == nil {
		// Perform additional custom validations if struct validation passes
		if customErrors := v.validateCustomRules(config); len(customErrors) > 0 {
			return customErrors
		}
		return nil
	}

	// Convert validator errors to our custom error format
	var validationErrors ValidationErrors

	if errs, ok := err.(validator.ValidationErrors); ok {
		for _, e := range errs {
			validationErrors = append(validationErrors, ValidationError{
				Field:   e.Field(),
				Tag:     e.Tag(),
				Value:   e.Value(),
				Message: getValidationMessage(e),
			})
		}
	} else {
		validationErrors = append(validationErrors, ValidationError{
			Message: err.Error(),
		})
	}

	// Perform additional custom validations
	if customErrors := v.validateCustomRules(config); len(customErrors) > 0 {
		validationErrors = append(validationErrors, customErrors...)
	}

	if len(validationErrors) > 0 {
		return validationErrors
	}

	return nil
}

// validateCustomRules performs additional custom validation rules.
func (v *Validator) validateCustomRules(config *Config) ValidationErrors {
	var errors ValidationErrors

	// Check for nil config
	if config == nil {
		errors = append(errors, ValidationError{
			Field:   "config",
			Tag:     "required",
			Value:   nil,
			Message: "config cannot be nil",
		})
		return errors
	}

	// Validate LLM configuration consistency
	if errs := v.validateLLMConfig(&config.LLM); len(errs) > 0 {
		errors = append(errors, errs...)
	}

	// Validate logging configuration
	if errs := v.validateLoggingConfig(&config.Logging); len(errs) > 0 {
		errors = append(errors, errs...)
	}

	// Validate execution configuration
	if errs := v.validateExecutionConfig(&config.Execution); len(errs) > 0 {
		errors = append(errors, errs...)
	}


	return errors
}

// validateLLMConfig validates LLM configuration.
func (v *Validator) validateLLMConfig(config *LLMConfig) ValidationErrors {
	var errors ValidationErrors

	// Validate that default LLM provider exists in providers map
	if config.Default.Provider != "" && len(config.Providers) > 0 {
		if _, exists := config.Providers[config.Default.Provider]; !exists {
			errors = append(errors, ValidationError{
				Field:   "LLM.Default.Provider",
				Message: fmt.Sprintf("default provider '%s' not found in providers map", config.Default.Provider),
			})
		}
	}

	// Validate that teacher LLM provider exists in providers map
	if config.Teacher.Provider != "" && len(config.Providers) > 0 {
		if _, exists := config.Providers[config.Teacher.Provider]; !exists {
			errors = append(errors, ValidationError{
				Field:   "LLM.Teacher.Provider",
				Message: fmt.Sprintf("teacher provider '%s' not found in providers map", config.Teacher.Provider),
			})
		}
	}

	// Validate provider configurations
	for providerName, provider := range config.Providers {
		if errs := v.validateLLMProviderConfig(providerName, &provider); len(errs) > 0 {
			errors = append(errors, errs...)
		}
	}

	return errors
}

// validateLLMProviderConfig validates a specific LLM provider configuration.
func (v *Validator) validateLLMProviderConfig(providerName string, config *LLMProviderConfig) ValidationErrors {
	var errors ValidationErrors

	// Validate provider-specific requirements
	switch config.Provider {
	case "anthropic":
		// Note: API key validation removed - secrets should be loaded from environment
		if !isValidAnthropicModel(config.ModelID) {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("LLM.Providers.%s.ModelID", providerName),
				Message: fmt.Sprintf("invalid Anthropic model ID: %s", config.ModelID),
			})
		}

	case "google":
		// Note: API key validation removed - secrets should be loaded from environment
		if !isValidGoogleModel(config.ModelID) {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("LLM.Providers.%s.ModelID", providerName),
				Message: fmt.Sprintf("invalid Google model ID: %s", config.ModelID),
			})
		}

	case "ollama":
		if config.Endpoint.BaseURL == "" {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("LLM.Providers.%s.Endpoint.BaseURL", providerName),
				Message: "base URL is required for Ollama provider",
			})
		}
		if !strings.HasPrefix(config.ModelID, "ollama:") {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("LLM.Providers.%s.ModelID", providerName),
				Message: "Ollama model ID must start with 'ollama:'",
			})
		}

	case "llamacpp":
		if config.Endpoint.BaseURL == "" {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("LLM.Providers.%s.Endpoint.BaseURL", providerName),
				Message: "base URL is required for LlamaCPP provider",
			})
		}
	}

	return errors
}

// validateLoggingConfig validates logging configuration.
func (v *Validator) validateLoggingConfig(config *LoggingConfig) ValidationErrors {
	var errors ValidationErrors

	// Validate log outputs
	for i, output := range config.Outputs {
		if errs := v.validateLogOutput(i, &output); len(errs) > 0 {
			errors = append(errors, errs...)
		}
	}

	return errors
}

// validateLogOutput validates a log output configuration.
func (v *Validator) validateLogOutput(index int, output *LogOutputConfig) ValidationErrors {
	var errors ValidationErrors

	// Validate file output
	if output.Type == "file" {
		if output.FilePath == "" {
			errors = append(errors, ValidationError{
				Field:   fmt.Sprintf("Logging.Outputs[%d].FilePath", index),
				Message: "file path is required for file output",
			})
		} else {
			// Validate that the directory path is valid
			dir := filepath.Dir(output.FilePath)
			if !filepath.IsAbs(dir) {
				errors = append(errors, ValidationError{
					Field:   fmt.Sprintf("Logging.Outputs[%d].FilePath", index),
					Message: "log file path must be absolute",
				})
			}
		}
	}

	return errors
}

// validateExecutionConfig validates execution configuration.
func (v *Validator) validateExecutionConfig(config *ExecutionConfig) ValidationErrors {
	var errors ValidationErrors

	// Validate timeout relationships
	if config.DefaultTimeout > 0 && config.Context.DefaultTimeout > 0 {
		if config.Context.DefaultTimeout > config.DefaultTimeout {
			errors = append(errors, ValidationError{
				Field:   "Execution.Context.DefaultTimeout",
				Message: "context timeout should not exceed execution timeout",
			})
		}
	}

	// Validate tracing configuration
	if config.Tracing.Enabled {
		if config.Tracing.Exporter.Type == "" {
			errors = append(errors, ValidationError{
				Field:   "Execution.Tracing.Exporter.Type",
				Message: "exporter type is required when tracing is enabled",
			})
		}
		if config.Tracing.Exporter.Endpoint == "" {
			errors = append(errors, ValidationError{
				Field:   "Execution.Tracing.Exporter.Endpoint",
				Message: "exporter endpoint is required when tracing is enabled",
			})
		}
	}

	return errors
}


// registerAllValidators registers all custom validators.
func registerAllValidators(validate *validator.Validate) error {
	validators := map[string]validator.Func{
		"min_duration":     validateMinDuration,
		"file_path":        validateFilePath,
		"url":              validateURL,
		"provider":         validateProvider,
		"model_id":         validateModelID,
		"log_level":        validateLogLevel,
		"output_type":      validateOutputType,
		"backend_type":     validateBackendType,
		"comparison_strat": validateComparisonStrategy,
		"selection_strat":  validateSelectionStrategy,
		"exporter_type":    validateExporterType,
	}

	for name, fn := range validators {
		if err := validate.RegisterValidation(name, fn); err != nil {
			return fmt.Errorf("failed to register validator '%s': %w", name, err)
		}
	}

	return nil
}

// validateMinDuration validates minimum duration.
func validateMinDuration(fl validator.FieldLevel) bool {
	duration := fl.Field().Interface().(time.Duration)
	minDuration, err := time.ParseDuration(fl.Param())
	if err != nil {
		return false
	}
	return duration >= minDuration
}

// validateFilePath validates file paths.
func validateFilePath(fl validator.FieldLevel) bool {
	path := fl.Field().String()
	if path == "" {
		return true // Allow empty paths
	}
	// Check if it's an absolute path
	return filepath.IsAbs(path)
}

// validateURL validates URLs.
func validateURL(fl validator.FieldLevel) bool {
	url := fl.Field().String()
	if url == "" {
		return true // Allow empty URLs
	}
	// Basic URL validation
	urlRegex := regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
	return urlRegex.MatchString(url)
}

// validateProvider validates LLM provider names.
func validateProvider(fl validator.FieldLevel) bool {
	provider := fl.Field().String()
	validProviders := []string{"anthropic", "google", "ollama", "llamacpp"}
	for _, valid := range validProviders {
		if provider == valid {
			return true
		}
	}
	return false
}

// validateModelID validates model IDs.
func validateModelID(fl validator.FieldLevel) bool {
	modelID := fl.Field().String()
	if modelID == "" {
		return false
	}
	// Basic model ID validation - not empty
	return len(modelID) > 0
}

// validateLogLevel validates log levels.
func validateLogLevel(fl validator.FieldLevel) bool {
	level := fl.Field().String()
	validLevels := []string{"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
	for _, valid := range validLevels {
		if level == valid {
			return true
		}
	}
	return false
}

// validateOutputType validates output types.
func validateOutputType(fl validator.FieldLevel) bool {
	outputType := fl.Field().String()
	validTypes := []string{"console", "file", "syslog"}
	for _, valid := range validTypes {
		if outputType == valid {
			return true
		}
	}
	return false
}

// validateBackendType validates backend types.
func validateBackendType(fl validator.FieldLevel) bool {
	backendType := fl.Field().String()
	validTypes := []string{"file", "sqlite", "redis", "memory"}
	for _, valid := range validTypes {
		if backendType == valid {
			return true
		}
	}
	return false
}



// validateComparisonStrategy validates comparison strategies.
func validateComparisonStrategy(fl validator.FieldLevel) bool {
	strategy := fl.Field().String()
	validStrategies := []string{"majority_vote", "highest_confidence", "best_score"}
	for _, valid := range validStrategies {
		if strategy == valid {
			return true
		}
	}
	return false
}

// validateSelectionStrategy validates selection strategies.
func validateSelectionStrategy(fl validator.FieldLevel) bool {
	strategy := fl.Field().String()
	validStrategies := []string{"random", "tournament", "roulette"}
	for _, valid := range validStrategies {
		if strategy == valid {
			return true
		}
	}
	return false
}

// validateExporterType validates exporter types.
func validateExporterType(fl validator.FieldLevel) bool {
	exporterType := fl.Field().String()
	validTypes := []string{"jaeger", "zipkin", "otlp"}
	for _, valid := range validTypes {
		if exporterType == valid {
			return true
		}
	}
	return false
}


// Helper functions for model validation.
// Note: These functions provide basic validation. Full validation is performed
// by the provider-specific packages during LLM initialization.
func isValidAnthropicModel(modelID string) bool {
	// Allow claude-3 models - full validation happens in the provider
	return strings.HasPrefix(modelID, "claude-3")
}

func isValidGoogleModel(modelID string) bool {
	// Allow any gemini, gemma, or palm model - full validation happens in the provider
	validPrefixes := []string{
		"gemini-",
		"gemma-",
		"palm-",
	}
	for _, prefix := range validPrefixes {
		if strings.HasPrefix(modelID, prefix) {
			return true
		}
	}
	return false
}

// getValidationMessage returns a human-readable validation message.
func getValidationMessage(e validator.FieldError) string {
	switch e.Tag() {
	case "required":
		return fmt.Sprintf("%s is required", e.Field())
	case "min":
		return fmt.Sprintf("%s must be at least %s", e.Field(), e.Param())
	case "max":
		return fmt.Sprintf("%s must be at most %s", e.Field(), e.Param())
	case "oneof":
		return fmt.Sprintf("%s must be one of: %s", e.Field(), e.Param())
	case "url":
		return fmt.Sprintf("%s must be a valid URL", e.Field())
	case "file_path":
		return fmt.Sprintf("%s must be a valid file path", e.Field())
	case "min_duration":
		return fmt.Sprintf("%s must be at least %s", e.Field(), e.Param())
	default:
		return fmt.Sprintf("%s failed validation", e.Field())
	}
}

// Global validator instance.
var (
	globalValidator *Validator
	validatorOnce   sync.Once
)

// GetValidator returns the global validator instance.
func GetValidator() *Validator {
	validatorOnce.Do(func() {
		var err error
		globalValidator, err = NewValidator()
		if err != nil {
			panic(fmt.Sprintf("failed to create global validator: %v", err))
		}
	})
	return globalValidator
}

// ValidateConfiguration validates a configuration using the global validator.
func ValidateConfiguration(config *Config) error {
	return GetValidator().ValidateConfig(config)
}
