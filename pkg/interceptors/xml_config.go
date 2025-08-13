package interceptors

import (
	"time"
)

// XMLConfig holds configuration for XML interceptors.
type XMLConfig struct {
	// StrictParsing requires all output fields to be present in XML
	StrictParsing bool

	// FallbackToText enables text fallback for malformed XML
	FallbackToText bool

	// ValidateXML performs XML syntax validation before parsing
	ValidateXML bool

	// MaxDepth limits XML nesting depth for security (default: 10)
	MaxDepth int

	// MaxSize limits XML response size in bytes (default: 1MB)
	MaxSize int64

	// Timeout for XML parsing operations
	ParseTimeout time.Duration

	// CustomTags allows overriding default XML tag names for fields
	CustomTags map[string]string

	// IncludeTypeHints adds type information to XML instructions
	IncludeTypeHints bool

	// PreserveWhitespace maintains whitespace in XML content
	PreserveWhitespace bool
}

// DefaultXMLConfig returns XMLConfig with sensible defaults.
func DefaultXMLConfig() XMLConfig {
	return XMLConfig{
		StrictParsing:      true,
		FallbackToText:     true,
		ValidateXML:        true,
		MaxDepth:           10,
		MaxSize:            1024 * 1024, // 1MB
		ParseTimeout:       30 * time.Second,
		CustomTags:         make(map[string]string),
		IncludeTypeHints:   false,
		PreserveWhitespace: false,
	}
}

// WithStrictParsing sets strict parsing mode.
func (c XMLConfig) WithStrictParsing(strict bool) XMLConfig {
	c.StrictParsing = strict
	return c
}

// WithFallback enables/disables text fallback.
func (c XMLConfig) WithFallback(fallback bool) XMLConfig {
	c.FallbackToText = fallback
	return c
}

// WithValidation enables/disables XML validation.
func (c XMLConfig) WithValidation(validate bool) XMLConfig {
	c.ValidateXML = validate
	return c
}

// WithMaxDepth sets maximum XML nesting depth.
func (c XMLConfig) WithMaxDepth(depth int) XMLConfig {
	c.MaxDepth = depth
	return c
}

// WithMaxSize sets maximum XML response size.
func (c XMLConfig) WithMaxSize(size int64) XMLConfig {
	c.MaxSize = size
	return c
}

// WithTimeout sets parsing timeout.
func (c XMLConfig) WithTimeout(timeout time.Duration) XMLConfig {
	c.ParseTimeout = timeout
	return c
}

// WithCustomTag sets a custom XML tag for a field.
func (c XMLConfig) WithCustomTag(fieldName, tagName string) XMLConfig {
	if c.CustomTags == nil {
		c.CustomTags = make(map[string]string)
	}
	c.CustomTags[fieldName] = tagName
	return c
}

// WithTypeHints enables/disables type hints in XML instructions.
func (c XMLConfig) WithTypeHints(hints bool) XMLConfig {
	c.IncludeTypeHints = hints
	return c
}

// WithPreserveWhitespace enables/disables whitespace preservation.
func (c XMLConfig) WithPreserveWhitespace(preserve bool) XMLConfig {
	c.PreserveWhitespace = preserve
	return c
}

// GetTagName returns the XML tag name for a field.
func (c XMLConfig) GetTagName(fieldName string) string {
	if tag, exists := c.CustomTags[fieldName]; exists {
		return tag
	}
	return fieldName
}

// Preset configurations

// StrictXMLConfig creates a configuration with strict parsing requirements.
func StrictXMLConfig() XMLConfig {
	return DefaultXMLConfig().
		WithStrictParsing(true).
		WithFallback(false).
		WithValidation(true)
}

// FlexibleXMLConfig creates a configuration with flexible parsing (allows fallback).
func FlexibleXMLConfig() XMLConfig {
	return DefaultXMLConfig().
		WithStrictParsing(false).
		WithFallback(true).
		WithValidation(false)
}

// PerformantXMLConfig creates a configuration optimized for performance.
func PerformantXMLConfig() XMLConfig {
	return DefaultXMLConfig().
		WithValidation(false). // Skip validation for speed
		WithMaxDepth(5).       // Limit depth for performance
		WithMaxSize(512 * 1024) // Limit size to 512KB
}

// SecureXMLConfig creates a configuration with enhanced security settings.
func SecureXMLConfig() XMLConfig {
	return DefaultXMLConfig().
		WithMaxDepth(3).        // Very limited depth
		WithMaxSize(64 * 1024). // Small size limit
		WithValidation(true).   // Always validate
		WithStrictParsing(true) // Strict requirements
}
