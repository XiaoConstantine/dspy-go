package interceptors

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// XMLFormatModuleInterceptor modifies module inputs to request XML-formatted responses.
// This interceptor operates on the input side, injecting XML formatting instructions
// into the prompt to guide the LLM to produce structured XML output.
func XMLFormatModuleInterceptor(config XMLConfig) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Create modified inputs with XML formatting instructions
		modifiedInputs := make(map[string]any)
		for k, v := range inputs {
			modifiedInputs[k] = v
		}

		// Generate XML formatting instructions based on the signature
		xmlInstructions := generateXMLInstructions(info.Signature, config)

		// Inject XML instructions into the appropriate input field
		if err := injectXMLInstructions(modifiedInputs, xmlInstructions, info.Signature); err != nil {
			return nil, fmt.Errorf("failed to inject XML instructions: %w", err)
		}

		// Call the next handler with modified inputs
		return handler(ctx, modifiedInputs, opts...)
	}
}

// XMLParseModuleInterceptor extracts structured data from XML responses.
// This interceptor operates on the output side, parsing XML-formatted LLM responses
// into structured field values according to the module's signature.
func XMLParseModuleInterceptor(config XMLConfig) core.ModuleInterceptor {
	parser := &XMLParser{
		config: config,
		cache:  make(map[string]*ParsedSignature),
	}

	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Call the handler first to get the raw outputs
		outputs, err := handler(ctx, inputs, opts...)
		if err != nil {
			return nil, err
		}

		// Parse XML from the outputs
		parsedOutputs, err := parser.ParseXMLOutputs(ctx, outputs, info.Signature)
		if err != nil {
			if config.FallbackToText {
				// Return original outputs if parsing fails and fallback is enabled
				return outputs, nil
			}
			return nil, fmt.Errorf("XML parsing failed: %w", err)
		}

		return parsedOutputs, nil
	}
}

// XMLModuleInterceptor creates a combined interceptor that both formats requests and parses responses.
// This is a convenience function that chains the format and parse interceptors.
func XMLModuleInterceptor(config XMLConfig) core.ModuleInterceptor {
	formatInterceptor := XMLFormatModuleInterceptor(config)
	parseInterceptor := XMLParseModuleInterceptor(config)

	return core.ChainModuleInterceptors(formatInterceptor, parseInterceptor)
}

// XMLParser handles the actual XML parsing logic.
type XMLParser struct {
	config XMLConfig
	cache  map[string]*ParsedSignature
	mutex  sync.RWMutex
}

// ParsedSignature caches parsing information for a signature.
type ParsedSignature struct {
	FieldMap    map[string]core.OutputField
	TagMap      map[string]string // tag -> field name
	RequiredSet map[string]bool
}

// ParseXMLOutputs extracts structured data from XML-formatted outputs.
func (p *XMLParser) ParseXMLOutputs(ctx context.Context, outputs map[string]any, signature core.Signature) (map[string]any, error) {
	// Find the response field (usually the main output)
	responseText := p.findResponseText(outputs)
	if responseText == "" {
		return outputs, nil // No text to parse
	}

	// Check size limits
	if len(responseText) > int(p.config.MaxSize) {
		return nil, fmt.Errorf("XML response size (%d bytes) exceeds limit (%d bytes)",
			len(responseText), p.config.MaxSize)
	}

	// Parse with timeout
	ctx, cancel := context.WithTimeout(ctx, p.config.ParseTimeout)
	defer cancel()

	parsedFields, err := p.parseXMLWithTimeout(ctx, responseText, signature)
	if err != nil {
		return nil, err
	}

	// Merge parsed fields with original outputs
	result := make(map[string]any)
	for k, v := range outputs {
		result[k] = v
	}
	for k, v := range parsedFields {
		result[k] = v
	}

	return result, nil
}

// findResponseText locates the text content to parse from outputs.
func (p *XMLParser) findResponseText(outputs map[string]any) string {
	// Priority order for finding response text
	candidates := []string{"response", "output", "result", "answer", "text"}

	for _, candidate := range candidates {
		if text, exists := outputs[candidate]; exists {
			if textStr, ok := text.(string); ok && textStr != "" {
				return textStr
			}
		}
	}

	// Fallback: use first string value
	for _, value := range outputs {
		if textStr, ok := value.(string); ok && textStr != "" {
			return textStr
		}
	}

	return ""
}

// parseXMLWithTimeout parses XML with context timeout support.
func (p *XMLParser) parseXMLWithTimeout(ctx context.Context, responseText string, signature core.Signature) (map[string]any, error) {
	// Channel for result communication
	resultChan := make(chan parseResult, 1)

	go func() {
		result, err := p.parseXML(responseText, signature)
		resultChan <- parseResult{result: result, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("XML parsing timeout: %w", ctx.Err())
	case res := <-resultChan:
		return res.result, res.err
	}
}

type parseResult struct {
	result map[string]any
	err    error
}

// parseXML performs the actual XML parsing.
func (p *XMLParser) parseXML(responseText string, signature core.Signature) (map[string]any, error) {
	// Get or create cached signature info
	sigInfo := p.getSignatureInfo(signature)

	// Pre-validate XML if enabled
	if p.config.ValidateXML {
		if err := p.validateXMLSyntax(responseText); err != nil {
			return nil, fmt.Errorf("XML validation failed: %w", err)
		}
	}

	// Extract XML content (handle cases where XML is embedded in text)
	xmlContent := p.extractXMLContent(responseText)
	if xmlContent == "" {
		return nil, fmt.Errorf("no XML content found in response")
	}

	// Parse XML using Go's encoding/xml
	decoder := xml.NewDecoder(strings.NewReader(xmlContent))
	decoder.CharsetReader = p.charsetReader

	fields := make(map[string]any)
	depth := 0
	var currentTag string

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("XML parsing error: %w", err)
		}

		switch t := token.(type) {
		case xml.StartElement:
			depth++
			if depth > p.config.MaxDepth {
				return nil, fmt.Errorf("XML depth limit exceeded: %d", depth)
			}
			currentTag = t.Name.Local

		case xml.EndElement:
			depth--
			currentTag = ""

		case xml.CharData:
			if currentTag != "" {
				if fieldName, exists := sigInfo.TagMap[currentTag]; exists {
					content := string(t)
					if !p.config.PreserveWhitespace {
						content = strings.TrimSpace(content)
					}

					// Type conversion based on field type
					if field, exists := sigInfo.FieldMap[fieldName]; exists {
						typedValue, err := p.convertFieldValue(content, field)
						if err != nil {
							return nil, fmt.Errorf("field %s conversion failed: %w", fieldName, err)
						}
						fields[fieldName] = typedValue
					}
				}
			}
		}
	}

	// Validate required fields if strict parsing is enabled
	if p.config.StrictParsing {
		if err := p.validateRequiredFields(fields, sigInfo); err != nil {
			return nil, err
		}
	}

	return fields, nil
}

// getSignatureInfo retrieves or creates cached signature parsing information.
func (p *XMLParser) getSignatureInfo(signature core.Signature) *ParsedSignature {
	key := p.signatureKey(signature)

	p.mutex.RLock()
	if cached, exists := p.cache[key]; exists {
		p.mutex.RUnlock()
		return cached
	}
	p.mutex.RUnlock()

	// Create new signature info
	sigInfo := &ParsedSignature{
		FieldMap:    make(map[string]core.OutputField),
		TagMap:      make(map[string]string),
		RequiredSet: make(map[string]bool),
	}

	for _, output := range signature.Outputs {
		sigInfo.FieldMap[output.Name] = output
		tagName := p.config.GetTagName(output.Name)
		sigInfo.TagMap[tagName] = output.Name
		sigInfo.RequiredSet[output.Name] = true // Consider all outputs required for now
	}

	// Cache the result
	p.mutex.Lock()
	p.cache[key] = sigInfo
	p.mutex.Unlock()

	return sigInfo
}

// signatureKey generates a cache key for the signature.
func (p *XMLParser) signatureKey(signature core.Signature) string {
	var sb strings.Builder
	for _, output := range signature.Outputs {
		sb.WriteString(output.Name)
		sb.WriteString(":")
		sb.WriteString(string(output.Type))
		sb.WriteString(";")
	}
	return sb.String()
}

// validateXMLSyntax performs basic XML syntax validation.
func (p *XMLParser) validateXMLSyntax(xmlText string) error {
	decoder := xml.NewDecoder(strings.NewReader(xmlText))
	for {
		_, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// extractXMLContent extracts XML from potentially mixed content.
func (p *XMLParser) extractXMLContent(text string) string {
	// Look for XML tags
	start := strings.Index(text, "<")
	if start == -1 {
		return ""
	}

	end := strings.LastIndex(text, ">")
	if end == -1 || end <= start {
		return ""
	}

	return text[start : end+1]
}

// convertFieldValue converts string content to the appropriate type.
func (p *XMLParser) convertFieldValue(content string, field core.OutputField) (any, error) {
	switch field.Type {
	case core.FieldTypeText, "":
		return content, nil
	case core.FieldTypeImage:
		// For image fields, return the content as-is (could be URL or description)
		return content, nil
	case core.FieldTypeAudio:
		// For audio fields, return the content as-is (could be URL or description)
		return content, nil
	default:
		// Try to infer type from content
		return p.inferTypeFromContent(content)
	}
}

// inferTypeFromContent attempts to infer and convert content type.
func (p *XMLParser) inferTypeFromContent(content string) (any, error) {
	content = strings.TrimSpace(content)

	// Try boolean
	if strings.EqualFold(content, "true") {
		return true, nil
	}
	if strings.EqualFold(content, "false") {
		return false, nil
	}

	// Try integer
	if intVal, err := strconv.ParseInt(content, 10, 64); err == nil {
		return intVal, nil
	}

	// Try float
	if floatVal, err := strconv.ParseFloat(content, 64); err == nil {
		return floatVal, nil
	}

	// Default to string
	return content, nil
}

// validateRequiredFields checks that all required fields are present.
func (p *XMLParser) validateRequiredFields(fields map[string]any, sigInfo *ParsedSignature) error {
	for fieldName := range sigInfo.RequiredSet {
		if _, exists := fields[fieldName]; !exists {
			return fmt.Errorf("required field missing: %s", fieldName)
		}
	}
	return nil
}

// charsetReader handles charset encoding for XML decoder.
func (p *XMLParser) charsetReader(charset string, input io.Reader) (io.Reader, error) {
	if charset != "utf-8" && charset != "UTF-8" && charset != "" {
		return nil, fmt.Errorf("unsupported charset: %s", charset)
	}
	return input, nil
}

// Helper functions for XML instruction generation

// generateXMLInstructions creates formatting instructions for the LLM.
func generateXMLInstructions(signature core.Signature, config XMLConfig) string {
	var sb strings.Builder

	sb.WriteString("Please format your response using the following XML structure:\n\n")

	// Generate example XML structure
	sb.WriteString("<response>\n")
	for _, output := range signature.Outputs {
		tagName := config.GetTagName(output.Name)

		if config.IncludeTypeHints {
			typeHint := getTypeHint(output.Type)
			sb.WriteString(fmt.Sprintf("  <%s>%s</%s>\n", tagName, typeHint, tagName))
		} else {
			sb.WriteString(fmt.Sprintf("  <%s>[your %s here]</%s>\n", tagName, output.Description, tagName))
		}
	}
	sb.WriteString("</response>\n\n")

	// Add parsing requirements
	sb.WriteString("Requirements:\n")
	sb.WriteString("- Use exactly the XML tags shown above\n")
	sb.WriteString("- Ensure proper XML formatting with opening and closing tags\n")

	if config.StrictParsing {
		sb.WriteString("- Include ALL required fields\n")
	}

	if !config.PreserveWhitespace {
		sb.WriteString("- Avoid unnecessary whitespace within tags\n")
	}

	sb.WriteString("- Do not include any text outside the XML structure\n")

	return sb.String()
}

// injectXMLInstructions adds XML formatting instructions to the input.
func injectXMLInstructions(inputs map[string]any, instructions string, signature core.Signature) error {
	// Find the best input field to inject instructions into
	targetField := findInstructionTargetField(signature)
	if targetField == "" {
		return fmt.Errorf("no suitable input field found for XML instructions")
	}

	// Get current value and append instructions
	currentValue, exists := inputs[targetField]
	if !exists {
		inputs[targetField] = instructions
		return nil
	}

	// Convert to string and append
	currentStr := fmt.Sprintf("%v", currentValue)
	inputs[targetField] = currentStr + "\n\n" + instructions

	return nil
}

// findInstructionTargetField identifies the best input field for XML instructions.
func findInstructionTargetField(signature core.Signature) string {
	// Priority order for instruction injection
	preferredFields := []string{"instruction", "prompt", "query", "question", "input"}

	// First, try preferred field names
	for _, preferred := range preferredFields {
		for _, input := range signature.Inputs {
			if strings.EqualFold(input.Name, preferred) {
				return input.Name
			}
		}
	}

	// Fallback to first text field
	for _, input := range signature.Inputs {
		if input.Type == core.FieldTypeText || input.Type == "" {
			return input.Name
		}
	}

	// Last resort: use first input field
	if len(signature.Inputs) > 0 {
		return signature.Inputs[0].Name
	}

	return ""
}

// getTypeHint returns a type hint for XML instructions.
func getTypeHint(fieldType core.FieldType) string {
	switch fieldType {
	case core.FieldTypeText:
		return "[text content]"
	case core.FieldTypeImage:
		return "[image description or URL]"
	case core.FieldTypeAudio:
		return "[audio description or URL]"
	default:
		return "[content]"
	}
}

// XMLHelpers provides utility functions for working with XML interceptors

// ApplyXMLInterceptors applies XML format and parse interceptors to a module.
// This is a convenience function for modules that implement InterceptableModule.
func ApplyXMLInterceptors(module core.Module, config XMLConfig) error {
	if interceptableModule, ok := module.(core.InterceptableModule); ok {
		xmlInterceptor := XMLModuleInterceptor(config)
		existing := interceptableModule.GetInterceptors()
		combined := append(existing, xmlInterceptor)
		interceptableModule.SetInterceptors(combined)
		return nil
	}
	return fmt.Errorf("module does not implement InterceptableModule")
}

// CreateXMLInterceptorChain creates a chain of XML and other interceptors.
func CreateXMLInterceptorChain(config XMLConfig, additionalInterceptors ...core.ModuleInterceptor) []core.ModuleInterceptor {
	formatInterceptor := XMLFormatModuleInterceptor(config)
	parseInterceptor := XMLParseModuleInterceptor(config)

	// Build chain: format -> additional interceptors -> parse
	chain := make([]core.ModuleInterceptor, 0, 2+len(additionalInterceptors))
	chain = append(chain, formatInterceptor)
	chain = append(chain, additionalInterceptors...)
	chain = append(chain, parseInterceptor)

	return chain
}
