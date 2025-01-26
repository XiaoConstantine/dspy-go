package agents

import (
	"encoding/xml"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// XMLNormalizer handles cleaning and standardization of XML content.
type XMLNormalizer struct {
	// Pattern for finding XML declarations
	xmlDeclPattern *regexp.Regexp
	// Pattern for finding markdown code blocks
	codeBlockPattern *regexp.Regexp
	// Pattern for finding tasks prefix
	tasksPrefixPattern *regexp.Regexp
}

type XMLTaskParser struct {
	// Configuration for XML parsing
	RequiredFields []string
}

type XMLTask struct {
	XMLName       xml.Name    `xml:"task"`
	ID            string      `xml:"id,attr"`
	Type          string      `xml:"type,attr"`      // Make sure this maps to the type attribute
	ProcessorType string      `xml:"processor,attr"` // Make sure this maps to the processor attribute
	Priority      int         `xml:"priority,attr"`
	Description   string      `xml:"description"`
	Dependencies  []string    `xml:"dependencies>dep"` // This maps to the <dependencies><dep>...</dep></dependencies> structure
	Metadata      XMLMetadata `xml:"metadata"`
}

type XMLMetadata struct {
	Items []XMLMetadataItem `xml:"item"`
}

type XMLMetadataItem struct {
	Key   string `xml:"key,attr"`
	Value string `xml:",chardata"`
}

// NewXMLNormalizer creates a new normalizer with compiled regex patterns.
func NewXMLNormalizer() *XMLNormalizer {
	return &XMLNormalizer{
		xmlDeclPattern:     regexp.MustCompile(`<\?xml.*?\?>`),
		codeBlockPattern:   regexp.MustCompile("```(?:xml)?\\n?"),
		tasksPrefixPattern: regexp.MustCompile(`(?m)^tasks:\s*$`),
	}
}

// NormalizeXML cleans and standardizes XML content for parsing.
func (n *XMLNormalizer) NormalizeXML(content string) (string, error) {
	// Remove any XML declaration
	content = n.xmlDeclPattern.ReplaceAllString(content, "")

	// Remove markdown code blocks
	content = n.codeBlockPattern.ReplaceAllString(content, "")

	// Remove tasks: prefix if present
	content = n.tasksPrefixPattern.ReplaceAllString(content, "")

	// Handle CDATA sections by removing CDATA markers
	content = strings.ReplaceAll(content, "<![CDATA[", "")
	content = strings.ReplaceAll(content, "]]>", "")

	// Extract just the tasks section
	startIdx := strings.Index(content, "<tasks>")
	endIdx := strings.LastIndex(content, "</tasks>")
	if startIdx == -1 || endIdx == -1 {
		return "", &XMLError{
			Code:    ErrNoTasksSection,
			Message: "could not find complete <tasks> section",
			Details: map[string]interface{}{
				"content_length": len(content),
				"start_found":    startIdx != -1,
				"end_found":      endIdx != -1,
			},
		}
	}

	content = content[startIdx : endIdx+8] // 8 is length of "</tasks>"

	// Normalize whitespace within the XML
	content = n.normalizeWhitespace(content)

	return content, nil
}

// normalizeWhitespace handles whitespace standardization.
func (n *XMLNormalizer) normalizeWhitespace(content string) string {
	// Split into lines and process each line
	lines := strings.Split(content, "\n")
	normalized := make([]string, 0, len(lines))

	for _, line := range lines {
		// Trim whitespace from each line
		line = strings.TrimSpace(line)
		if line != "" {
			normalized = append(normalized, line)
		}
	}

	// Join with newlines for pretty formatting
	return strings.Join(normalized, "\n")
}

// XMLError represents structured errors during XML processing.
type XMLError struct {
	Code    ErrorCode
	Message string
	Details map[string]interface{}
}

func (e *XMLError) Error() string {
	return e.Message
}

// ErrorCode represents specific error conditions.
type ErrorCode int

const (
	ErrNoTasksSection ErrorCode = iota
	ErrInvalidXML
	ErrMalformedTasks
)

func (p *XMLTaskParser) Parse(analyzerOutput map[string]interface{}) ([]Task, error) {
	tasksXML, ok := analyzerOutput["tasks"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid tasks format in analyzer output")
	}

	// Parse XML tasks
	var xmlTasks struct {
		Tasks []XMLTask `xml:"task"`
	}

	normalizer := NewXMLNormalizer()

	normalizedXML, err := normalizer.NormalizeXML(tasksXML)

	if err != nil {
		return nil, fmt.Errorf("failed to normalize XML: %w", err)
	}

	if err := xml.Unmarshal([]byte(normalizedXML), &xmlTasks); err != nil {
		return nil, fmt.Errorf("failed to parse XML tasks: %w", err)
	}

	// Convert to Task objects
	tasks := make([]Task, len(xmlTasks.Tasks))
	for i, xmlTask := range xmlTasks.Tasks {
		// Validate required fields
		if err := p.validateTask(xmlTask); err != nil {
			return nil, fmt.Errorf("invalid task %s: %w", xmlTask.ID, err)
		}

		// Convert metadata to map
		metadata := make(map[string]interface{})
		for _, item := range xmlTask.Metadata.Items {
			metadata[item.Key] = item.Value
		}

		tasks[i] = Task{
			ID:            xmlTask.ID,
			Type:          xmlTask.Type,
			ProcessorType: xmlTask.ProcessorType,
			Dependencies:  xmlTask.Dependencies,
			Priority:      xmlTask.Priority,
			Metadata:      metadata,
		}
	}

	return tasks, nil
}

func (p *XMLTaskParser) validateTask(task XMLTask) error {
	if task.ID == "" {
		return fmt.Errorf("missing task ID")
	}
	if task.Type == "" {
		return fmt.Errorf("missing task type")
	}
	if task.ProcessorType == "" {
		return fmt.Errorf("missing processor type")
	}
	return nil
}

// DependencyPlanCreator creates execution plans based on task dependencies.
type DependencyPlanCreator struct {
	// Optional configuration for planning
	MaxTasksPerPhase int
}

func NewDependencyPlanCreator(maxTasksPerPhase int) *DependencyPlanCreator {
	if maxTasksPerPhase <= 0 {
		maxTasksPerPhase = 10 // Default value
	}
	return &DependencyPlanCreator{
		MaxTasksPerPhase: maxTasksPerPhase,
	}
}

func (p *DependencyPlanCreator) CreatePlan(tasks []Task) ([][]Task, error) {
	// Build dependency graph
	graph := buildDependencyGraph(tasks)

	// Detect cycles
	if err := detectCycles(graph); err != nil {
		return nil, fmt.Errorf("invalid task dependencies: %w", err)
	}

	// Create phases based on dependencies
	phases := [][]Task{}
	remaining := make(map[string]Task)
	completed := make(map[string]bool)

	// Initialize remaining tasks
	for _, task := range tasks {
		remaining[task.ID] = task
	}

	// Create phases until all tasks are allocated
	for len(remaining) > 0 {
		phase := []Task{}

		// Find tasks with satisfied dependencies
		for _, task := range remaining {
			if canExecute(task, completed) {
				phase = append(phase, task)
				delete(remaining, task.ID)

				// Respect max tasks per phase
				if len(phase) >= p.MaxTasksPerPhase {
					break
				}
			}
		}

		// If no tasks can be executed, we have a problem
		if len(phase) == 0 {
			return nil, fmt.Errorf("circular dependency or missing dependency detected")
		}

		// Sort phase by priority
		sort.Slice(phase, func(i, j int) bool {
			return phase[i].Priority < phase[j].Priority
		})

		phases = append(phases, phase)

		// Mark phase tasks as completed
		for _, task := range phase {
			completed[task.ID] = true
		}
	}

	return phases, nil
}

// Helper function to build dependency graph.
func buildDependencyGraph(tasks []Task) map[string][]string {
	graph := make(map[string][]string)
	for _, task := range tasks {
		graph[task.ID] = task.Dependencies
	}
	return graph
}

// Helper function to detect cycles in the dependency graph.
func detectCycles(graph map[string][]string) error {
	visited := make(map[string]bool)
	path := make(map[string]bool)

	var checkCycle func(string) error
	checkCycle = func(node string) error {
		visited[node] = true
		path[node] = true

		for _, dep := range graph[node] {
			if !visited[dep] {
				if err := checkCycle(dep); err != nil {
					return err
				}
			} else if path[dep] {
				return fmt.Errorf("cycle detected involving task %s", node)
			}
		}

		path[node] = false
		return nil
	}

	for node := range graph {
		if !visited[node] {
			if err := checkCycle(node); err != nil {
				return err
			}
		}
	}

	return nil
}

// Helper function to check if a task can be executed.
func canExecute(task Task, completed map[string]bool) bool {
	for _, dep := range task.Dependencies {
		if !completed[dep] {
			return false
		}
	}
	return true
}
