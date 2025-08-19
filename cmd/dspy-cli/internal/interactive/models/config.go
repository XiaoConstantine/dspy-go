package models

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/samples"
)

// ConfigModel represents the configuration screen state
type ConfigModel struct {
	optimizer     string
	selectedField int
	config        runner.OptimizerConfig
	fields        []ConfigField
	width         int
	height        int
	nextScreen    string
	editing       bool
	editValue     string
}

// ConfigField represents a configurable parameter
type ConfigField struct {
	Name        string
	Description string
	Value       interface{}
	Type        string // "int", "bool", "string", "select"
	Options     []string // For select type
	Min         int
	Max         int
}

// NewConfigModel creates a new configuration model
func NewConfigModel(optimizerName string) ConfigModel {
	config := runner.OptimizerConfig{
		OptimizerName: optimizerName,
		DatasetName:   "gsm8k", // Default dataset (will be updated by updateConfig)
		APIKey:        "",
		MaxExamples:   5,
		Verbose:       false,
		SuppressLogs:  true,
	}

	fields := getFieldsForOptimizer(optimizerName)

	return ConfigModel{
		optimizer:     optimizerName,
		selectedField: 0,
		config:        config,
		fields:        fields,
		editing:       false,
		width:         80,  // Default width to prevent loading screen
		height:        24,  // Default height to prevent loading screen
	}
}

// getFieldsForOptimizer returns configurable fields for each optimizer
func getFieldsForOptimizer(optimizer string) []ConfigField {
	baseFields := []ConfigField{
		{
			Name:        "Dataset",
			Description: "Test dataset to use for optimization",
			Value:       "gsm8k (Grade School Math 8K)",
			Type:        "select",
			Options:     samples.ListAvailableDatasets(),
		},
		{
			Name:        "Max Examples",
			Description: "Maximum number of examples to use (higher = more accurate, slower)",
			Value:       5,
			Type:        "int",
			Min:         1,
			Max:         20,
		},
		{
			Name:        "Verbose Logging",
			Description: "Enable detailed logging output",
			Value:       false,
			Type:        "bool",
		},
	}

	// Add optimizer-specific fields
	switch optimizer {
	case "mipro":
		optimizerFields := []ConfigField{
			{
				Name:        "Number of Trials",
				Description: "Number of optimization trials (higher = better results, slower)",
				Value:       3,
				Type:        "int",
				Min:         1,
				Max:         10,
			},
			{
				Name:        "Mini Batch Size",
				Description: "Size of mini-batches for optimization",
				Value:       2,
				Type:        "int",
				Min:         1,
				Max:         10,
			},
		}
		baseFields = append(baseFields, optimizerFields...)
	case "simba":
		optimizerFields := []ConfigField{
			{
				Name:        "Batch Size",
				Description: "Batch size for SIMBA optimization",
				Value:       2,
				Type:        "int",
				Min:         1,
				Max:         10,
			},
			{
				Name:        "Fast Mode",
				Description: "Enable fast mode for quicker optimization",
				Value:       true,
				Type:        "bool",
			},
		}
		baseFields = append(baseFields, optimizerFields...)
	case "gepa":
		optimizerFields := []ConfigField{
			{
				Name:        "Population Size",
				Description: "Size of evolutionary population",
				Value:       4,
				Type:        "int",
				Min:         2,
				Max:         20,
			},
			{
				Name:        "Max Generations",
				Description: "Maximum number of evolutionary generations",
				Value:       2,
				Type:        "int",
				Min:         1,
				Max:         10,
			},
		}
		baseFields = append(baseFields, optimizerFields...)
	}

	return baseFields
}

// Init initializes the model
func (m ConfigModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m ConfigModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		if m.editing {
			switch msg.String() {
			case "enter":
				m.editing = false
				m.updateFieldValue()
			case "esc":
				m.editing = false
				m.editValue = ""
			case "backspace":
				if len(m.editValue) > 0 {
					m.editValue = m.editValue[:len(m.editValue)-1]
				}
			default:
				// Add character to edit value
				if len(msg.String()) == 1 {
					m.editValue += msg.String()
				}
			}
			return m, nil
		}

		switch msg.String() {
		case "up", "k":
			if m.selectedField > 0 {
				m.selectedField--
			} else {
				m.selectedField = len(m.fields) - 1
			}
		case "down", "j":
			if m.selectedField < len(m.fields)-1 {
				m.selectedField++
			} else {
				m.selectedField = 0
			}
		case "enter", " ":
			m.startEditing()
		case "r":
			// Run with current configuration
			m.nextScreen = "run"
		case "b", "esc":
			m.nextScreen = "back"
		case "q":
			return m, tea.Quit
		}
	}

	return m, nil
}

// startEditing begins editing the selected field
func (m *ConfigModel) startEditing() {
	field := m.fields[m.selectedField]

	if field.Type == "bool" {
		// Toggle boolean values directly
		m.fields[m.selectedField].Value = !m.fields[m.selectedField].Value.(bool)
		m.updateConfig()
	} else if field.Type == "select" {
		// Cycle through select options
		options := field.Options
		currentValue := field.Value.(string)
		currentIndex := 0
		for i, option := range options {
			if option == currentValue {
				currentIndex = i
				break
			}
		}
		nextIndex := (currentIndex + 1) % len(options)
		m.fields[m.selectedField].Value = options[nextIndex]
		m.updateConfig()
	} else {
		// Start editing text/number fields
		m.editing = true
		m.editValue = fmt.Sprintf("%v", field.Value)
	}
}

// updateFieldValue updates the field value from edit input
func (m *ConfigModel) updateFieldValue() {
	field := &m.fields[m.selectedField]

	if field.Type == "int" {
		if val, err := strconv.Atoi(m.editValue); err == nil {
			if val >= field.Min && val <= field.Max {
				field.Value = val
			}
		}
	} else if field.Type == "string" {
		field.Value = m.editValue
	}

	m.editValue = ""
	m.updateConfig()
}

// updateConfig updates the internal config based on field values
func (m *ConfigModel) updateConfig() {
	for _, field := range m.fields {
		switch field.Name {
		case "Dataset":
			datasetDisplay := field.Value.(string)
			m.config.DatasetName = extractDatasetKey(datasetDisplay)
		case "Max Examples":
			m.config.MaxExamples = field.Value.(int)
		case "Verbose Logging":
			m.config.Verbose = field.Value.(bool)
		}
	}
}

// View renders the configuration screen
func (m ConfigModel) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	var content []string

	// Header
	header := styles.TitleStyle.Render(fmt.Sprintf("⚙️ Configure %s", strings.ToUpper(m.optimizer)))
	content = append(content, header)
	content = append(content, "")

	// Instructions
	instructions := styles.BodyStyle.Render("Customize optimization parameters for your specific needs")
	content = append(content, instructions)
	content = append(content, "")

	// Configuration fields
	for i, field := range m.fields {
		var fieldStr string

		// Field name and description
		fieldName := styles.HeadingStyle.Render(field.Name)
		fieldDesc := styles.CaptionStyle.Render(field.Description)

		// Field value with appropriate styling
		var valueStr string
		if m.editing && i == m.selectedField {
			valueStr = styles.HighlightStyle.Render(fmt.Sprintf("[%s_]", m.editValue))
		} else {
			switch field.Type {
			case "bool":
				if field.Value.(bool) {
					valueStr = styles.SuccessStyle.Render("✓ Enabled")
				} else {
					valueStr = styles.MutedStyle.Render("✗ Disabled")
				}
			case "select":
				valueStr = styles.InfoStyle.Render(fmt.Sprintf("› %s", field.Value))
			default:
				valueStr = styles.InfoStyle.Render(fmt.Sprintf("%v", field.Value))
			}
		}

		if i == m.selectedField && !m.editing {
			fieldStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s", styles.IconSelected, fieldName))
			fieldStr += "\n   " + fieldDesc
			fieldStr += "\n   " + valueStr
		} else {
			fieldStr = styles.UnselectedStyle.Render(fmt.Sprintf("  %s", fieldName))
			fieldStr += "\n   " + fieldDesc
			fieldStr += "\n   " + valueStr
		}

		content = append(content, fieldStr)
		content = append(content, "")
	}

	// Footer with controls
	var controls []string
	if m.editing {
		controls = append(controls, "[Type] Edit value", "[Enter] Save", "[Esc] Cancel")
	} else {
		controls = append(controls, "[↑↓/jk] Navigate", "[Enter/Space] Edit", "[r] Run optimization", "[b] Back", "[q] Quit")
	}

	footer := styles.FooterStyle.Render(strings.Join(controls, "  "))
	content = append(content, footer)

	// Join all content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Apply box styling and center
	boxed := styles.BoxStyle.Copy().
		Width(min(m.width-4, 100)).
		Render(fullContent)

	// Center the box
	return lipgloss.Place(
		m.width,
		m.height,
		lipgloss.Center,
		lipgloss.Center,
		boxed,
	)
}

// GetNextScreen returns the next screen to navigate to
func (m ConfigModel) GetNextScreen() string {
	return m.nextScreen
}

// GetConfig returns the current configuration
func (m ConfigModel) GetConfig() runner.OptimizerConfig {
	m.updateConfig() // Ensure config is up to date
	return m.config
}

// ResetNavigation clears the next screen state
func (m *ConfigModel) ResetNavigation() {
	m.nextScreen = ""
}
