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
	presets       []Preset
	selectedPreset int
	showPresets   bool
	advancedMode  bool
}

// ConfigField represents a configurable parameter
type ConfigField struct {
	Name        string
	Description string
	Value       interface{}
	Type        string // "int", "bool", "string", "select", "float", "slider"
	Options     []string // For select type
	Min         int
	Max         int
	Step        float64 // For float/slider types
	Advanced    bool    // Show only in advanced mode
	Category    string  // Group related fields
}

// Preset represents a configuration preset
type Preset struct {
	Name        string
	Description string
	Icon        string
	Values      map[string]interface{}
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
	presets := getPresetsForOptimizer(optimizerName)

	return ConfigModel{
		optimizer:     optimizerName,
		selectedField: 0,
		config:        config,
		fields:        fields,
		presets:       presets,
		editing:       false,
		advancedMode:  false,
		showPresets:   false,
		width:         80,  // Default width to prevent loading screen
		height:        24,  // Default height to prevent loading screen
	}
}

// getPresetsForOptimizer returns configuration presets for each optimizer
func getPresetsForOptimizer(optimizer string) []Preset {
	basePresets := []Preset{
		{
			Name:        "Quick Test",
			Description: "Fast testing with minimal examples",
			Icon:        "⚡",
			Values: map[string]interface{}{
				"Max Examples": 3,
				"Temperature":  float64(0.5),
				"Cache Results": true,
			},
		},
		{
			Name:        "Balanced",
			Description: "Good balance of speed and accuracy",
			Icon:        "⚖️",
			Values: map[string]interface{}{
				"Max Examples": 10,
				"Temperature":  float64(0.7),
				"Cache Results": true,
			},
		},
		{
			Name:        "High Quality",
			Description: "Maximum accuracy, slower execution",
			Icon:        "🎯",
			Values: map[string]interface{}{
				"Max Examples": 20,
				"Temperature":  float64(0.3),
				"Cache Results": false,
			},
		},
	}

	// Add optimizer-specific presets
	switch optimizer {
	case "mipro":
		basePresets = append(basePresets, Preset{
			Name:        "Research Mode",
			Description: "Thorough exploration for research",
			Icon:        "🔬",
			Values: map[string]interface{}{
				"Number of Trials": 10,
				"Mini Batch Size": 5,
				"Learning Rate": float64(0.001),
				"Max Examples": 15,
			},
		})
	case "gepa":
		basePresets = append(basePresets, Preset{
			Name:        "Evolution Max",
			Description: "Maximum evolutionary exploration",
			Icon:        "🧬",
			Values: map[string]interface{}{
				"Population Size": 20,
				"Max Generations": 10,
				"Mutation Rate": float64(0.2),
				"Crossover Rate": float64(0.9),
			},
		})
	}

	return basePresets
}

// getFieldsForOptimizer returns configurable fields for each optimizer
func getFieldsForOptimizer(optimizer string) []ConfigField {
	baseFields := []ConfigField{
		// Basic Fields
		{
			Name:        "Dataset",
			Description: "Test dataset to use for optimization",
			Value:       "gsm8k (Grade School Math 8K)",
			Type:        "select",
			Options:     samples.ListAvailableDatasets(),
			Category:    "Basic",
		},
		{
			Name:        "Max Examples",
			Description: "Maximum number of examples to use (higher = more accurate, slower)",
			Value:       5,
			Type:        "slider",
			Min:         1,
			Max:         20,
			Step:        1,
			Category:    "Basic",
		},
		{
			Name:        "Verbose Logging",
			Description: "Enable detailed logging output",
			Value:       false,
			Type:        "bool",
			Category:    "Basic",
		},
		// Advanced Fields
		{
			Name:        "Temperature",
			Description: "LLM temperature for generation (0.0 = deterministic, 1.0 = creative)",
			Value:       float64(0.7),
			Type:        "slider",
			Min:         0,
			Max:         1,
			Step:        0.1,
			Advanced:    true,
			Category:    "Advanced",
		},
		{
			Name:        "API Timeout",
			Description: "API request timeout in seconds",
			Value:       30,
			Type:        "int",
			Min:         10,
			Max:         120,
			Advanced:    true,
			Category:    "Advanced",
		},
		{
			Name:        "Cache Results",
			Description: "Cache optimization results for faster subsequent runs",
			Value:       true,
			Type:        "bool",
			Advanced:    true,
			Category:    "Advanced",
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
				Type:        "slider",
				Min:         1,
				Max:         10,
				Step:        1,
				Category:    "Optimizer",
			},
			{
				Name:        "Mini Batch Size",
				Description: "Size of mini-batches for optimization",
				Value:       2,
				Type:        "slider",
				Min:         1,
				Max:         10,
				Step:        1,
				Category:    "Optimizer",
			},
			{
				Name:        "Learning Rate",
				Description: "Learning rate for gradient-based optimization",
				Value:       float64(0.01),
				Type:        "slider",
				Min:         0,
				Max:         1,
				Step:        0.001,
				Advanced:    true,
				Category:    "Optimizer",
			},
			{
				Name:        "Use TPE",
				Description: "Use Tree-structured Parzen Estimator for hyperparameter optimization",
				Value:       true,
				Type:        "bool",
				Advanced:    true,
				Category:    "Optimizer",
			},
		}
		baseFields = append(baseFields, optimizerFields...)
	case "simba":
		optimizerFields := []ConfigField{
			{
				Name:        "Batch Size",
				Description: "Batch size for SIMBA optimization",
				Value:       2,
				Type:        "slider",
				Min:         1,
				Max:         10,
				Step:        1,
				Category:    "Optimizer",
			},
			{
				Name:        "Fast Mode",
				Description: "Enable fast mode for quicker optimization",
				Value:       true,
				Type:        "bool",
				Category:    "Optimizer",
			},
			{
				Name:        "Exploration Rate",
				Description: "Rate of exploration vs exploitation",
				Value:       float64(0.3),
				Type:        "slider",
				Min:         0,
				Max:         1,
				Step:        0.05,
				Advanced:    true,
				Category:    "Optimizer",
			},
		}
		baseFields = append(baseFields, optimizerFields...)
	case "gepa":
		optimizerFields := []ConfigField{
			{
				Name:        "Population Size",
				Description: "Size of evolutionary population",
				Value:       4,
				Type:        "slider",
				Min:         2,
				Max:         20,
				Step:        1,
				Category:    "Optimizer",
			},
			{
				Name:        "Max Generations",
				Description: "Maximum number of evolutionary generations",
				Value:       2,
				Type:        "slider",
				Min:         1,
				Max:         10,
				Step:        1,
				Category:    "Optimizer",
			},
			{
				Name:        "Mutation Rate",
				Description: "Probability of mutation in genetic algorithm",
				Value:       float64(0.1),
				Type:        "slider",
				Min:         0,
				Max:         1,
				Step:        0.01,
				Advanced:    true,
				Category:    "Optimizer",
			},
			{
				Name:        "Crossover Rate",
				Description: "Probability of crossover in genetic algorithm",
				Value:       float64(0.8),
				Type:        "slider",
				Min:         0,
				Max:         1,
				Step:        0.01,
				Advanced:    true,
				Category:    "Optimizer",
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

		// Handle preset mode
		if m.showPresets {
			switch msg.String() {
			case "up", "k":
				if m.selectedPreset > 0 {
					m.selectedPreset--
				} else {
					m.selectedPreset = len(m.presets) - 1
				}
			case "down", "j":
				if m.selectedPreset < len(m.presets)-1 {
					m.selectedPreset++
				} else {
					m.selectedPreset = 0
				}
			case "enter", " ":
				// Apply selected preset
				m.applyPreset(m.presets[m.selectedPreset])
				m.showPresets = false
			case "tab", "p":
				m.showPresets = false
			case "b", "esc":
				m.nextScreen = "back"
			case "q":
				return m, tea.Quit
			}
			return m, nil
		}

		// Handle normal configuration mode
		switch msg.String() {
		case "up", "k":
			m.navigateUp()
		case "down", "j":
			m.navigateDown()
		case "enter", " ":
			m.startEditing()
		case "p":
			// Toggle preset view
			m.showPresets = true
			m.selectedPreset = 0
		case "a":
			// Toggle advanced mode
			m.advancedMode = !m.advancedMode
			// Reset selection if current field becomes hidden
			if m.advancedMode == false && m.fields[m.selectedField].Advanced {
				m.selectedField = 0
			}
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

	// Header with mode toggle
	header := m.renderHeader()
	content = append(content, header)
	content = append(content, "")

	// Show presets if enabled
	if m.showPresets {
		presetView := m.renderPresets()
		content = append(content, presetView)
	} else {
		// Show configuration fields
		fieldsView := m.renderFields()
		content = append(content, fieldsView)
	}

	// Footer with controls
	footer := m.renderFooter()
	content = append(content, footer)

	// Join all content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Apply box styling and center
	boxed := styles.BoxStyle.Copy().
		Width(min(m.width-4, 120)).
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

// renderHeader renders the configuration header
func (m ConfigModel) renderHeader() string {
	var headerParts []string

	// Title
	title := fmt.Sprintf("⚙️ Configure %s", strings.ToUpper(m.optimizer))
	headerParts = append(headerParts, styles.TitleStyle.Render(title))

	// Mode indicators
	var modes []string
	if m.showPresets {
		modes = append(modes, styles.HighlightStyle.Render("📦 Presets"))
	} else {
		modes = append(modes, styles.InfoStyle.Render("🎛 Parameters"))
	}

	if m.advancedMode {
		modes = append(modes, styles.WarningStyle.Render("🔧 Advanced"))
	}

	modeStr := strings.Join(modes, " | ")
	headerParts = append(headerParts, modeStr)

	return lipgloss.JoinVertical(lipgloss.Center, headerParts...)
}

// renderPresets renders the preset selection view
func (m ConfigModel) renderPresets() string {
	var presetItems []string

	presetItems = append(presetItems, styles.SubheadStyle.Render("Choose a Configuration Preset"))
	presetItems = append(presetItems, "")

	for i, preset := range m.presets {
		var presetStr string

		// Preset header
		presetName := fmt.Sprintf("%s %s", preset.Icon, preset.Name)

		if i == m.selectedPreset {
			presetStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s", styles.IconSelected, presetName))
			presetStr += "\n  " + styles.BodyStyle.Render(preset.Description)

			// Show what this preset changes
			var changes []string
			for key, value := range preset.Values {
				changeStr := fmt.Sprintf("%s: %v", key, value)
				changes = append(changes, styles.CaptionStyle.Render("  • " + changeStr))
			}
			if len(changes) > 0 {
				presetStr += "\n" + strings.Join(changes, "\n")
			}
		} else {
			presetStr = styles.UnselectedStyle.Render("  " + presetName)
			presetStr += "\n  " + styles.MutedStyle.Render(preset.Description)
		}

		presetItems = append(presetItems, presetStr)
		presetItems = append(presetItems, "")
	}

	return strings.Join(presetItems, "\n")
}

// renderFields renders the configuration fields
func (m ConfigModel) renderFields() string {
	var fieldItems []string

	// Group fields by category
	categories := m.groupFieldsByCategory()

	for _, category := range []string{"Basic", "Optimizer", "Advanced"} {
		fields, exists := categories[category]
		if !exists || len(fields) == 0 {
			continue
		}

		// Skip advanced fields if not in advanced mode
		if category == "Advanced" && !m.advancedMode {
			continue
		}

		// Category header
		categoryStyle := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color(styles.DSPyBlue)).
			MarginBottom(1)
		fieldItems = append(fieldItems, categoryStyle.Render(fmt.Sprintf("── %s Settings ──", category)))
		fieldItems = append(fieldItems, "")

		// Render fields in this category
		for _, fieldIdx := range fields {
			field := m.fields[fieldIdx]

			// Skip advanced fields if not in advanced mode
			if field.Advanced && !m.advancedMode {
				continue
			}

			fieldView := m.renderField(fieldIdx)
			fieldItems = append(fieldItems, fieldView)
			fieldItems = append(fieldItems, "")
		}
	}

	return strings.Join(fieldItems, "\n")
}

// renderField renders a single configuration field
func (m ConfigModel) renderField(idx int) string {
	field := m.fields[idx]
	isSelected := idx == m.selectedField

	var lines []string

	// Field name
	var nameStr string
	if isSelected && !m.editing {
		nameStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s", styles.IconSelected, field.Name))
	} else {
		nameStr = styles.UnselectedStyle.Render(fmt.Sprintf("  %s", field.Name))
	}
	lines = append(lines, nameStr)

	// Field description
	lines = append(lines, "  " + styles.CaptionStyle.Render(field.Description))

	// Field value
	var valueStr string
	if m.editing && isSelected {
		valueStr = styles.HighlightStyle.Render(fmt.Sprintf("[%s_]", m.editValue))
	} else {
		valueStr = m.renderFieldValue(field)
	}
	lines = append(lines, "  " + valueStr)

	return strings.Join(lines, "\n")
}

// renderFieldValue renders the value of a field based on its type
func (m ConfigModel) renderFieldValue(field ConfigField) string {
	switch field.Type {
	case "bool":
		if field.Value.(bool) {
			return styles.SuccessStyle.Render("✓ Enabled")
		}
		return styles.MutedStyle.Render("✗ Disabled")

	case "select":
		return styles.InfoStyle.Render(fmt.Sprintf("› %s", field.Value))

	case "slider":
		return m.renderSlider(field)

	case "int":
		return styles.InfoStyle.Render(fmt.Sprintf("%d", field.Value))

	case "float":
		if floatVal, ok := field.Value.(float64); ok {
			return styles.InfoStyle.Render(fmt.Sprintf("%.2f", floatVal))
		}
		return styles.InfoStyle.Render(fmt.Sprintf("%v", field.Value))

	default:
		return styles.InfoStyle.Render(fmt.Sprintf("%v", field.Value))
	}
}

// renderSlider renders a visual slider for numeric fields
func (m ConfigModel) renderSlider(field ConfigField) string {
	var value float64

	switch v := field.Value.(type) {
	case int:
		value = float64(v)
	case float64:
		value = v
	default:
		return fmt.Sprintf("%v", field.Value)
	}

	// Calculate position on slider
	min := float64(field.Min)
	max := float64(field.Max)
	if max <= min {
		max = min + 1
	}

	percentage := (value - min) / (max - min)

	// Create visual slider
	sliderWidth := 20
	filledWidth := int(float64(sliderWidth) * percentage)
	if filledWidth > sliderWidth {
		filledWidth = sliderWidth
	}
	if filledWidth < 0 {
		filledWidth = 0
	}

	filled := strings.Repeat("█", filledWidth)
	empty := strings.Repeat("░", sliderWidth-filledWidth)

	sliderStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyGreen))
	emptyStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.MediumGray))

	slider := sliderStyle.Render(filled) + emptyStyle.Render(empty)

	// Format value display
	var valueDisplay string
	if field.Type == "slider" && field.Step < 1 {
		valueDisplay = fmt.Sprintf("%.2f", value)
	} else {
		valueDisplay = fmt.Sprintf("%d", int(value))
	}

	return fmt.Sprintf("%s %s", slider, styles.InfoStyle.Render(valueDisplay))
}

// renderFooter renders the footer with context-sensitive controls
func (m ConfigModel) renderFooter() string {
	var controls []string

	if m.editing {
		controls = append(controls, "[Type] Edit", "[Enter] Save", "[Esc] Cancel")
	} else if m.showPresets {
		controls = append(controls, "[↑↓] Select", "[Enter] Apply", "[Tab] Parameters", "[b] Back")
	} else {
		controls = append(controls, "[↑↓] Navigate", "[Enter] Edit", "[p] Presets")
		controls = append(controls, "[a] " + m.getAdvancedToggleText())
		controls = append(controls, "[r] Run", "[b] Back")
	}

	controls = append(controls, "[q] Quit")

	return styles.FooterStyle.Render(strings.Join(controls, " • "))
}

// Helper methods

func (m ConfigModel) getAdvancedToggleText() string {
	if m.advancedMode {
		return "Hide Advanced"
	}
	return "Show Advanced"
}

func (m ConfigModel) groupFieldsByCategory() map[string][]int {
	categories := make(map[string][]int)
	for i, field := range m.fields {
		category := field.Category
		if category == "" {
			category = "General"
		}
		categories[category] = append(categories[category], i)
	}
	return categories
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

// navigateUp moves selection to previous visible field
func (m *ConfigModel) navigateUp() {
	startIdx := m.selectedField
	for {
		if m.selectedField > 0 {
			m.selectedField--
		} else {
			m.selectedField = len(m.fields) - 1
		}

		// Check if field is visible
		field := m.fields[m.selectedField]
		if !field.Advanced || m.advancedMode {
			break
		}

		// Prevent infinite loop
		if m.selectedField == startIdx {
			break
		}
	}
}

// navigateDown moves selection to next visible field
func (m *ConfigModel) navigateDown() {
	startIdx := m.selectedField
	for {
		if m.selectedField < len(m.fields)-1 {
			m.selectedField++
		} else {
			m.selectedField = 0
		}

		// Check if field is visible
		field := m.fields[m.selectedField]
		if !field.Advanced || m.advancedMode {
			break
		}

		// Prevent infinite loop
		if m.selectedField == startIdx {
			break
		}
	}
}

// applyPreset applies a preset's values to the configuration
func (m *ConfigModel) applyPreset(preset Preset) {
	for key, value := range preset.Values {
		// Find matching field and update its value
		for i := range m.fields {
			if m.fields[i].Name == key {
				m.fields[i].Value = value
				break
			}
		}
	}
	// Update the internal config
	m.updateConfig()
}

// GetOptimizer returns the optimizer name
func (m ConfigModel) GetOptimizer() string {
	return m.optimizer
}
