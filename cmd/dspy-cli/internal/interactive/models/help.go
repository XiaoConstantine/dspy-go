package models

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
)

// HelpModel represents the help system state
type HelpModel struct {
	width        int
	height       int
	viewport     viewport.Model
	currentTopic string
	topics       []HelpTopic
	selectedTopic int
	nextScreen   string
}

// HelpTopic represents a help topic
type HelpTopic struct {
	ID          string
	Title       string
	Icon        string
	Content     string
	KeyBindings []KeyBinding
}

// KeyBinding represents a keyboard shortcut
type KeyBinding struct {
	Key         string
	Description string
}

// NewHelpModel creates a new help model
func NewHelpModel() HelpModel {
	vp := viewport.New(80, 20)
	vp.Style = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue))

	topics := getHelpTopics()

	model := HelpModel{
		viewport:     vp,
		topics:       topics,
		currentTopic: "overview",
		width:        80,
		height:       24,
	}

	model.updateContent()
	return model
}

// getHelpTopics returns all available help topics
func getHelpTopics() []HelpTopic {
	return []HelpTopic{
		{
			ID:    "overview",
			Title: "Getting Started",
			Icon:  "🚀",
			Content: `Welcome to DSPy-CLI Interactive Mode!

DSPy-CLI transforms prompt optimization from a manual, trial-and-error process into a guided, scientific approach. Think of it as your AI optimization laboratory.

## What is DSPy?
DSPy is a framework that automatically optimizes prompts for language models. Instead of manually tweaking prompts, DSPy uses algorithms to systematically improve them.

## How This Works
1. 📋 Choose your task type (math, reasoning, QA, etc.)
2. 🧠 Get AI-powered optimizer recommendations
3. ⚙️ Configure parameters (or use smart presets)
4. 🔄 Watch live optimization progress
5. 📊 View detailed results and export reports

## Why Use DSPy-CLI?
- ✅ **Scientific Approach**: Systematic optimization vs guesswork
- ⚡ **Fast Results**: Optimized prompts in minutes, not hours
- 📈 **Measurable Gains**: Track exact improvement percentages
- 🎯 **Task-Specific**: Optimizers tuned for different problem types
- 💰 **Cost Efficient**: Better results with fewer API calls`,
			KeyBindings: []KeyBinding{
				{"Enter", "Select your task type"},
				{"Tab", "Switch between views"},
				{"?", "Show this help"},
				{"q", "Quit application"},
			},
		},
		{
			ID:    "optimizers",
			Title: "Optimizer Guide",
			Icon:  "🧠",
			Content: `Choose the Right Optimizer for Your Task

## 🚀 Bootstrap (Recommended for Beginners)
**Best for**: Quick improvements, getting started
**Speed**: ⚡ Very Fast | **Complexity**: 🟢 Simple
Bootstrap selects the best examples for few-shot learning. Like having an expert curator pick the perfect examples for your prompt.

## 🧠 MIPRO (Most Popular)
**Best for**: Balanced optimization, systematic improvement
**Speed**: ⚖️ Moderate | **Complexity**: 🟡 Medium
Multi-step Interactive Prompt Optimization. The Swiss Army knife of optimizers - good for almost everything.

## ⚡ SIMBA (Advanced)
**Best for**: Complex reasoning, introspective tasks
**Speed**: 🐌 Slower | **Complexity**: 🔴 High
Stochastic Introspective Mini-Batch Ascent. Uses self-reflection to improve reasoning chains.

## 🤝 COPRO (Collaborative)
**Best for**: Multi-module systems, complex pipelines
**Speed**: ⚖️ Moderate | **Complexity**: 🟡 Medium-High
Collaborative Prompt Optimization. Perfect when multiple AI components need to work together.

## 🔬 GEPA (Research-Grade)
**Best for**: Cutting-edge optimization, research projects
**Speed**: 🐌 Slowest | **Complexity**: 🔴 Very High
Generative Evolutionary Prompt Adaptation. State-of-the-art optimization using genetic algorithms.`,
			KeyBindings: []KeyBinding{
				{"Enter", "Select optimizer"},
				{"d", "View detailed optimizer info"},
				{"c", "Compare optimizers"},
				{"?", "Show help"},
			},
		},
		{
			ID:    "configuration",
			Title: "Configuration Guide",
			Icon:  "⚙️",
			Content: `Master the Configuration Screen

## 🎚️ Using Sliders
Visual sliders show current values and ranges:
████████░░░░░░░░░░░░ 8/20

- **Left/Right arrows**: Adjust values
- **Enter**: Edit exact number
- **Space**: Quick toggle for booleans

## 📦 Presets (Recommended)
Smart presets for common scenarios:
- ⚡ **Quick Test**: Fast results, minimal examples
- ⚖️ **Balanced**: Good speed/accuracy balance
- 🎯 **High Quality**: Maximum accuracy, slower
- 🔬 **Research**: Custom presets per optimizer

## 🔧 Advanced Mode
Toggle advanced parameters:
- **Temperature**: Model creativity (0.0-1.0)
- **Learning Rate**: Optimization speed
- **Batch Size**: Processing efficiency
- **API Timeout**: Request timeouts

## 💡 Pro Tips
- Start with presets, then fine-tune
- Higher examples = better results but slower
- Use Quick Test for experimentation
- Enable caching for repeated runs`,
			KeyBindings: []KeyBinding{
				{"↑↓", "Navigate fields"},
				{"Enter", "Edit value"},
				{"p", "Show presets"},
				{"a", "Toggle advanced mode"},
				{"r", "Run optimization"},
			},
		},
		{
			ID:    "live-optimization",
			Title: "Live Optimization",
			Icon:  "🔄",
			Content: `Understanding Live Optimization

## 📊 Progress Phases
Watch your optimization unfold in real-time:

1. **🔄 Initializing**: Setting up configuration
2. **📥 Loading Dataset**: Preparing examples
3. **🧠 Optimizing**: Running the optimization algorithm
4. **📊 Evaluating**: Testing final results
5. **✅ Complete**: Ready to view results

## 📈 Metrics Dashboard
Track key performance indicators:
- **🏆 Best Score**: Highest accuracy achieved
- **📊 Current**: Latest trial results
- **📈 Improvement**: Percentage gain over baseline
- **⏱ Time**: Elapsed and estimated remaining

## 📋 Log Viewer
Detailed progress messages:
- Scroll with **↑↓** arrows
- **Page Up/Down** for quick navigation
- Real-time updates as optimization runs

## 🛑 Controls
- **Ctrl+C**: Cancel optimization (graceful)
- **Enter**: View results when complete
- **q**: Force quit`,
			KeyBindings: []KeyBinding{
				{"↑↓", "Scroll logs"},
				{"Ctrl+C", "Cancel optimization"},
				{"Enter", "View results (when complete)"},
				{"q", "Quit"},
			},
		},
		{
			ID:    "results",
			Title: "Results & Export",
			Icon:  "📊",
			Content: `Understanding Your Results

## 🎯 Success Metrics
Your optimization results include:
- **Initial Accuracy**: Baseline performance
- **Final Accuracy**: Optimized performance
- **Improvement**: Percentage gain achieved
- **Confidence Interval**: Statistical reliability

## 📈 Visual Charts
Beautiful progress visualization:
- **Accuracy Bars**: Before/after comparison
- **Progress Indicators**: Visual improvement
- **Statistical Analysis**: Confidence intervals

## 💾 Export Options
Share and save your results:

### 📋 JSON Export
{
  "optimizer": "mipro",
  "improvement": "45.2%",
  "final_accuracy": "87.3%"
}

### 📊 CSV Export
Perfect for spreadsheets and analysis tools

### 📝 Markdown Report
Professional reports with:
- Executive summary
- Statistical analysis
- Methodology notes

### 📎 Clipboard Copy
Quick sharing for Slack, emails, etc.

## 🔍 Statistical Analysis
- **95% Confidence Interval**: Reliability measure
- **Significance Testing**: Is improvement real?
- **Sample Size**: Number of examples used`,
			KeyBindings: []KeyBinding{
				{"Tab", "Switch view (Summary/Details/Export)"},
				{"s", "Summary view"},
				{"d", "Detailed stats"},
				{"e", "Export options"},
				{"r", "Run again"},
			},
		},
		{
			ID:    "keyboard",
			Title: "Keyboard Shortcuts",
			Icon:  "⌨️",
			Content: `Master the Keyboard Shortcuts

## Global Shortcuts (Work Everywhere)
- q: Quit application
- Ctrl+C: Force quit
- ?: Show help
- Tab: Switch between sections

## Welcome Screen
- ↑↓ or j/k: Navigate options
- Enter: Select task type
- c: Compare optimizers
- ?: Show help

## Configuration Screen
- ↑↓ or j/k: Navigate fields
- Enter: Edit field value
- p: Show presets
- a: Toggle advanced mode
- r: Run optimization
- b: Back to previous screen

## Live Optimization
- ↑↓: Scroll log viewer
- Page Up/Down: Quick scroll
- Ctrl+C: Cancel optimization
- Enter: View results (when done)

## Results Screen
- Tab: Switch view mode
- s: Summary view
- d: Detailed statistics
- e: Export options
- r: Run again
- c: Compare results

## Pro Tips
- Vim users: j/k work for navigation
- Hold Shift: For faster scrolling
- Escape: Universal "go back" key`,
			KeyBindings: []KeyBinding{
				{"Any key", "Practice the shortcuts!"},
			},
		},
		{
			ID:    "troubleshooting",
			Title: "Troubleshooting",
			Icon:  "🔧",
			Content: `Common Issues and Solutions

## API Key Issues
Problem: "API key not found" error
Solutions:
- Set environment variable: export OPENAI_API_KEY=your_key
- Or use config file: ~/.dspy/config.yaml
- Verify key has sufficient credits

## Slow Performance
Problem: Optimization takes too long
Solutions:
- Use "Quick Test" preset (3 examples)
- Choose Bootstrap optimizer (fastest)
- Reduce max examples in configuration
- Enable result caching

## Poor Results
Problem: No improvement or low accuracy
Solutions:
- Try different optimizer (MIPRO is reliable)
- Increase max examples (more data = better results)
- Use "High Quality" preset
- Check if your task matches the optimizer

## Terminal Issues
Problem: Display looks broken
Solutions:
- Ensure terminal supports colors
- Resize terminal window (minimum 80x24)
- Update terminal application
- Try different terminal emulator

## High API Costs
Problem: Using too many API calls
Solutions:
- Enable result caching
- Use fewer examples for testing
- Start with Bootstrap optimizer
- Use "Quick Test" preset first

## Interface Issues
Problem: Text is cut off or overlapping
Solutions:
- Make terminal window larger
- Use full-screen mode
- Minimum recommended: 100x30 characters

## Getting Help
- Press ? anywhere for context help
- Check logs in ~/.dspy/logs/
- Report issues: github.com/XiaoConstantine/dspy-go`,
			KeyBindings: []KeyBinding{
				{"?", "Context-sensitive help"},
				{"q", "Back to main interface"},
			},
		},
	}
}

// Init initializes the model
func (m HelpModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m HelpModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Resize viewport
		m.viewport.Width = m.width - 4
		m.viewport.Height = m.height - 12 // Leave room for header and nav

		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.selectedTopic > 0 {
				m.selectedTopic--
				m.currentTopic = m.topics[m.selectedTopic].ID
				m.updateContent()
			}

		case "down", "j":
			if m.selectedTopic < len(m.topics)-1 {
				m.selectedTopic++
				m.currentTopic = m.topics[m.selectedTopic].ID
				m.updateContent()
			}

		case "1", "2", "3", "4", "5", "6", "7":
			// Quick topic selection
			topicNum := int(msg.String()[0] - '1')
			if topicNum < len(m.topics) {
				m.selectedTopic = topicNum
				m.currentTopic = m.topics[m.selectedTopic].ID
				m.updateContent()
			}

		case "tab":
			// Cycle through topics
			m.selectedTopic = (m.selectedTopic + 1) % len(m.topics)
			m.currentTopic = m.topics[m.selectedTopic].ID
			m.updateContent()

		case "b", "esc":
			m.nextScreen = "back"

		case "q":
			return m, tea.Quit
		}
	}

	// Update viewport
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)

	return m, cmd
}

// View renders the help screen
func (m HelpModel) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading help..."
	}

	var sections []string

	// Header
	header := m.renderHeader()
	sections = append(sections, header)

	// Topic navigation
	navigation := m.renderNavigation()
	sections = append(sections, navigation)

	// Main content
	content := m.viewport.View()
	sections = append(sections, content)

	// Footer
	footer := m.renderFooter()
	sections = append(sections, footer)

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// renderHeader renders the help system header
func (m HelpModel) renderHeader() string {
	title := "📚 DSPy-CLI Help Center"

	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Background(lipgloss.Color(styles.DSPyBlue)).
		Foreground(lipgloss.Color(styles.White)).
		Width(m.width).
		Padding(0, 2)

	return headerStyle.Render(title)
}

// renderNavigation renders the topic navigation
func (m HelpModel) renderNavigation() string {
	var topics []string

	for i, topic := range m.topics {
		topicStyle := lipgloss.NewStyle().Padding(0, 1)

		if i == m.selectedTopic {
			topicStyle = topicStyle.
				Bold(true).
				Foreground(lipgloss.Color(styles.DSPyGreen)).
				Background(lipgloss.Color(styles.LightGray))
			topics = append(topics, topicStyle.Render(fmt.Sprintf("%s %d. %s",
				topic.Icon, i+1, topic.Title)))
		} else {
			topicStyle = topicStyle.Foreground(lipgloss.Color(styles.MediumGray))
			topics = append(topics, topicStyle.Render(fmt.Sprintf("%s %d. %s",
				topic.Icon, i+1, topic.Title)))
		}
	}

	navStyle := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder(), false, false, true, false).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1, 0)

	return navStyle.Render(lipgloss.JoinHorizontal(lipgloss.Top, topics...))
}

// renderFooter renders the help footer
func (m HelpModel) renderFooter() string {
	currentTopic := m.topics[m.selectedTopic]

	var keyBindings []string
	for _, kb := range currentTopic.KeyBindings {
		keyBindings = append(keyBindings, fmt.Sprintf("[%s] %s", kb.Key, kb.Description))
	}

	// Add global shortcuts
	globalShortcuts := []string{
		"[↑↓/1-7] Navigate",
		"[Tab] Next Topic",
		"[b] Back",
		"[q] Quit",
	}

	allShortcuts := append(keyBindings, globalShortcuts...)

	return styles.FooterStyle.Render(strings.Join(allShortcuts, " • "))
}

// updateContent updates the viewport content based on current topic
func (m *HelpModel) updateContent() {
	for _, topic := range m.topics {
		if topic.ID == m.currentTopic {
			m.viewport.SetContent(topic.Content)
			break
		}
	}
}

// GetNextScreen returns the next screen to navigate to
func (m HelpModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation resets the navigation state
func (m *HelpModel) ResetNavigation() {
	m.nextScreen = ""
}
