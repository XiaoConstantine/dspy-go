package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// Manager handles configuration loading, validation, and management.
type Manager struct {
	config      *Config
	configPath  string
	mu          sync.RWMutex
	watchers    []ConfigWatcher
	discovery   *Discovery
	sources     []Source
	watcherDone chan struct{}
}

// ConfigWatcher is called when configuration changes.
type ConfigWatcher func(*Config) error

// NewManager creates a new configuration manager.
func NewManager(options ...ManagerOption) (*Manager, error) {
	m := &Manager{
		watchers:    make([]ConfigWatcher, 0),
		watcherDone: make(chan struct{}),
	}

	// Apply options
	for _, opt := range options {
		if err := opt(m); err != nil {
			return nil, fmt.Errorf("failed to apply manager option: %w", err)
		}
	}

	// Initialize discovery if not provided
	if m.discovery == nil {
		m.discovery = NewDiscovery()
	}

	// Initialize sources if not provided
	if len(m.sources) == 0 {
		m.sources = []Source{
			&FileSource{},
			&EnvironmentSource{},
		}
	}

	return m, nil
}

// ManagerOption represents an option for configuring the Manager.
type ManagerOption func(*Manager) error

// WithConfigPath sets the configuration file path.
func WithConfigPath(path string) ManagerOption {
	return func(m *Manager) error {
		m.configPath = path
		return nil
	}
}

// WithDiscovery sets the discovery mechanism.
func WithDiscovery(discovery *Discovery) ManagerOption {
	return func(m *Manager) error {
		m.discovery = discovery
		return nil
	}
}

// WithSources sets the configuration sources.
func WithSources(sources ...Source) ManagerOption {
	return func(m *Manager) error {
		m.sources = sources
		return nil
	}
}

// WithWatcher adds a configuration watcher.
func WithWatcher(watcher ConfigWatcher) ManagerOption {
	return func(m *Manager) error {
		m.watchers = append(m.watchers, watcher)
		return nil
	}
}

// Load loads the configuration from available sources.
func (m *Manager) Load() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Discover configuration files if no specific path is provided
	var configPaths []string
	if m.configPath != "" {
		configPaths = []string{m.configPath}
	} else {
		discoveredPaths, err := m.discovery.Discover()
		if err != nil {
			return fmt.Errorf("failed to discover configuration files: %w", err)
		}
		configPaths = discoveredPaths
	}

	// Load configuration from sources
	config := &Config{}

	// Start with default configuration
	if err := m.loadDefaults(config); err != nil {
		return fmt.Errorf("failed to load defaults: %w", err)
	}

	// Load from each source
	for _, source := range m.sources {
		if err := source.Load(config, configPaths); err != nil {
			return fmt.Errorf("failed to load from source %T: %w", source, err)
		}
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	// Store the loaded configuration
	m.config = config

	// If we have a primary config file, store its path
	if len(configPaths) > 0 {
		m.configPath = configPaths[0]
	}

	return nil
}

// loadDefaults loads default configuration values.
func (m *Manager) loadDefaults(config *Config) error {
	defaults := GetDefaultConfig()

	// Use YAML marshaling/unmarshaling to merge defaults
	data, err := yaml.Marshal(defaults)
	if err != nil {
		return fmt.Errorf("failed to marshal defaults: %w", err)
	}

	if err := yaml.Unmarshal(data, config); err != nil {
		return fmt.Errorf("failed to unmarshal defaults: %w", err)
	}

	return nil
}

// Get returns the current configuration.
func (m *Manager) Get() *Config {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config
}

// GetLLMConfig returns the LLM configuration.
func (m *Manager) GetLLMConfig() *LLMConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.LLM
}

// GetLoggingConfig returns the logging configuration.
func (m *Manager) GetLoggingConfig() *LoggingConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Logging
}

// GetExecutionConfig returns the execution configuration.
func (m *Manager) GetExecutionConfig() *ExecutionConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Execution
}

// GetModulesConfig returns the modules configuration.
func (m *Manager) GetModulesConfig() *ModulesConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Modules
}

// GetAgentsConfig returns the agents configuration.
func (m *Manager) GetAgentsConfig() *AgentsConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Agents
}

// GetToolsConfig returns the tools configuration.
func (m *Manager) GetToolsConfig() *ToolsConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Tools
}

// GetOptimizersConfig returns the optimizers configuration.
func (m *Manager) GetOptimizersConfig() *OptimizersConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config == nil {
		return nil
	}
	return &m.config.Optimizers
}


// Reload reloads the configuration from sources.
func (m *Manager) Reload() error {
	oldConfig := m.Get()

	if err := m.Load(); err != nil {
		return fmt.Errorf("failed to reload configuration: %w", err)
	}

	// Notify watchers of the change
	newConfig := m.Get()
	if err := m.notifyWatchers(newConfig); err != nil {
		// If watchers fail, rollback to old config
		m.mu.Lock()
		m.config = oldConfig
		m.mu.Unlock()
		return fmt.Errorf("failed to notify watchers, configuration rolled back: %w", err)
	}

	return nil
}

// notifyWatchers notifies all registered watchers of configuration changes.
func (m *Manager) notifyWatchers(config *Config) error {
	for i, watcher := range m.watchers {
		if err := watcher(config); err != nil {
			return fmt.Errorf("watcher %d failed: %w", i, err)
		}
	}
	return nil
}

// Watch starts watching for configuration changes.
func (m *Manager) Watch() error {
	if m.configPath == "" {
		return fmt.Errorf("no configuration file path to watch")
	}

	go m.watchFile()
	return nil
}

// watchFile watches the configuration file for changes.
func (m *Manager) watchFile() {
	var lastMod time.Time

	// Get initial modification time
	if stat, err := os.Stat(m.configPath); err == nil {
		lastMod = stat.ModTime()
	}

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if stat, err := os.Stat(m.configPath); err == nil {
				if stat.ModTime().After(lastMod) {
					lastMod = stat.ModTime()

					// Reload configuration
					if err := m.Reload(); err != nil {
						// Log error but continue watching
						fmt.Printf("Failed to reload configuration: %v\n", err)
					}
				}
			}
		case <-m.watcherDone:
			return
		}
	}
}

// StopWatching stops watching for configuration changes.
func (m *Manager) StopWatching() {
	close(m.watcherDone)
}

// Save saves the current configuration to file.
func (m *Manager) Save() error {
	m.mu.RLock()
	config := m.config
	path := m.configPath
	m.mu.RUnlock()

	if config == nil {
		return fmt.Errorf("no configuration to save")
	}

	if path == "" {
		return fmt.Errorf("no configuration file path specified")
	}

	return m.SaveToFile(path)
}

// SaveToFile saves the configuration to a specific file.
func (m *Manager) SaveToFile(path string) error {
	m.mu.RLock()
	config := m.config
	m.mu.RUnlock()

	if config == nil {
		return fmt.Errorf("no configuration to save")
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Marshal to YAML
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %w", err)
	}

	// Write to file
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write configuration file: %w", err)
	}

	return nil
}

// Update updates the configuration with new values.
func (m *Manager) Update(updater func(*Config) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.config == nil {
		return fmt.Errorf("no configuration loaded")
	}

	// Create a copy of the current config
	configCopy := *m.config

	// Apply the update
	if err := updater(&configCopy); err != nil {
		return fmt.Errorf("failed to apply update: %w", err)
	}

	// Validate the updated configuration
	if err := configCopy.Validate(); err != nil {
		return fmt.Errorf("updated configuration validation failed: %w", err)
	}

	// Update the configuration
	m.config = &configCopy

	// Notify watchers
	if err := m.notifyWatchers(m.config); err != nil {
		return fmt.Errorf("failed to notify watchers: %w", err)
	}

	return nil
}

// Reset resets the configuration to defaults.
func (m *Manager) Reset() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	defaults := GetDefaultConfig()
	if err := defaults.Validate(); err != nil {
		return fmt.Errorf("default configuration validation failed: %w", err)
	}

	m.config = defaults

	// Notify watchers
	if err := m.notifyWatchers(m.config); err != nil {
		return fmt.Errorf("failed to notify watchers: %w", err)
	}

	return nil
}

// GetConfigPath returns the current configuration file path.
func (m *Manager) GetConfigPath() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.configPath
}

// IsLoaded returns true if configuration is loaded.
func (m *Manager) IsLoaded() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config != nil
}

// Clone returns a deep copy of the current configuration.
func (m *Manager) Clone() (*Config, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.config == nil {
		return nil, fmt.Errorf("no configuration loaded")
	}

	// Use YAML marshaling for deep copy
	data, err := yaml.Marshal(m.config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal configuration: %w", err)
	}

	var configCopy Config
	if err := yaml.Unmarshal(data, &configCopy); err != nil {
		return nil, fmt.Errorf("failed to unmarshal configuration: %w", err)
	}

	return &configCopy, nil
}

// Merge merges another configuration into the current one.
func (m *Manager) Merge(other *Config) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.config == nil {
		m.config = other
		return nil
	}

	// Use YAML marshaling for merging
	// This is a simple approach - for more sophisticated merging,
	// we could implement custom merge logic

	// Marshal both configs
	currentData, err := yaml.Marshal(m.config)
	if err != nil {
		return fmt.Errorf("failed to marshal current configuration: %w", err)
	}

	otherData, err := yaml.Marshal(other)
	if err != nil {
		return fmt.Errorf("failed to marshal other configuration: %w", err)
	}

	// Unmarshal other into current (this will override fields)
	var merged Config
	if err := yaml.Unmarshal(currentData, &merged); err != nil {
		return fmt.Errorf("failed to unmarshal current configuration: %w", err)
	}

	if err := yaml.Unmarshal(otherData, &merged); err != nil {
		return fmt.Errorf("failed to unmarshal other configuration: %w", err)
	}

	// Validate merged configuration
	if err := merged.Validate(); err != nil {
		return fmt.Errorf("merged configuration validation failed: %w", err)
	}

	m.config = &merged

	// Notify watchers
	if err := m.notifyWatchers(m.config); err != nil {
		return fmt.Errorf("failed to notify watchers: %w", err)
	}

	return nil
}

// Export exports the configuration to a map for external use.
func (m *Manager) Export() (map[string]interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.config == nil {
		return nil, fmt.Errorf("no configuration loaded")
	}

	// Use YAML marshaling to convert to map
	data, err := yaml.Marshal(m.config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal configuration: %w", err)
	}

	var result map[string]interface{}
	if err := yaml.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal to map: %w", err)
	}

	return result, nil
}

// Import imports configuration from a map.
func (m *Manager) Import(data map[string]interface{}) error {
	// Convert map to YAML and back to Config
	yamlData, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal map to YAML: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(yamlData, &config); err != nil {
		return fmt.Errorf("failed to unmarshal YAML to config: %w", err)
	}

	// Validate imported configuration
	if err := config.Validate(); err != nil {
		return fmt.Errorf("imported configuration validation failed: %w", err)
	}

	m.mu.Lock()
	m.config = &config
	m.mu.Unlock()

	// Notify watchers
	if err := m.notifyWatchers(m.config); err != nil {
		return fmt.Errorf("failed to notify watchers: %w", err)
	}

	return nil
}

// Global configuration manager instance.
var (
	globalManager     *Manager
	globalManagerOnce sync.Once
	globalManagerMu   sync.RWMutex
)

// GetGlobalManager returns the global configuration manager.
func GetGlobalManager() *Manager {
	globalManagerMu.RLock()
	if globalManager != nil {
		defer globalManagerMu.RUnlock()
		return globalManager
	}
	globalManagerMu.RUnlock()

	globalManagerOnce.Do(func() {
		globalManagerMu.Lock()
		defer globalManagerMu.Unlock()

		manager, err := NewManager()
		if err != nil {
			// If we can't create the manager, create a basic one
			manager = &Manager{
				watchers:    make([]ConfigWatcher, 0),
				watcherDone: make(chan struct{}),
				discovery:   NewDiscovery(),
				sources: []Source{
					&FileSource{},
					&EnvironmentSource{},
				},
			}
		}

		globalManager = manager
	})

	return globalManager
}

// SetGlobalManager sets the global configuration manager.
func SetGlobalManager(manager *Manager) {
	globalManagerMu.Lock()
	defer globalManagerMu.Unlock()
	globalManager = manager
}

// LoadGlobalConfig loads the global configuration.
func LoadGlobalConfig() error {
	return GetGlobalManager().Load()
}

// GetGlobalConfig returns the global configuration.
func GetGlobalConfig() *Config {
	return GetGlobalManager().Get()
}

// ReloadGlobalConfig reloads the global configuration.
func ReloadGlobalConfig() error {
	return GetGlobalManager().Reload()
}
