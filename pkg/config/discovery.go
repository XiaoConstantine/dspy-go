package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Discovery handles configuration file discovery.
type Discovery struct {
	searchPaths []string
	filenames   []string
}

// NewDiscovery creates a new configuration discovery instance.
func NewDiscovery() *Discovery {
	return &Discovery{
		searchPaths: getDefaultSearchPaths(),
		filenames:   getDefaultFilenames(),
	}
}

// NewDiscoveryWithPaths creates a discovery instance with custom search paths.
func NewDiscoveryWithPaths(searchPaths []string) *Discovery {
	return &Discovery{
		searchPaths: searchPaths,
		filenames:   getDefaultFilenames(),
	}
}

// NewDiscoveryWithFilenames creates a discovery instance with custom filenames.
func NewDiscoveryWithFilenames(filenames []string) *Discovery {
	return &Discovery{
		searchPaths: getDefaultSearchPaths(),
		filenames:   filenames,
	}
}

// NewDiscoveryWithOptions creates a discovery instance with custom options.
func NewDiscoveryWithOptions(searchPaths, filenames []string) *Discovery {
	return &Discovery{
		searchPaths: searchPaths,
		filenames:   filenames,
	}
}

// getDefaultSearchPaths returns the default search paths for configuration files.
func getDefaultSearchPaths() []string {
	paths := []string{
		".", // Current directory
	}

	// Add user home directory
	if homeDir, err := os.UserHomeDir(); err == nil {
		paths = append(paths, homeDir)
		paths = append(paths, filepath.Join(homeDir, ".config", "dspy-go"))
		paths = append(paths, filepath.Join(homeDir, ".dspy-go"))
	}

	// Add system-wide configuration directories
	paths = append(paths, "/etc/dspy-go")
	paths = append(paths, "/usr/local/etc/dspy-go")

	// Add XDG config directories
	if xdgConfigHome := os.Getenv("XDG_CONFIG_HOME"); xdgConfigHome != "" {
		paths = append(paths, filepath.Join(xdgConfigHome, "dspy-go"))
	}

	if xdgConfigDirs := os.Getenv("XDG_CONFIG_DIRS"); xdgConfigDirs != "" {
		for _, dir := range strings.Split(xdgConfigDirs, ":") {
			if dir != "" {
				paths = append(paths, filepath.Join(dir, "dspy-go"))
			}
		}
	}

	// Add application-specific paths
	if appDir := os.Getenv("DSPY_CONFIG_DIR"); appDir != "" {
		paths = append(paths, appDir)
	}

	// Add current working directory subdirectories
	if cwd, err := os.Getwd(); err == nil {
		paths = append(paths, filepath.Join(cwd, "config"))
		paths = append(paths, filepath.Join(cwd, "configs"))
		paths = append(paths, filepath.Join(cwd, ".config"))
	}

	return paths
}

// getDefaultFilenames returns the default configuration filenames to search for.
func getDefaultFilenames() []string {
	return []string{
		"dspy.yaml",
		"dspy.yml",
		"dspy-go.yaml",
		"dspy-go.yml",
		"config.yaml",
		"config.yml",
		".dspy.yaml",
		".dspy.yml",
		".dspy-go.yaml",
		".dspy-go.yml",
	}
}

// Discover searches for configuration files in the configured paths.
func (d *Discovery) Discover() ([]string, error) {
	var foundFiles []string

	for _, searchPath := range d.searchPaths {
		for _, filename := range d.filenames {
			fullPath := filepath.Join(searchPath, filename)

			if fileExists(fullPath) {
				absPath, err := filepath.Abs(fullPath)
				if err != nil {
					return nil, fmt.Errorf("failed to get absolute path for %s: %w", fullPath, err)
				}
				foundFiles = append(foundFiles, absPath)
			}
		}
	}

	// Remove duplicates while preserving order
	foundFiles = removeDuplicates(foundFiles)

	return foundFiles, nil
}

// DiscoverFirst returns the first configuration file found.
func (d *Discovery) DiscoverFirst() (string, error) {
	files, err := d.Discover()
	if err != nil {
		return "", err
	}

	if len(files) == 0 {
		return "", fmt.Errorf("no configuration files found")
	}

	return files[0], nil
}

// DiscoverInPath searches for configuration files in a specific path.
func (d *Discovery) DiscoverInPath(path string) ([]string, error) {
	var foundFiles []string

	for _, filename := range d.filenames {
		fullPath := filepath.Join(path, filename)

		if fileExists(fullPath) {
			absPath, err := filepath.Abs(fullPath)
			if err != nil {
				return nil, fmt.Errorf("failed to get absolute path for %s: %w", fullPath, err)
			}
			foundFiles = append(foundFiles, absPath)
		}
	}

	return foundFiles, nil
}

// DiscoverWithPattern searches for configuration files matching a pattern.
func (d *Discovery) DiscoverWithPattern(pattern string) ([]string, error) {
	var foundFiles []string

	for _, searchPath := range d.searchPaths {
		matches, err := filepath.Glob(filepath.Join(searchPath, pattern))
		if err != nil {
			return nil, fmt.Errorf("failed to match pattern %s in %s: %w", pattern, searchPath, err)
		}

		for _, match := range matches {
			if fileExists(match) {
				absPath, err := filepath.Abs(match)
				if err != nil {
					return nil, fmt.Errorf("failed to get absolute path for %s: %w", match, err)
				}
				foundFiles = append(foundFiles, absPath)
			}
		}
	}

	// Remove duplicates while preserving order
	foundFiles = removeDuplicates(foundFiles)

	return foundFiles, nil
}

// DiscoverRecursive searches for configuration files recursively in the search paths.
func (d *Discovery) DiscoverRecursive() ([]string, error) {
	var foundFiles []string

	for _, searchPath := range d.searchPaths {
		err := filepath.Walk(searchPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				// Skip directories that can't be accessed
				return nil
			}

			if info.IsDir() {
				return nil
			}

			filename := filepath.Base(path)
			for _, targetFilename := range d.filenames {
				if filename == targetFilename {
					absPath, err := filepath.Abs(path)
					if err != nil {
						return fmt.Errorf("failed to get absolute path for %s: %w", path, err)
					}
					foundFiles = append(foundFiles, absPath)
					break
				}
			}

			return nil
		})

		if err != nil {
			return nil, fmt.Errorf("failed to walk directory %s: %w", searchPath, err)
		}
	}

	// Remove duplicates while preserving order
	foundFiles = removeDuplicates(foundFiles)

	return foundFiles, nil
}

// AddSearchPath adds a search path to the discovery.
func (d *Discovery) AddSearchPath(path string) {
	d.searchPaths = append(d.searchPaths, path)
}

// AddSearchPaths adds multiple search paths to the discovery.
func (d *Discovery) AddSearchPaths(paths []string) {
	d.searchPaths = append(d.searchPaths, paths...)
}

// SetSearchPaths sets the search paths for discovery.
func (d *Discovery) SetSearchPaths(paths []string) {
	d.searchPaths = paths
}

// GetSearchPaths returns the current search paths.
func (d *Discovery) GetSearchPaths() []string {
	return d.searchPaths
}

// AddFilename adds a filename to search for.
func (d *Discovery) AddFilename(filename string) {
	d.filenames = append(d.filenames, filename)
}

// AddFilenames adds multiple filenames to search for.
func (d *Discovery) AddFilenames(filenames []string) {
	d.filenames = append(d.filenames, filenames...)
}

// SetFilenames sets the filenames to search for.
func (d *Discovery) SetFilenames(filenames []string) {
	d.filenames = filenames
}

// GetFilenames returns the current filenames being searched for.
func (d *Discovery) GetFilenames() []string {
	return d.filenames
}

// CreateDefaultConfigFile creates a default configuration file in the first search path.
func (d *Discovery) CreateDefaultConfigFile() (string, error) {
	if len(d.searchPaths) == 0 {
		return "", fmt.Errorf("no search paths configured")
	}

	if len(d.filenames) == 0 {
		return "", fmt.Errorf("no filenames configured")
	}

	// Use the first search path and first filename
	configDir := d.searchPaths[0]
	filename := d.filenames[0]

	// Create directory if it doesn't exist
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create config directory %s: %w", configDir, err)
	}

	configPath := filepath.Join(configDir, filename)

	// Check if file already exists
	if fileExists(configPath) {
		return "", fmt.Errorf("configuration file already exists at %s", configPath)
	}

	// Get default configuration
	defaultConfig := GetDefaultConfig()

	// Create a temporary manager to save the config
	manager := &Manager{config: defaultConfig}
	if err := manager.SaveToFile(configPath); err != nil {
		return "", fmt.Errorf("failed to save default config to %s: %w", configPath, err)
	}

	return configPath, nil
}

// CreateDefaultConfigFileInPath creates a default configuration file in a specific path.
func (d *Discovery) CreateDefaultConfigFileInPath(path string) (string, error) {
	if len(d.filenames) == 0 {
		return "", fmt.Errorf("no filenames configured")
	}

	filename := d.filenames[0]

	// Create directory if it doesn't exist
	if err := os.MkdirAll(path, 0755); err != nil {
		return "", fmt.Errorf("failed to create config directory %s: %w", path, err)
	}

	configPath := filepath.Join(path, filename)

	// Check if file already exists
	if fileExists(configPath) {
		return "", fmt.Errorf("configuration file already exists at %s", configPath)
	}

	// Get default configuration
	defaultConfig := GetDefaultConfig()

	// Create a temporary manager to save the config
	manager := &Manager{config: defaultConfig}
	if err := manager.SaveToFile(configPath); err != nil {
		return "", fmt.Errorf("failed to save default config to %s: %w", configPath, err)
	}

	return configPath, nil
}

// Validate validates the discovery configuration.
func (d *Discovery) Validate() error {
	if len(d.searchPaths) == 0 {
		return fmt.Errorf("no search paths configured")
	}

	if len(d.filenames) == 0 {
		return fmt.Errorf("no filenames configured")
	}

	// Validate that at least one search path exists
	foundPath := false
	for _, path := range d.searchPaths {
		if dirExists(path) {
			foundPath = true
			break
		}
	}

	if !foundPath {
		return fmt.Errorf("none of the configured search paths exist")
	}

	return nil
}

// GetEnvironmentOverrides returns configuration overrides from environment variables.
func (d *Discovery) GetEnvironmentOverrides() map[string]string {
	overrides := make(map[string]string)

	// Common environment variable patterns
	envPrefixes := []string{
		"DSPY_",
		"DSPY_GO_",
	}

	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key, value := parts[0], parts[1]

		for _, prefix := range envPrefixes {
			if strings.HasPrefix(key, prefix) {
				// Convert environment variable to config key
				configKey := strings.ToLower(strings.TrimPrefix(key, prefix))
				configKey = strings.ReplaceAll(configKey, "_", ".")
				overrides[configKey] = value
				break
			}
		}
	}

	return overrides
}

// Helper functions

// fileExists checks if a file exists and is not a directory.
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

// dirExists checks if a directory exists.
func dirExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// removeDuplicates removes duplicate strings while preserving order.
func removeDuplicates(strings []string) []string {
	seen := make(map[string]bool)
	result := []string{}

	for _, str := range strings {
		if !seen[str] {
			seen[str] = true
			result = append(result, str)
		}
	}

	return result
}

// Convenience functions for common discovery patterns

// DiscoverConfigFiles discovers configuration files using default settings.
func DiscoverConfigFiles() ([]string, error) {
	discovery := NewDiscovery()
	return discovery.Discover()
}

// DiscoverFirstConfigFile discovers the first configuration file found.
func DiscoverFirstConfigFile() (string, error) {
	discovery := NewDiscovery()
	return discovery.DiscoverFirst()
}

// DiscoverConfigFilesInPath discovers configuration files in a specific path.
func DiscoverConfigFilesInPath(path string) ([]string, error) {
	discovery := NewDiscovery()
	return discovery.DiscoverInPath(path)
}

// CreateDefaultConfigFileInCurrentDir creates a default configuration file in the current directory.
func CreateDefaultConfigFileInCurrentDir() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current directory: %w", err)
	}

	discovery := NewDiscovery()
	return discovery.CreateDefaultConfigFileInPath(cwd)
}

// CreateDefaultConfigFileInHomeDir creates a default configuration file in the user's home directory.
func CreateDefaultConfigFileInHomeDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}

	configDir := filepath.Join(homeDir, ".config", "dspy-go")
	discovery := NewDiscovery()
	return discovery.CreateDefaultConfigFileInPath(configDir)
}
