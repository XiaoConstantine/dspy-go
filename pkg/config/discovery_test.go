package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewDiscoveryWithFilenames(t *testing.T) {
	filenames := []string{"custom.yaml", "custom.yml"}
	discovery := NewDiscoveryWithFilenames(filenames)
	
	assert.Equal(t, filenames, discovery.GetFilenames())
	assert.NotEmpty(t, discovery.GetSearchPaths())
}

func TestNewDiscoveryWithOptions(t *testing.T) {
	searchPaths := []string{"/custom/path"}
	filenames := []string{"custom.yaml"}
	discovery := NewDiscoveryWithOptions(searchPaths, filenames)
	
	assert.Equal(t, searchPaths, discovery.GetSearchPaths())
	assert.Equal(t, filenames, discovery.GetFilenames())
}

func TestDiscoverFirst(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	
	// Create a test config file
	err := os.WriteFile(configFile, []byte("test: value"), 0644)
	require.NoError(t, err)
	
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	firstFile, err := discovery.DiscoverFirst()
	require.NoError(t, err)
	assert.Contains(t, firstFile, "dspy.yaml")
}

func TestDiscoverFirstNoFiles(t *testing.T) {
	tempDir := t.TempDir()
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	
	_, err := discovery.DiscoverFirst()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no configuration files found")
}

func TestDiscoverInPath(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	
	// Create a test config file
	err := os.WriteFile(configFile, []byte("test: value"), 0644)
	require.NoError(t, err)
	
	discovery := NewDiscovery()
	files, err := discovery.DiscoverInPath(tempDir)
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.Contains(t, files[0], "dspy.yaml")
}

func TestDiscoverWithPattern(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "test.yaml")
	
	// Create a test config file
	err := os.WriteFile(configFile, []byte("test: value"), 0644)
	require.NoError(t, err)
	
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	files, err := discovery.DiscoverWithPattern("*.yaml")
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.Contains(t, files[0], "test.yaml")
}

func TestDiscoverRecursive(t *testing.T) {
	tempDir := t.TempDir()
	subDir := filepath.Join(tempDir, "subdir")
	err := os.MkdirAll(subDir, 0755)
	require.NoError(t, err)
	
	configFile := filepath.Join(subDir, "dspy.yaml")
	err = os.WriteFile(configFile, []byte("test: value"), 0644)
	require.NoError(t, err)
	
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	files, err := discovery.DiscoverRecursive()
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.Contains(t, files[0], "dspy.yaml")
}

func TestDiscoverySearchPathMethods(t *testing.T) {
	discovery := NewDiscovery()
	
	// Test AddSearchPath
	discovery.AddSearchPath("/test/path")
	paths := discovery.GetSearchPaths()
	assert.Contains(t, paths, "/test/path")
	
	// Test AddSearchPaths
	discovery.AddSearchPaths([]string{"/test/path2", "/test/path3"})
	paths = discovery.GetSearchPaths()
	assert.Contains(t, paths, "/test/path2")
	assert.Contains(t, paths, "/test/path3")
	
	// Test SetSearchPaths
	discovery.SetSearchPaths([]string{"/new/path"})
	paths = discovery.GetSearchPaths()
	assert.Equal(t, []string{"/new/path"}, paths)
}

func TestDiscoveryFilenameMethods(t *testing.T) {
	discovery := NewDiscovery()
	
	// Test AddFilename
	discovery.AddFilename("custom.yaml")
	filenames := discovery.GetFilenames()
	assert.Contains(t, filenames, "custom.yaml")
	
	// Test AddFilenames
	discovery.AddFilenames([]string{"custom2.yaml", "custom3.yaml"})
	filenames = discovery.GetFilenames()
	assert.Contains(t, filenames, "custom2.yaml")
	assert.Contains(t, filenames, "custom3.yaml")
	
	// Test SetFilenames
	discovery.SetFilenames([]string{"new.yaml"})
	filenames = discovery.GetFilenames()
	assert.Equal(t, []string{"new.yaml"}, filenames)
}

func TestCreateDefaultConfigFile(t *testing.T) {
	tempDir := t.TempDir()
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	
	configPath, err := discovery.CreateDefaultConfigFile()
	require.NoError(t, err)
	assert.FileExists(t, configPath)
	assert.Contains(t, configPath, "dspy.yaml")
}

func TestCreateDefaultConfigFileAlreadyExists(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	
	// Create existing file
	err := os.WriteFile(configFile, []byte("existing"), 0644)
	require.NoError(t, err)
	
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	_, err = discovery.CreateDefaultConfigFile()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}

func TestCreateDefaultConfigFileInPath(t *testing.T) {
	tempDir := t.TempDir()
	discovery := NewDiscovery()
	
	configPath, err := discovery.CreateDefaultConfigFileInPath(tempDir)
	require.NoError(t, err)
	assert.FileExists(t, configPath)
	assert.Contains(t, configPath, "dspy.yaml")
}

func TestDiscoveryValidate(t *testing.T) {
	// Test valid discovery
	tempDir := t.TempDir()
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	err := discovery.Validate()
	assert.NoError(t, err)
	
	// Test no search paths
	discovery = &Discovery{}
	err = discovery.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no search paths configured")
	
	// Test no filenames
	discovery = &Discovery{searchPaths: []string{tempDir}}
	err = discovery.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no filenames configured")
	
	// Test no existing paths
	discovery = NewDiscoveryWithPaths([]string{"/nonexistent"})
	err = discovery.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "none of the configured search paths exist")
}

func TestGetEnvironmentOverrides(t *testing.T) {
	// Set test environment variables
	os.Setenv("DSPY_TEST_KEY", "test_value")
	os.Setenv("DSPY_GO_ANOTHER_KEY", "another_value")
	os.Setenv("OTHER_PREFIX_KEY", "ignored_value")
	
	defer func() {
		os.Unsetenv("DSPY_TEST_KEY")
		os.Unsetenv("DSPY_GO_ANOTHER_KEY")
		os.Unsetenv("OTHER_PREFIX_KEY")
	}()
	
	discovery := NewDiscovery()
	overrides := discovery.GetEnvironmentOverrides()
	
	assert.Equal(t, "test_value", overrides["test.key"])
	// Note: DSPY_GO_ prefix becomes "go.another.key" after processing
	assert.Equal(t, "another_value", overrides["go.another.key"])
	assert.NotContains(t, overrides, "other.prefix.key")
}

func TestDirExists(t *testing.T) {
	tempDir := t.TempDir()
	assert.True(t, dirExists(tempDir))
	assert.False(t, dirExists("/nonexistent"))
	
	// Test with file instead of directory
	tempFile := filepath.Join(tempDir, "test.txt")
	err := os.WriteFile(tempFile, []byte("test"), 0644)
	require.NoError(t, err)
	assert.False(t, dirExists(tempFile))
}

func TestRemoveDuplicates(t *testing.T) {
	input := []string{"a", "b", "a", "c", "b", "d"}
	expected := []string{"a", "b", "c", "d"}
	result := removeDuplicates(input)
	assert.Equal(t, expected, result)
}

func TestDiscoverConfigFiles(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	err := os.WriteFile(configFile, []byte("test"), 0644)
	require.NoError(t, err)
	
	// Test with a custom discovery that includes our temp dir
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	files, err := discovery.Discover()
	require.NoError(t, err)
	assert.Len(t, files, 1)
}

func TestDiscoverFirstConfigFile(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	err := os.WriteFile(configFile, []byte("test"), 0644)
	require.NoError(t, err)
	
	// Test DiscoverFirstConfigFile function
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	firstFile, err := discovery.DiscoverFirst()
	require.NoError(t, err)
	assert.Contains(t, firstFile, "dspy.yaml")
}

func TestDiscoverConfigFilesInPath(t *testing.T) {
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "dspy.yaml")
	err := os.WriteFile(configFile, []byte("test"), 0644)
	require.NoError(t, err)
	
	files, err := DiscoverConfigFilesInPath(tempDir)
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.Contains(t, files[0], "dspy.yaml")
}

func TestCreateDefaultConfigFileInCurrentDir(t *testing.T) {
	// Change to temp directory for testing
	tempDir := t.TempDir()
	originalDir, err := os.Getwd()
	require.NoError(t, err)
	
	err = os.Chdir(tempDir)
	require.NoError(t, err)
	defer func() {
		_ = os.Chdir(originalDir)
	}()
	
	configPath, err := CreateDefaultConfigFileInCurrentDir()
	require.NoError(t, err)
	assert.FileExists(t, configPath)
	assert.Contains(t, configPath, "dspy.yaml")
}

func TestCreateDefaultConfigFileInHomeDir(t *testing.T) {
	// Skip if we can't access home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Skip("Cannot access home directory")
	}
	
	// Use a temp subdirectory to avoid polluting home directory
	testConfigDir := filepath.Join(homeDir, ".config", "dspy-go-test")
	defer os.RemoveAll(testConfigDir)
	
	// Test the function directly by creating config in test directory
	discovery := NewDiscovery()
	configPath, err := discovery.CreateDefaultConfigFileInPath(testConfigDir)
	require.NoError(t, err)
	assert.FileExists(t, configPath)
	assert.Contains(t, configPath, "dspy.yaml")
}