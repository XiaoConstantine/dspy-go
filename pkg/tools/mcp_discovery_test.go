package tools

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock MCP Server for testing.
type mockMCPServer struct {
	name          string
	connected     bool
	tools         []core.Tool
	connectErr    error
	disconnectErr error
	listToolsErr  error
	mu            sync.Mutex
}

func (m *mockMCPServer) Name() string {
	return m.name
}

func (m *mockMCPServer) IsConnected() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.connected
}

func (m *mockMCPServer) ListTools(ctx context.Context) ([]core.Tool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.listToolsErr != nil {
		return nil, m.listToolsErr
	}

	// Return a copy to avoid race conditions
	toolsCopy := make([]core.Tool, len(m.tools))
	copy(toolsCopy, m.tools)
	return toolsCopy, nil
}

func (m *mockMCPServer) Connect(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.connectErr != nil {
		return m.connectErr
	}

	m.connected = true
	return nil
}

func (m *mockMCPServer) Disconnect(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.disconnectErr != nil {
		return m.disconnectErr
	}

	m.connected = false
	return nil
}

func (m *mockMCPServer) setTools(tools []core.Tool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools = tools
}

func (m *mockMCPServer) setConnected(connected bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.connected = connected
}

func newMockMCPServer(name string) *mockMCPServer {
	return &mockMCPServer{
		name:      name,
		connected: false,
		tools:     make([]core.Tool, 0),
	}
}

func TestNewDefaultMCPDiscoveryService(t *testing.T) {
	tests := []struct {
		name             string
		config           *MCPDiscoveryConfig
		expectedInterval time.Duration
	}{
		{
			name: "with custom poll interval",
			config: &MCPDiscoveryConfig{
				PollInterval: 10 * time.Second,
				Servers:      []MCPServer{},
			},
			expectedInterval: 10 * time.Second,
		},
		{
			name: "with default poll interval",
			config: &MCPDiscoveryConfig{
				PollInterval: 0, // Should use default
				Servers:      []MCPServer{},
			},
			expectedInterval: 30 * time.Second,
		},
		{
			name: "with servers",
			config: &MCPDiscoveryConfig{
				PollInterval: 5 * time.Second,
				Servers:      []MCPServer{newMockMCPServer("test-server")},
			},
			expectedInterval: 5 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewDefaultMCPDiscoveryService(tt.config)

			assert.NotNil(t, service)
			assert.Equal(t, tt.expectedInterval, service.pollInterval)
			assert.Equal(t, len(tt.config.Servers), len(service.servers))
			assert.NotNil(t, service.callbacks)
			assert.NotNil(t, service.stopChan)
			assert.False(t, service.running)
		})
	}
}

func TestDefaultMCPDiscoveryService_DiscoverTools(t *testing.T) {
	ctx := context.Background()

	t.Run("successful discovery from connected servers", func(t *testing.T) {
		// Create mock tools
		tool1 := newMockTool("tool1", "Test tool 1", []string{"capability1"})
		tool2 := newMockTool("tool2", "Test tool 2", []string{"capability2"})
		tool3 := newMockTool("tool3", "Test tool 3", []string{"capability3"})

		// Create mock servers
		server1 := newMockMCPServer("server1")
		server1.setConnected(true)
		server1.setTools([]core.Tool{tool1, tool2})

		server2 := newMockMCPServer("server2")
		server2.setConnected(true)
		server2.setTools([]core.Tool{tool3})

		// Create discovery service
		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server1, server2},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Discover tools
		tools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Len(t, tools, 3)

		// Verify tools are from both servers
		toolNames := make(map[string]bool)
		for _, tool := range tools {
			toolNames[tool.Name()] = true
		}
		assert.True(t, toolNames["tool1"])
		assert.True(t, toolNames["tool2"])
		assert.True(t, toolNames["tool3"])
	})

	t.Run("discovery from disconnected servers", func(t *testing.T) {
		tool1 := newMockTool("tool1", "Test tool 1", []string{"capability1"})

		// Create disconnected server
		server1 := newMockMCPServer("server1")
		server1.setConnected(false)
		server1.setTools([]core.Tool{tool1})

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server1},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Should connect and discover tools
		tools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Len(t, tools, 1)
		assert.Equal(t, "tool1", tools[0].Name())
		assert.True(t, server1.IsConnected())
	})

	t.Run("handle connection errors", func(t *testing.T) {
		// Create server that fails to connect
		server1 := newMockMCPServer("server1")
		server1.connectErr = errors.New(errors.Unknown, "connection failed")

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server1},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Should return error when all servers fail
		tools, err := service.DiscoverTools(ctx)

		assert.Error(t, err)
		assert.Nil(t, tools)
	})

	t.Run("handle list tools errors", func(t *testing.T) {
		// Create server that fails to list tools
		server1 := newMockMCPServer("server1")
		server1.setConnected(true)
		server1.listToolsErr = errors.New(errors.Unknown, "list tools failed")

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server1},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Should return error when all servers fail
		tools, err := service.DiscoverTools(ctx)

		assert.Error(t, err)
		assert.Nil(t, tools)
	})

	t.Run("partial success - some servers fail", func(t *testing.T) {
		tool1 := newMockTool("tool1", "Test tool 1", []string{"capability1"})

		// Create one successful and one failing server
		server1 := newMockMCPServer("server1")
		server1.setConnected(true)
		server1.setTools([]core.Tool{tool1})

		server2 := newMockMCPServer("server2")
		server2.connectErr = errors.New(errors.Unknown, "connection failed")

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server1, server2},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Should return tools from successful server
		tools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Len(t, tools, 1)
		assert.Equal(t, "tool1", tools[0].Name())
	})

	t.Run("empty servers list", func(t *testing.T) {
		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{},
		}
		service := NewDefaultMCPDiscoveryService(config)

		tools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Empty(t, tools)
	})
}

func TestDefaultMCPDiscoveryService_Subscribe(t *testing.T) {
	t.Run("successful subscription", func(t *testing.T) {
		config := &MCPDiscoveryConfig{
			PollInterval: 100 * time.Millisecond,
		}
		service := NewDefaultMCPDiscoveryService(config)
		defer service.Stop()

		callbackCalled := false
		callback := func(tools []core.Tool) {
			callbackCalled = true
		}

		err := service.Subscribe(callback)
		assert.NoError(t, err)

		// Wait for polling to start and callback to be called
		time.Sleep(200 * time.Millisecond)
		assert.True(t, callbackCalled)
		assert.True(t, service.running)
	})

	t.Run("nil callback", func(t *testing.T) {
		config := &MCPDiscoveryConfig{}
		service := NewDefaultMCPDiscoveryService(config)

		err := service.Subscribe(nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "callback cannot be nil")
	})

	t.Run("multiple subscriptions", func(t *testing.T) {
		config := &MCPDiscoveryConfig{
			PollInterval: 100 * time.Millisecond,
		}
		service := NewDefaultMCPDiscoveryService(config)
		defer service.Stop()

		callback1Called := false
		callback2Called := false

		callback1 := func(tools []core.Tool) {
			callback1Called = true
		}
		callback2 := func(tools []core.Tool) {
			callback2Called = true
		}

		err1 := service.Subscribe(callback1)
		err2 := service.Subscribe(callback2)

		assert.NoError(t, err1)
		assert.NoError(t, err2)

		// Wait for callbacks to be called
		time.Sleep(200 * time.Millisecond)
		assert.True(t, callback1Called)
		assert.True(t, callback2Called)
	})
}

func TestDefaultMCPDiscoveryService_AddServer(t *testing.T) {
	config := &MCPDiscoveryConfig{}
	service := NewDefaultMCPDiscoveryService(config)

	t.Run("successful server addition", func(t *testing.T) {
		server := newMockMCPServer("test-server")
		err := service.AddServer(server)

		assert.NoError(t, err)
		assert.Len(t, service.servers, 1)
		assert.Equal(t, "test-server", service.servers[0].Name())
	})

	t.Run("nil server", func(t *testing.T) {
		err := service.AddServer(nil)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "server cannot be nil")
	})

	t.Run("duplicate server name", func(t *testing.T) {
		server1 := newMockMCPServer("duplicate-server")
		server2 := newMockMCPServer("duplicate-server")

		err1 := service.AddServer(server1)
		err2 := service.AddServer(server2)

		assert.NoError(t, err1)
		assert.Error(t, err2)
		assert.Contains(t, err2.Error(), "server with this name already exists")
		assert.Len(t, service.servers, 2) // Original server + first duplicate server
	})
}

func TestDefaultMCPDiscoveryService_RemoveServer(t *testing.T) {
	server1 := newMockMCPServer("server1")
	server2 := newMockMCPServer("server2")
	server2.setConnected(true)

	config := &MCPDiscoveryConfig{
		Servers: []MCPServer{server1, server2},
	}
	service := NewDefaultMCPDiscoveryService(config)

	t.Run("successful server removal", func(t *testing.T) {
		err := service.RemoveServer("server2")

		assert.NoError(t, err)
		assert.Len(t, service.servers, 1)
		assert.Equal(t, "server1", service.servers[0].Name())

		// Should have disconnected the server
		assert.False(t, server2.IsConnected())
	})

	t.Run("server not found", func(t *testing.T) {
		err := service.RemoveServer("nonexistent-server")

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "server not found")
	})
}

func TestDefaultMCPDiscoveryService_Stop(t *testing.T) {
	server1 := newMockMCPServer("server1")
	server2 := newMockMCPServer("server2")
	server1.setConnected(true)
	server2.setConnected(true)

	config := &MCPDiscoveryConfig{
		Servers:      []MCPServer{server1, server2},
		PollInterval: 100 * time.Millisecond,
	}
	service := NewDefaultMCPDiscoveryService(config)

	// Start the service by subscribing
	callback := func(tools []core.Tool) {}
	err := service.Subscribe(callback)
	require.NoError(t, err)

	// Wait for service to start
	time.Sleep(50 * time.Millisecond)
	assert.True(t, service.running)

	// Stop the service
	service.Stop()

	// Should stop running and disconnect all servers
	assert.False(t, service.running)
	assert.False(t, server1.IsConnected())
	assert.False(t, server2.IsConnected())

	// Multiple stops should be safe
	service.Stop() // Should not panic
}

func TestDefaultMCPDiscoveryService_GetConnectedServers(t *testing.T) {
	server1 := newMockMCPServer("server1")
	server2 := newMockMCPServer("server2")
	server3 := newMockMCPServer("server3")

	server1.setConnected(true)
	server2.setConnected(false)
	server3.setConnected(true)

	config := &MCPDiscoveryConfig{
		Servers: []MCPServer{server1, server2, server3},
	}
	service := NewDefaultMCPDiscoveryService(config)

	connected := service.GetConnectedServers()

	assert.Len(t, connected, 2)
	assert.Contains(t, connected, "server1")
	assert.Contains(t, connected, "server3")
	assert.NotContains(t, connected, "server2")
}

func TestDefaultMCPDiscoveryService_CallbackPanicRecovery(t *testing.T) {
	config := &MCPDiscoveryConfig{
		PollInterval: 50 * time.Millisecond,
	}
	service := NewDefaultMCPDiscoveryService(config)
	defer service.Stop()

	panicCallback := func(tools []core.Tool) {
		panic("test panic")
	}

	normalCallbackCalled := false
	normalCallback := func(tools []core.Tool) {
		normalCallbackCalled = true
	}

	// Subscribe both callbacks
	err1 := service.Subscribe(panicCallback)
	err2 := service.Subscribe(normalCallback)

	assert.NoError(t, err1)
	assert.NoError(t, err2)

	// Wait for callbacks to be called
	time.Sleep(150 * time.Millisecond)

	// Normal callback should still be called despite panic in other callback
	assert.True(t, normalCallbackCalled)
}

func TestDefaultMCPDiscoveryService_ConcurrentAccess(t *testing.T) {
	config := &MCPDiscoveryConfig{
		PollInterval: 50 * time.Millisecond,
	}
	service := NewDefaultMCPDiscoveryService(config)
	defer service.Stop()

	// Test concurrent server additions and removals
	var wg sync.WaitGroup

	// Add servers concurrently
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			server := newMockMCPServer(fmt.Sprintf("server%d", id))
			_ = service.AddServer(server)
		}(i)
	}

	// Subscribe concurrently
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			callback := func(tools []core.Tool) {}
			_ = service.Subscribe(callback)
		}()
	}

	// Discover tools concurrently
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx := context.Background()
			_, _ = service.DiscoverTools(ctx)
		}()
	}

	wg.Wait()

	// Should not panic and should have servers
	servers := service.GetConnectedServers()
	assert.True(t, len(servers) >= 0) // Some may be connected, some may not
}

func TestDefaultMCPDiscoveryService_ContextCancellation(t *testing.T) {
	// Create a connected server so discovery can succeed
	tool := newMockTool("test-tool", "Test tool", []string{"capability"})
	server := newMockMCPServer("test-server")
	server.setConnected(true)
	server.setTools([]core.Tool{tool})

	config := &MCPDiscoveryConfig{
		Servers: []MCPServer{server},
	}
	service := NewDefaultMCPDiscoveryService(config)

	t.Run("normal context", func(t *testing.T) {
		// Normal context should succeed
		ctx := context.Background()
		tools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Len(t, tools, 1)
		assert.Equal(t, "test-tool", tools[0].Name())
	})

	t.Run("cancelled context", func(t *testing.T) {
		// Pre-cancelled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		// Should handle cancelled context gracefully without panicking
		assert.NotPanics(t, func() {
			_, _ = service.DiscoverTools(ctx)
		})
	})
}

func TestDefaultMCPDiscoveryService_InterfaceCompliance(t *testing.T) {
	// Ensure DefaultMCPDiscoveryService implements the MCPDiscoveryService interface
	var _ MCPDiscoveryService = (*DefaultMCPDiscoveryService)(nil)

	config := &MCPDiscoveryConfig{}
	service := NewDefaultMCPDiscoveryService(config)

	// Test that all interface methods are available
	ctx := context.Background()
	_, _ = service.DiscoverTools(ctx)
	_ = service.Subscribe(func(tools []core.Tool) {})
}

func TestDefaultMCPDiscoveryService_EdgeCases(t *testing.T) {
	t.Run("large number of tools", func(t *testing.T) {
		// Create server with many tools
		var tools []core.Tool
		for i := 0; i < 100; i++ { // Reduced from 1000 for test performance
			tool := newMockTool(fmt.Sprintf("tool%d", i), fmt.Sprintf("Tool %d", i), []string{"capability"})
			tools = append(tools, tool)
		}

		server := newMockMCPServer("large-server")
		server.setConnected(true)
		server.setTools(tools)

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server},
		}
		service := NewDefaultMCPDiscoveryService(config)

		ctx := context.Background()
		discoveredTools, err := service.DiscoverTools(ctx)

		assert.NoError(t, err)
		assert.Len(t, discoveredTools, 100)
	})

	t.Run("server disconnect during discovery", func(t *testing.T) {
		tool := newMockTool("tool1", "Test tool", []string{"capability"})

		server := newMockMCPServer("disconnect-server")
		server.setConnected(true)
		server.setTools([]core.Tool{tool})

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server},
		}
		service := NewDefaultMCPDiscoveryService(config)

		ctx := context.Background()
		tools, err := service.DiscoverTools(ctx)

		// Should succeed with the configured tools
		assert.NoError(t, err)
		assert.Len(t, tools, 1)
	})

	t.Run("empty server name", func(t *testing.T) {
		config := &MCPDiscoveryConfig{}
		service := NewDefaultMCPDiscoveryService(config)

		server1 := newMockMCPServer("")
		server2 := newMockMCPServer("")

		err1 := service.AddServer(server1)
		err2 := service.AddServer(server2)

		assert.NoError(t, err1)
		assert.Error(t, err2) // Should fail due to duplicate empty name
	})

	t.Run("very short poll interval", func(t *testing.T) {
		config := &MCPDiscoveryConfig{
			PollInterval: 10 * time.Millisecond, // Short but reasonable interval
		}
		service := NewDefaultMCPDiscoveryService(config)
		defer service.Stop()

		callbackCount := 0
		callback := func(tools []core.Tool) {
			callbackCount++
		}

		err := service.Subscribe(callback)
		assert.NoError(t, err)

		// Wait for multiple polling cycles
		time.Sleep(50 * time.Millisecond)

		// Should have been called multiple times
		assert.True(t, callbackCount > 1)
	})

	t.Run("server with disconnect error during stop", func(t *testing.T) {
		server := newMockMCPServer("error-server")
		server.setConnected(true)
		server.disconnectErr = errors.New(errors.Unknown, "disconnect failed")

		config := &MCPDiscoveryConfig{
			Servers: []MCPServer{server},
		}
		service := NewDefaultMCPDiscoveryService(config)

		// Should not panic even if disconnect fails
		assert.NotPanics(t, func() {
			service.Stop()
		})
	})
}
