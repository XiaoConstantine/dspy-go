package tools

import (
	"context"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// DefaultMCPDiscoveryService provides automatic tool discovery from MCP servers.
type DefaultMCPDiscoveryService struct {
	mu           sync.RWMutex
	servers      []MCPServer
	callbacks    []func(tools []core.Tool)
	pollInterval time.Duration

	lifecycleMu sync.Mutex
	pollCancel  context.CancelFunc
	pollDone    chan struct{}
	running     bool
}

// MCPServer represents an MCP server connection. Implementations must return
// promptly when an operation's context is canceled; discovery shutdown waits
// for in-flight operations to release their resources.
type MCPServer interface {
	Name() string
	IsConnected() bool
	ListTools(ctx context.Context) ([]core.Tool, error)
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error
}

// MCPDiscoveryConfig configures the MCP discovery service.
type MCPDiscoveryConfig struct {
	PollInterval time.Duration
	Servers      []MCPServer
}

// NewDefaultMCPDiscoveryService creates a new MCP discovery service.
func NewDefaultMCPDiscoveryService(config *MCPDiscoveryConfig) *DefaultMCPDiscoveryService {
	if config.PollInterval == 0 {
		config.PollInterval = 30 * time.Second // Default poll interval
	}

	return &DefaultMCPDiscoveryService{
		servers:      config.Servers,
		callbacks:    make([]func(tools []core.Tool), 0),
		pollInterval: config.PollInterval,
	}
}

// DiscoverTools discovers tools from all connected MCP servers.
func (d *DefaultMCPDiscoveryService) DiscoverTools(ctx context.Context) ([]core.Tool, error) {
	d.mu.RLock()
	servers := make([]MCPServer, len(d.servers))
	copy(servers, d.servers)
	d.mu.RUnlock()

	var allTools []core.Tool
	var discoveryErrors []error

	for _, server := range servers {
		// Ensure server is connected
		if !server.IsConnected() {
			if err := server.Connect(ctx); err != nil {
				discoveryErrors = append(discoveryErrors, err)
				continue
			}
		}

		// List tools from this server
		tools, err := server.ListTools(ctx)
		if err != nil {
			discoveryErrors = append(discoveryErrors, err)
			continue
		}

		allTools = append(allTools, tools...)
	}

	// Return partial results even if some servers failed
	if len(allTools) == 0 && len(discoveryErrors) > 0 {
		return nil, errors.WithFields(
			errors.New(errors.Unknown, "failed to discover tools from all servers"),
			errors.Fields{
				"error_count": len(discoveryErrors),
				"errors":      discoveryErrors,
			},
		)
	}

	return allTools, nil
}

// Subscribe adds a callback for tool discovery updates.
func (d *DefaultMCPDiscoveryService) Subscribe(callback func(tools []core.Tool)) error {
	if callback == nil {
		return errors.New(errors.InvalidInput, "callback cannot be nil")
	}

	d.lifecycleMu.Lock()
	defer d.lifecycleMu.Unlock()

	d.mu.Lock()
	d.callbacks = append(d.callbacks, callback)
	d.mu.Unlock()

	if !d.running {
		pollCtx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		d.pollCancel = cancel
		d.pollDone = done
		d.running = true
		go d.startPolling(pollCtx, done)
	}

	return nil
}

// AddServer adds an MCP server to the discovery service.
func (d *DefaultMCPDiscoveryService) AddServer(server MCPServer) error {
	if server == nil {
		return errors.New(errors.InvalidInput, "server cannot be nil")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Check for duplicate server names
	for _, existingServer := range d.servers {
		if existingServer.Name() == server.Name() {
			return errors.WithFields(
				errors.New(errors.InvalidInput, "server with this name already exists"),
				errors.Fields{"server_name": server.Name()},
			)
		}
	}

	d.servers = append(d.servers, server)
	return nil
}

// RemoveServer removes an MCP server from the discovery service.
func (d *DefaultMCPDiscoveryService) RemoveServer(serverName string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	for i, server := range d.servers {
		if server.Name() == serverName {
			// Disconnect the server
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = server.Disconnect(ctx)

			// Remove from slice
			d.servers = append(d.servers[:i], d.servers[i+1:]...)
			return nil
		}
	}

	return errors.WithFields(
		errors.New(errors.ResourceNotFound, "server not found"),
		errors.Fields{"server_name": serverName},
	)
}

// Stop stops the discovery service. It cancels and joins any in-flight
// discovery before disconnecting servers, so MCPServer implementations must
// cooperate with context cancellation.
func (d *DefaultMCPDiscoveryService) Stop() {
	d.lifecycleMu.Lock()
	defer d.lifecycleMu.Unlock()

	if d.running {
		d.pollCancel()
		<-d.pollDone
		d.pollCancel = nil
		d.pollDone = nil
		d.running = false
	}

	d.mu.RLock()
	servers := append([]MCPServer(nil), d.servers...)
	d.mu.RUnlock()

	// Disconnect only after the polling goroutine can no longer use the servers.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, server := range servers {
		if server.IsConnected() {
			_ = server.Disconnect(ctx)
		}
	}
}

// GetConnectedServers returns the list of connected servers.
func (d *DefaultMCPDiscoveryService) GetConnectedServers() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var connected []string
	for _, server := range d.servers {
		if server.IsConnected() {
			connected = append(connected, server.Name())
		}
	}

	return connected
}

// IsRunning returns true if the discovery service is currently running.
func (d *DefaultMCPDiscoveryService) IsRunning() bool {
	d.lifecycleMu.Lock()
	defer d.lifecycleMu.Unlock()
	return d.running
}

// Private methods

func (d *DefaultMCPDiscoveryService) startPolling(ctx context.Context, done chan struct{}) {
	defer close(done)

	ticker := time.NewTicker(d.pollInterval)
	defer ticker.Stop()

	select {
	case <-ctx.Done():
		return
	default:
		d.performDiscovery(ctx)
	}

	for {
		select {
		case <-ticker.C:
			d.performDiscovery(ctx)
		case <-ctx.Done():
			return
		}
	}
}

func (d *DefaultMCPDiscoveryService) performDiscovery(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()

	tools, err := d.DiscoverTools(ctx)
	if err != nil || ctx.Err() != nil {
		// Log error but continue
		return
	}

	// Notify all subscribers
	d.mu.RLock()
	callbacks := make([]func(tools []core.Tool), len(d.callbacks))
	copy(callbacks, d.callbacks)
	d.mu.RUnlock()

	for _, callback := range callbacks {
		// Run callbacks in goroutines to prevent blocking
		go func(cb func(tools []core.Tool)) {
			defer func() {
				if r := recover(); r != nil {
					// Log panic but continue - recovery is intentionally handled silently
					_ = r
				}
			}()

			cb(tools)
		}(callback)
	}
}

// Ensure DefaultMCPDiscoveryService implements the MCPDiscoveryService interface.
var _ MCPDiscoveryService = (*DefaultMCPDiscoveryService)(nil)
