package localfs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Resolver struct {
	root string
}

func NewResolver(root string) (*Resolver, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, fmt.Errorf("workspace root is required")
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(absRoot, 0o755); err != nil {
		return nil, err
	}
	resolvedRoot, err := filepath.EvalSymlinks(absRoot)
	if err == nil {
		absRoot = resolvedRoot
	}
	return &Resolver{root: filepath.Clean(absRoot)}, nil
}

func (r *Resolver) Root() string {
	if r == nil {
		return ""
	}
	return r.root
}

func (r *Resolver) ResolveSecurePath(input string) (string, error) {
	if r == nil {
		return "", fmt.Errorf("resolver is nil")
	}
	raw := strings.TrimSpace(input)
	if raw == "" || raw == "." {
		return r.root, nil
	}

	target := filepath.Join(r.root, filepath.Clean(raw))
	resolved, err := ResolvePathThroughExistingSymlinks(target)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", input, err)
	}
	return r.ensureWithinRoot(resolved, input)
}

func (r *Resolver) DisplayPath(target string) string {
	if r == nil {
		return ""
	}
	rel, err := filepath.Rel(r.root, target)
	if err != nil || rel == "." {
		return "."
	}
	return filepath.ToSlash(rel)
}

func (r *Resolver) ensureWithinRoot(target, original string) (string, error) {
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(r.root, absTarget)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes workspace root", original)
	}
	return filepath.Clean(absTarget), nil
}

func ResolvePathThroughExistingSymlinks(target string) (string, error) {
	current := filepath.Clean(target)
	missing := make([]string, 0, 4)

	for {
		_, err := os.Lstat(current)
		if err == nil {
			resolved, err := filepath.EvalSymlinks(current)
			if err != nil {
				return "", err
			}
			current = resolved
			break
		}
		if !os.IsNotExist(err) {
			return "", err
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		missing = append(missing, filepath.Base(current))
		current = parent
	}

	for i := len(missing) - 1; i >= 0; i-- {
		current = filepath.Join(current, missing[i])
	}
	return filepath.Clean(current), nil
}
