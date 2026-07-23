package agents_test

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPortableAgentsPackage_DoesNotImportIntegrationPackages(t *testing.T) {
	cmd := exec.Command("go", "list", "-f", `{{join .Imports "\n"}}`, "./")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, string(out))
	imports := string(out)
	forbidden := []string{
		"github.com/XiaoConstantine/dspy-go/pkg/llms",
		"github.com/XiaoConstantine/dspy-go/pkg/agents/native",
		"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent",
		"github.com/XiaoConstantine/dspy-go/pkg/agents/skills",
		"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize",
		"github.com/XiaoConstantine/dspy-go/pkg/agents/communication",
	}
	for _, path := range forbidden {
		require.False(t, strings.Contains(imports, path), "pkg/agents must not import %s\nimports:\n%s", path, imports)
	}
}
