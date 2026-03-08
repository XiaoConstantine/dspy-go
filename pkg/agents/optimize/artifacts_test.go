package optimize

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAgentArtifactsClone(t *testing.T) {
	original := AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack: "use focused repo knowledge",
		},
		Int: map[string]int{
			"max_iterations": 8,
		},
		Bool: map[string]bool{
			"parallel_tools": true,
		},
	}

	cloned := original.Clone()
	cloned.Text[ArtifactSkillPack] = "changed"
	cloned.Int["max_iterations"] = 3
	cloned.Bool["parallel_tools"] = false

	assert.Equal(t, "use focused repo knowledge", original.Text[ArtifactSkillPack])
	assert.Equal(t, 8, original.Int["max_iterations"])
	assert.True(t, original.Bool["parallel_tools"])
}
