package optimizers

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOfflineGEPAProposalRequiresEvidence(t *testing.T) {
	const evidence = `1. Inputs: {"prompt":"ticket=9 symbols=dcba"}
   Expected: {"completion":"dusk coral blue amber"}
   Actual: {"completion":"amber amber amber amber"}`
	for _, tc := range []struct {
		name, parent, evidence, want string
	}{
		{"seed", offlineBaseline, evidence, offlinePartial},
		{"partial", offlinePartial, evidence, offlineCorrect},
		{"bad parent", offlineDecoy, evidence, offlineDecoy},
		{"other bad parent", offlineOther, evidence, offlineDecoy},
		{"no evidence seed", offlineBaseline, "", offlineDecoy},
		{"no evidence partial", offlinePartial, "", offlineDecoy},
		{"headings only", offlinePartial, "Inputs:\nExpected:\nActual:", offlineDecoy},
		{"answer only", offlinePartial, `Expected: {"completion":"dusk coral blue amber"}`, offlineDecoy},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newOfflineLLM()
			prompt := fmt.Sprintf("You are improving a GEPA instruction using reflection-guided evidence.\n\nCURRENT INSTRUCTION: %q\nMODULE: route\n\nEXAMPLE-LEVEL EVIDENCE:\n%s", tc.parent, tc.evidence)
			response, err := model.Generate(context.Background(), prompt)
			require.NoError(t, err)
			require.Equal(t, tc.want, response.Content)
			if tc.want == offlineDecoy {
				score, _, err := offlineScore(context.Background(), offlineWithInstruction(offlineProgram(model), response.Content), offlineExamples(1, 10000), offlineMetric)
				require.NoError(t, err)
				require.Error(t, offlineGate(0.25, score), "proposal without usable evidence or a useful parent survived the gate")
				t.Logf("negative control: score=%.3f, score gate rejected", score)
			}
		})
	}
}
