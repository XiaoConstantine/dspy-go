package agents_test

import (
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

func ExampleMessagesToChatMessages() {
	transcript := []agents.Message{
		agents.NewTextMessage(agents.RoleSystem, "Be concise."),
		agents.NewTextMessage(agents.RoleInternal, "private planning note"),
		agents.NewTextMessage(agents.RoleUser, "What is DSPy?"),
	}

	for _, message := range agents.MessagesToChatMessages(transcript) {
		fmt.Printf("%s: %s\n", message.Role, message.Content[0].Text)
	}

	// Output:
	// system: Be concise.
	// user: What is DSPy?
}
