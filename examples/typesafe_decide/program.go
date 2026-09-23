package main

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type ticketCategory string

const (
	categoryBilling   ticketCategory = "billing"
	categoryTechnical ticketCategory = "technical"
	categoryAccount   ticketCategory = "account"
	categoryProduct   ticketCategory = "product"
	categoryOther     ticketCategory = "other"

	routeDraft = "draft_reply"
	routeTools = "tools_required"
	routeHuman = "human_review"
)

type sampleTicket struct {
	Name string
	Text string
}

var replayTickets = []sampleTicket{
	{
		Name: "answerable account question",
		Text: "How do I update the email address on my account?",
	},
	{
		Name: "production incident",
		Text: "Checkout is unavailable for every customer and no payments can complete.",
	},
	{
		Name: "account-specific billing action",
		Text: "Please refund the duplicate charge on invoice INV-2048.",
	},
	{
		Name: "answerable product question",
		Text: "Where can I export my project data as CSV?",
	},
}

func newCheapGateProgram(client decide.SystemOneClient) (core.Program, error) {
	gate, err := decide.New(
		client,
		gateSignature(),
		decide.Noul("answerable"),
		decide.Noul("needs_human"),
		decide.Choice[ticketCategory]("category",
			decide.Option(categoryBilling, "Charges, invoices, payments, or refunds"),
			decide.Option(categoryTechnical, "Product failures, bugs, or service outages"),
			decide.Option(categoryAccount, "Login, identity, permissions, or account settings"),
			decide.Option(categoryProduct, "Product usage, features, or how-to questions"),
			decide.Option(categoryOther, "Anything not covered by the other categories"),
		),
	)
	if err != nil {
		return core.Program{}, err
	}
	gate.WithName("support gate")

	drafter := modules.NewChainOfThought(draftSignature()).
		WithName("support reply drafter").
		WithStructuredOutput()

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"gate":  gate,
			"draft": drafter,
		},
		cheapGateForward,
	)
	return program, nil
}

func cheapGateForward(programModules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
	gate, gateOK := programModules["gate"].(*decide.Decide)
	drafter, draftOK := programModules["draft"].(*modules.ChainOfThought)

	return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
		if !gateOK || !draftOK {
			return nil, fmt.Errorf("typesafe example: program modules have unexpected types")
		}
		ticket, ok := inputs["ticket"].(string)
		if !ok || ticket == "" {
			return nil, fmt.Errorf("typesafe example: ticket must be a non-empty string")
		}

		decisionResult, err := gate.ProcessDecision(ctx, map[string]any{"ticket": ticket})
		if err != nil {
			return nil, err
		}
		answerable, ok := decide.Get[decide.NoulDecision](decisionResult, "answerable")
		if !ok {
			return nil, fmt.Errorf("typesafe example: answerable evidence is missing")
		}
		needsHuman, ok := decide.Get[decide.NoulDecision](decisionResult, "needs_human")
		if !ok {
			return nil, fmt.Errorf("typesafe example: needs_human evidence is missing")
		}
		category, ok := decide.Get[decide.ChoiceDecision[ticketCategory]](decisionResult, "category")
		if !ok {
			return nil, fmt.Errorf("typesafe example: category evidence is missing")
		}

		outputs := map[string]any{
			"ticket":                  ticket,
			"answerable":              answerable.Value,
			"answerable_probability":  answerable.Probability,
			"needs_human":             needsHuman.Value,
			"needs_human_probability": needsHuman.Probability,
			"category":                category.Value,
			"category_confidence":     category.ProviderConfidence,
			"decision_model":          decisionResult.Model,
			"decision_request_id":     decisionResult.RequestID,
		}

		switch {
		case needsHuman.Value:
			outputs["route"] = routeHuman
			outputs["message"] = "Escalated before generation; a human should review this ticket."
			return outputs, nil
		case !answerable.Value:
			outputs["route"] = routeTools
			outputs["message"] = "No reply drafted; account tools must complete or inspect the requested action first."
			return outputs, nil
		default:
			outputs["route"] = routeDraft
		}

		draft, err := drafter.Process(ctx, map[string]any{
			"ticket":   ticket,
			"category": string(category.Value),
		})
		if err != nil {
			return nil, fmt.Errorf("draft support reply: %w", err)
		}
		reply, ok := draft["reply"].(string)
		if !ok || reply == "" {
			return nil, fmt.Errorf("draft support reply: reply output is missing")
		}
		outputs["reply"] = reply
		if reasoning, found := draft["reasoning"]; found {
			outputs["reasoning"] = reasoning
		}
		return outputs, nil
	}
}

func configureProgramLLM(program *core.Program, llm core.LLM) {
	for _, module := range program.GetModules() {
		// Decide deliberately ignores this call; generative modules retain it.
		module.SetLLM(llm)
	}
}

func gateSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answerable", core.WithDescription("Can a bounded acknowledgment be drafted from the ticket alone, without private account data, external tools, product-specific instructions, or human judgment?"))},
			{Field: core.NewField("needs_human", core.WithDescription("Does policy, security, ambiguity, or incident impact require human review before replying?"))},
			{Field: core.NewField("category", core.WithDescription("Which support category best matches this ticket?"))},
		},
	).WithInstruction("Triage the ticket. Treat account-specific changes and transactions as requiring tools, and broad outages or security-sensitive requests as requiring human review.")
}

func draftSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))},
			{Field: core.NewField("category", core.WithDescription("Category selected by the decision gate"))},
		},
		[]core.OutputField{
			{Field: core.NewField("reply", core.WithDescription("A concise acknowledgment that does not invent account facts, product behavior, navigation steps, or policy"))},
		},
	).WithInstruction("Draft a concise acknowledgment using only the supplied ticket and category. Do not provide product-specific procedures or navigation, invent product behavior or policy, or claim that an account action was completed. Direct the customer to official documentation or support for exact steps.")
}
