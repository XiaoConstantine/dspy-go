# ADR 0002: Keep OpenAI Codex subscription transport separate

## Status

Accepted

## Context

OpenAI API keys and ChatGPT subscription OAuth credentials do not use the same
wire protocol. API keys use the public OpenAI API. Codex subscription access
uses `https://chatgpt.com/backend-api/codex/responses`, requires an OAuth bearer
token plus `ChatGPT-Account-ID`, and returns Responses API server-sent events.
Routing an OAuth token through the existing Chat Completions provider is both
incorrect and difficult to audit.

OAuth access tokens are short-lived. Refresh tokens may rotate, and credential
storage belongs to the consuming application rather than the reusable provider.
Long-lived agents therefore cannot copy an access token only when the LLM is
constructed.

## Decision

DSPy-Go registers a distinct `openai-codex` provider implemented in
`pkg/llms/openai_codex.go`.

The provider:

- resolves credentials immediately before each operation;
- reports the rejected access token once after an HTTP 401 so refresh rotation can be correlated safely;
- sends Responses requests with the required account and originator headers;
- translates canonical chat history, function calls, and function outputs;
- preserves ordered opaque Responses output items on the canonical message so
  encrypted reasoning and calls can be replayed exactly on the next turn;
- parses SSE text, function calls, errors, completion status, and token usage;
- implements `core.ToolCallingChatLLM`, preserving `pkg/agents.RunLoop` as the
  sole execution loop;
- does not offer embeddings, which are not part of subscription Codex access.

The application owns OAuth interaction, permission-restricted persistence, and
cross-process serialization of rotating refresh tokens. DSPy-Go supplies PKCE,
independent state generation, context-aware token exchange/refresh helpers, JWT
account-ID extraction, and the operation-time credential resolver contract.

The ordinary `openai` provider accepts API keys only. Explicit configuration
wins over `OPENAI_API_KEY`; `OPENAI_OAUTH_TOKEN` is reserved for
`openai-codex`.

## Consequences

Provider selection is explicit and cannot silently send subscription tokens to
the wrong host. Applications can refresh credentials without rebuilding agents
or introducing another execution loop. Consumers must implement a credential
store and refresh coordinator, which keeps security-sensitive persistence out
of the reusable provider layer.

## Verification

Recorded-request tests assert the endpoint, headers, Responses payload,
canonical tool round trip, ordered continuation, SSE usage parsing,
operation-time resolution, forced refresh retry, and execution through the
shared `agents.RunLoop`. OAuth tests assert independent state and context
cancellation. A live ChatGPT subscription smoke test completed a two-turn tool
call during development; releases should repeat it because it requires
interactive credentials and validates an external protocol.
