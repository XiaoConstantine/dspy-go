# PR Reviewer

PR Reviewer demonstrates how to build an intelligent code review assistant using DSPy-Go, implementing practical applications of LLM-powered agents for real-world software development workflows

## Overview

The PR Reviewer is a practical example of using DSPy-Go to create an intelligent system that performs automated code reviews on GitHub pull requests. It showcases how to compose LLM capabilities with traditional software engineering practices to create useful developer tools. The system breaks down the complex task of code review into manageable steps, applies different types of analysis, and generates contextual feedback.

## Getting started
```bash
go run cmd/prreviewer/main.go \
  --api-key="your-llm-api-key" \
  --github-token="your-github-token" \
  --owner="repo-owner" \
  --repo="repo-name" \
  --pr=123
```

### Configuration
The PR Reviewer can be configured to use different LLM backends:

* Anthropic
```go
err = core.ConfigureDefaultLLM(apiKey, core.ModelAnthropicSonnet)

```

* Ollama
```go
err = core.ConfigureDefaultLLM("", "ollama:deepseek-r1:14b-qwen-distill-q4_K_M")
```

