---
title: "Getting Started"
description: "Get started with dspy-go"
summary: ""
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 100
toc: true
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---

# Getting Started with dspy-go

This guide will walk you through the basics of setting up `dspy-go` and running your first prediction.

## 1. Installation

To get started, add `dspy-go` to your project using `go get`:

```bash
go get github.com/XiaoConstantine/dspy-go
```

## 2. Set Up Your API Key

`dspy-go` requires an API key for a supported LLM provider (e.g., Google Gemini). The recommended way to configure this is by setting an environment variable.

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

## 3. Your First Program

Now you're ready to write your first `dspy-go` program. The simplest way to start is by using the zero-configuration setup, which automatically configures a default LLM based on your environment variables.

Here's an example of a simple program that asks a large language model to classify the sentiment of a sentence.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// 1. Define a Signature
// A signature describes the task you want the LLM to perform.
// It defines the inputs (InputFields) and outputs (OutputFields).
type SentimentSignature struct {
	core.Signature
}

func (s SentimentSignature) Inputs() []core.InputField {
	return []core.InputField{
		{Field: core.NewField("sentence", core.WithDescription("The sentence to classify."))},
	}
}

func (s SentimentSignature) Outputs() []core.OutputField {
	return []core.OutputField{
		{Field: core.NewField("sentiment", core.WithDescription("The sentiment of the sentence (e.g., Positive, Negative, Neutral)."))},
	}
}

func (s SentimentSignature) Instruction() string {
	return "You are a helpful sentiment analysis expert. Classify the sentiment of the given sentence."
}

func main() {
	// 2. Configure the Default LLM (Zero-Config)
	// This will automatically pick up the GEMINI_API_KEY from your environment.
	err := core.ConfigureDefaultLLMFromEnv()
	if err != nil {
		log.Fatalf("Failed to configure LLM: %v", err)
	}

	// 3. Create a Predict Module
	// The Predict module is the simplest way to use a signature.
	predictor := modules.NewPredict(SentimentSignature{})

	// 4. Execute the Predictor
	ctx := context.Background()
	input := map[string]interface{}{
		"sentence": "dspy-go makes building AI applications easy and fun!",
	}

	result, err := predictor.Process(ctx, input)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// 5. Print the Result
	fmt.Printf("Sentence: %s\n", input["sentence"])
	fmt.Printf("Sentiment: %s\n", result["sentiment"])
}
```

### Running the Code

Save the code above as `main.go`, and then run it from your terminal:

```bash
go run main.go
```

You should see an output like this:

```
Sentence: dspy-go makes building AI applications easy and fun!
Sentiment: Positive
```

You've just successfully used `dspy-go` to run a prediction. From here, you can explore more advanced Modules and Optimizers to build more complex and powerful applications.
