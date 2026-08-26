---
title: "Multimodal Processing"
description: "Work with images, vision Q&A, multiple images, and streaming"
summary: "Send dspy-go content blocks through llm-go-backed vision models"
date: 2025-10-13T00:00:00+00:00
lastmod: 2026-08-26T00:00:00+00:00
draft: false
weight: 600
toc: true
seo:
  title: "Multimodal Processing - dspy-go"
  description: "Build image analysis and multimodal streaming workflows with dspy-go"
  canonical: ""
  noindex: false
---

# Multimodal Processing

dspy-go represents mixed text and binary inputs with `core.ContentBlock`.
`pkg/llms` converts those blocks to llm-go's provider-neutral message parts,
and llm-go owns the provider wire format.

Gemini, Anthropic, OpenAI, OpenAI Codex, and compatible endpoints can expose
vision through the adapter. Capability varies by model, so check
`llm.Capabilities()` for `core.CapabilityVision` or
`core.CapabilityMultimodal` before relying on image input.

## Image Analysis with a Module

Declare image inputs with `core.NewImageField` and provide values with
`core.NewImageBlock`:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiFlash)
    if err != nil {
        panic(err)
    }

    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewImageField("image",
                core.WithDescription("The image to analyze"))},
            {Field: core.NewTextField("question",
                core.WithDescription("Question about the image"))},
        },
        []core.OutputField{
            {Field: core.NewTextField("answer",
                core.WithDescription("Answer based on the image"))},
        },
    ).WithInstruction("Analyze the image and answer the question.")

    predictor := modules.NewPredict(signature).WithStructuredOutput()
    predictor.SetLLM(llm)

    imageData, err := os.ReadFile("image.jpg")
    if err != nil {
        panic(err)
    }

    result, err := predictor.Process(context.Background(), map[string]any{
        "image":    core.NewImageBlock(imageData, "image/jpeg"),
        "question": "What objects are visible?",
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(result["answer"])
}
```

`Predict` detects non-text fields in the signature and calls
`GenerateWithContent` rather than flattening the image into a text prompt.

## Direct Content Generation

Use the content API when you do not need signature formatting or parsed module
outputs:

```go
content := []core.ContentBlock{
    core.NewTextBlock("Describe this image in detail."),
    core.NewImageBlock(imageData, "image/jpeg"),
}

response, err := llm.GenerateWithContent(ctx, content)
if err != nil {
    return err
}
fmt.Println(response.Content)
```

The MIME type must match the supplied bytes. Common image types include
`image/jpeg`, `image/png`, `image/gif`, and `image/webp`; actual support is
provider- and model-specific.

## Streaming Multimodal Content

```go
content := []core.ContentBlock{
    core.NewTextBlock("Describe the scene."),
    core.NewImageBlock(imageData, "image/jpeg"),
}

stream, err := llm.StreamGenerateWithContent(ctx, content)
if err != nil {
    return err
}
defer stream.Cancel()

for chunk := range stream.ChunkChannel {
    if chunk.Error != nil {
        return chunk.Error
    }
    fmt.Print(chunk.Content)
}
```

Call `Cancel` if the consumer stops before the channel closes. Context
cancellation also propagates to the provider stream.

## Multiple Images

Content blocks preserve order, so a prompt can refer to images by position:

```go
content := []core.ContentBlock{
    core.NewTextBlock("Compare the first image with the second."),
    core.NewImageBlock(before, "image/jpeg"),
    core.NewImageBlock(after, "image/jpeg"),
    core.NewTextBlock("List the important differences."),
}

response, err := llm.GenerateWithContent(ctx, content)
```

For a module, declare one `core.NewImageField` per image and pass a
`core.NewImageBlock` for each corresponding input.

## Audio Blocks

dspy-go can represent audio with `core.NewAudioBlock(data, mimeType)`. Before
using it, check for `core.CapabilityAudio`; not every llm-go provider or model
accepts audio input.

## Model Selection

Use current `core.ModelID` constants rather than retired model strings. For
example:

- Gemini: `core.ModelGoogleGeminiFlash` or `core.ModelGoogleGeminiPro`
- OpenAI: a vision-capable model such as `core.ModelOpenAIGPT4o`
- Anthropic: one of the currently exported Claude family constants

Provider model availability changes independently of dspy-go. Confirm that the
selected model supports the content type and size you send.

## Practical Guidance

- Resize unnecessarily large images before sending them to reduce latency and
  provider cost.
- Preserve the original MIME type after image conversion.
- Use clear questions and output field descriptions; vision output is still
  model generation, not deterministic OCR.
- Keep conversation history in your application or agent transcript. Reusing a
  `Predict` module does not by itself create a persistent multimodal chat.

## Complete Example

The [`examples/multimodal`](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/multimodal)
program covers image analysis, vision Q&A, repeated image interactions,
streaming, and multiple-image input using Gemini:

```bash
export GEMINI_API_KEY="your-api-key"
go run ./examples/multimodal/...
```

## Next Steps

- **[Core Concepts →](../core-concepts/)**
- **[Agents →](../agents/)**
- **[Providers →](../../reference/providers/)**
