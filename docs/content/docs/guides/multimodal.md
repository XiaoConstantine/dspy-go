---
title: "Multimodal Processing"
description: "Work with images, vision Q&A, and streaming"
summary: "Process images and build multimodal applications with native vision support"
date: 2025-10-13T00:00:00+00:00
lastmod: 2025-10-13T00:00:00+00:00
draft: false
weight: 600
toc: true
seo:
  title: "Multimodal Processing - dspy-go"
  description: "Complete guide to multimodal AI with image analysis, vision Q&A, and streaming in dspy-go"
  canonical: ""
  noindex: false
---

# Multimodal Processing

dspy-go has **native multimodal support** from day one. Process images, build vision Q&A systems, and create multimodal chat applications with seamless integration.

## Why Multimodal?

Modern LLM applications need to work with more than just text:
- 📷 **Image Analysis**: Describe, analyze, and understand images
- 👁️ **Vision Q&A**: Answer questions about visual content
- 💬 **Multimodal Chat**: Conversations mixing text and images
- 🎬 **Streaming**: Real-time multimodal content processing

---

## Supported Providers

| Provider | Image Support | Streaming | Models |
|----------|--------------|-----------|---------|
| **Google Gemini** | ✅ Yes | ✅ Yes | gemini-pro-vision, gemini-1.5-pro |
| **Anthropic Claude** | ✅ Yes | ✅ Yes | claude-3-opus, claude-3-sonnet |
| **OpenAI** | ✅ Yes | ✅ Yes | gpt-4-vision-preview, gpt-4o |

---

## Image Analysis

**Analyze images** with natural language questions.

### Basic Image Analysis

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/llms"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Configure Gemini (has multimodal support)
    llm, err := llms.NewGeminiLLM("your-api-key", core.ModelGoogleGeminiPro)
    if err != nil {
        log.Fatal(err)
    }
    core.SetDefaultLLM(llm)

    // Define signature for image analysis
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("image",
                core.WithDescription("The image to analyze"))},
            {Field: core.NewField("question",
                core.WithDescription("Question about the image"))},
        },
        []core.OutputField{
            {Field: core.NewField("answer",
                core.WithDescription("Answer based on the image"))},
        },
    )

    // Create Predict module
    predictor := modules.NewPredict(signature)

    // Load image
    imageData, err := os.ReadFile("path/to/image.jpg")
    if err != nil {
        log.Fatal(err)
    }

    // Analyze image
    ctx := context.Background()
    result, err := predictor.Process(ctx, map[string]interface{}{
        "image": core.NewImageContent(imageData, "image/jpeg"),
        "question": "What objects are in this image?",
    })

    fmt.Printf("Answer: %s\n", result["answer"])
}
```

---

## Vision Question Answering

**Structured analysis** of visual content.

### Vision Q&A Example

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Configure LLM with vision support
    llm, err := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro) // Will use GEMINI_API_KEY
    if err != nil {
        log.Fatal(err)
    }
    core.SetDefaultLLM(llm)

    // Define a comprehensive vision analysis signature
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewImageField("image",
                core.WithDescription("The image to analyze in detail"))},
            {Field: core.NewField("focus",
                core.WithDescription("Specific aspect to focus on"))},
        },
        []core.OutputField{
            {Field: core.NewField("description",
                core.WithDescription("Detailed description of the image"))},
            {Field: core.NewField("objects",
                core.WithDescription("List of objects identified"))},
            {Field: core.NewField("colors",
                core.WithDescription("Dominant colors in the image"))},
            {Field: core.NewField("mood",
                core.WithDescription("Overall mood or atmosphere"))},
        },
    ).WithInstruction("Analyze the image thoroughly and provide detailed observations.")

    // Create ChainOfThought for detailed analysis
    analyzer := modules.NewChainOfThought(signature)

    // Load image
    imageData, _ := os.ReadFile("photo.jpg")

    // Analyze
    ctx := context.Background()
    result, err := analyzer.Process(ctx, map[string]interface{}{
        "image": core.NewImageContent(imageData, "image/jpeg"),
        "focus": "architectural details and lighting",
    })

    // Print detailed analysis
    fmt.Printf("Description: %s\n", result["description"])
    fmt.Printf("Objects: %s\n", result["objects"])
    fmt.Printf("Colors: %s\n", result["colors"])
    fmt.Printf("Mood: %s\n", result["mood"])
    fmt.Printf("Reasoning: %s\n", result["rationale"]) // From ChainOfThought
}
```

---

## Multimodal Chat

**Interactive conversations** with images.

### Chat with Images

```go
package main

import (
    "context"
    "fmt"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/agents/memory"
)

type MultimodalChat struct {
    predictor modules.Module
    memory    memory.Memory
}

func NewMultimodalChat() *MultimodalChat {
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("history")},
            {Field: core.NewField("user_message")},
            {Field: core.NewField("image")}, // Optional
        },
        []core.OutputField{
            {Field: core.NewField("response")},
        },
    )

    return &MultimodalChat{
        predictor: modules.NewPredict(signature),
        memory:    memory.NewBufferMemory(20),
    }
}

func (c *MultimodalChat) SendText(ctx context.Context, message string) (string, error) {
    history, _ := c.memory.Get(ctx)

    result, err := c.predictor.Process(ctx, map[string]interface{}{
        "history":      formatHistory(history),
        "user_message": message,
    })
    if err != nil {
        return "", err
    }

    response := result["response"].(string)
    c.memory.Add(ctx, "user", message)
    c.memory.Add(ctx, "assistant", response)

    return response, nil
}

func (c *MultimodalChat) SendImage(ctx context.Context, message string, imageData []byte) (string, error) {
    history, _ := c.memory.Get(ctx)

    result, err := c.predictor.Process(ctx, map[string]interface{}{
        "history":      formatHistory(history),
        "user_message": message,
        "image":        core.NewImageContent(imageData, "image/jpeg"),
    })
    if err != nil {
        return "", err
    }

    response := result["response"].(string)
    c.memory.Add(ctx, "user", fmt.Sprintf("%s [image]", message))
    c.memory.Add(ctx, "assistant", response)

    return response, nil
}

func main() {
    chat := NewMultimodalChat()
    ctx := context.Background()

    // Text conversation
    response, _ := chat.SendText(ctx, "Hello! I'm going to show you a photo.")
    fmt.Println("Assistant:", response)

    // Send image
    imageData, _ := os.ReadFile("vacation.jpg")
    response, _ = chat.SendImage(ctx, "Where was this photo taken?", imageData)
    fmt.Println("Assistant:", response)

    // Follow-up question (using conversation memory)
    response, _ = chat.SendText(ctx, "What's the weather like there?")
    fmt.Println("Assistant:", response)
}
```

---

## Streaming Multimodal Content

**Real-time processing** of multimodal content.

### Streaming Example

```go
package main

import (
    "context"
    "fmt"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // Create signature
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("image")},
            {Field: core.NewField("prompt")},
        },
        []core.OutputField{
            {Field: core.NewField("description")},
        },
    )

    // Create module
    predictor := modules.NewPredict(signature)

    // Set streaming handler
    predictor.SetStreamingHandler(func(chunk string) {
        fmt.Print(chunk) // Print each chunk as it arrives
    })

    // Load image
    imageData, _ := os.ReadFile("scene.jpg")

    // Process with streaming
    ctx := context.Background()
    result, err := predictor.Process(ctx, map[string]interface{}{
        "image": core.NewImageContent(imageData, "image/jpeg"),
        "prompt": "Describe this scene in vivid detail",
    })

    fmt.Printf("\n\nFinal: %s\n", result["description"])
}
```

---

## Multiple Images

**Compare and analyze** multiple images simultaneously.

### Multi-Image Analysis

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
    "github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
    // Configure LLM
    llm, _ := llms.NewGeminiLLM("", core.ModelGoogleGeminiPro)
    core.SetDefaultLLM(llm)

    // Define signature for comparing images
    signature := core.NewSignature(
        []core.InputField{
            {Field: core.NewField("image1")},
            {Field: core.NewField("image2")},
            {Field: core.NewField("question")},
        },
        []core.OutputField{
            {Field: core.NewField("comparison")},
            {Field: core.NewField("differences")},
            {Field: core.NewField("similarities")},
        },
    )

    // Create module
    comparator := modules.NewChainOfThought(signature)

    // Load images
    image1, _ := os.ReadFile("before.jpg")
    image2, _ := os.ReadFile("after.jpg")

    // Compare
    ctx := context.Background()
    result, err := comparator.Process(ctx, map[string]interface{}{
        "image1": core.NewImageContent(image1, "image/jpeg"),
        "image2": core.NewImageContent(image2, "image/jpeg"),
        "question": "What changed between these two images?",
    })

    fmt.Printf("Comparison: %s\n", result["comparison"])
    fmt.Printf("Differences: %s\n", result["differences"])
    fmt.Printf("Similarities: %s\n", result["similarities"])
}
```

---

## Content Block System

**Flexible handling** of mixed content types.

### Content Blocks

```go
package main

import (
    "github.com/XiaoConstantine/dspy-go/pkg/core"
)

func main() {
    // Create mixed content
    content := []core.ContentBlock{
        core.NewTextContent("Please analyze this image:"),
        core.NewImageContent(imageData1, "image/jpeg"),
        core.NewTextContent("And compare it to this one:"),
        core.NewImageContent(imageData2, "image/jpeg"),
        core.NewTextContent("What are the key differences?"),
    }

    // Use in module
    result, err := predictor.Process(ctx, map[string]interface{}{
        "content": content,
    })
}
```

---

## Use Cases

### Document Analysis

```go
// Analyze scanned documents
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("document_image")},
    },
    []core.OutputField{
        {Field: core.NewField("text_content")},
        {Field: core.NewField("document_type")},
        {Field: core.NewField("key_information")},
    },
)

extractor := modules.NewChainOfThought(signature)

result, _ := extractor.Process(ctx, map[string]interface{}{
    "document_image": core.NewImageContent(scanData, "image/jpeg"),
})
```

### Chart Analysis

```go
// Extract data from charts and graphs
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("chart_image")},
    },
    []core.OutputField{
        {Field: core.NewField("chart_type")},
        {Field: core.NewField("data_points")},
        {Field: core.NewField("trends")},
        {Field: core.NewField("insights")},
    },
)

analyzer := modules.NewPredict(signature)
```

### Visual Search

```go
// Find similar products
signature := core.NewSignature(
    []core.InputField{
        {Field: core.NewField("query_image")},
        {Field: core.NewField("description")},
    },
    []core.OutputField{
        {Field: core.NewField("product_name")},
        {Field: core.NewField("category")},
        {Field: core.NewField("attributes")},
    },
)

searcher := modules.NewPredict(signature)
```

---

## Best Practices

### Image Preparation

✅ **DO:**
- Use appropriate image formats (JPEG, PNG)
- Resize large images to reduce costs
- Ensure good image quality
- Provide clear, specific questions

❌ **DON'T:**
- Send extremely high-resolution images unnecessarily
- Use corrupted or unclear images
- Ask vague questions
- Expect pixel-perfect OCR (use specialized tools)

### Performance Optimization

```go
// Resize images before sending
func resizeImage(data []byte, maxWidth, maxHeight int) []byte {
    // Your resize implementation
    return resizedData
}

// Use appropriate compression
imageData := resizeImage(originalData, 1024, 1024)
```

### Cost Management

- **Cache results** for repeated queries
- **Use lower-resolution images** when possible
- **Batch similar queries** together
- **Monitor API usage** and costs

---

## Examples

### Complete Multimodal Examples
- **[Multimodal Processing](https://github.com/XiaoConstantine/dspy-go/tree/main/examples/multimodal)** - All multimodal capabilities
  - Basic image analysis
  - Vision Q&A
  - Multimodal chat
  - Streaming
  - Multiple images

### Running the Examples

```bash
# Set API key
export GEMINI_API_KEY="your-api-key"

# Run multimodal example
cd examples/multimodal && go run main.go
```

---

## Next Steps

- **[Core Concepts →](core-concepts/)** - Understand signatures and modules
- **[Agents →](agents/)** - Build agents with vision capabilities
- **[Examples →](https://github.com/XiaoConstantine/dspy-go/tree/main/examples)** - More multimodal examples
