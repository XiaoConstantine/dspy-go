# Multimodal Gemini Example

This example demonstrates how to use Google's Gemini model for multimodal tasks including image analysis, vision-based question answering, and multimodal chat capabilities.

## Features

- **Image Analysis**: Analyze images and answer questions about them
- **Vision Question Answering**: Detailed visual analysis with structured outputs
- **Multimodal Chat**: Conversational interactions with images
- **Streaming Support**: Real-time streaming of multimodal responses
- **Multiple Images**: Compare and analyze multiple images simultaneously

## Prerequisites

1. **Google Gemini API Key**: You need a valid API key from Google AI Studio
2. **Go 1.21+**: Make sure you have Go installed
3. **Sample Images**: Place sample images in this directory (optional - placeholder will be created)

## Setup

1. **Set your API key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

2. **Install dependencies**:
   ```bash
   go mod download
   ```

3. **Add sample images** (optional):
   - Place any JPEG images named `sample.jpg` in this directory
   - The example will create a placeholder if no image is found

## Usage

Run the example:
```bash
go run main.go
```

## Examples Included

### 1. Basic Image Analysis
```go
// Analyze an image and answer questions about it
inputs := map[string]any{
    "image":    core.NewImageBlock(imageData, "image/jpeg"),
    "question": "What objects can you see in this image?",
}
```

### 2. Vision Question Answering
```go
// Structured analysis with observations and answers
inputs := map[string]any{
    "image": core.NewImageBlock(imageData, "image/jpeg"),
    "task":  "Count the number of people in this image and describe what they are doing.",
}
```

### 3. Multimodal Chat
```go
// Conversational interactions with images
inputs := map[string]any{
    "image":   core.NewImageBlock(imageData, "image/jpeg"),
    "message": "Hello! Can you tell me what you see in this image?",
}
```

### 4. Streaming Multimodal Generation
```go
// Real-time streaming responses
content := []core.ContentBlock{
    core.NewTextBlock("Please describe this image in detail..."),
    core.NewImageBlock(imageData, "image/jpeg"),
}
streamResp, err := llm.StreamGenerateWithContent(ctx, content)
```

### 5. Multiple Images Analysis
```go
// Compare and analyze multiple images
content := []core.ContentBlock{
    core.NewTextBlock("Compare these two images:"),
    core.NewImageBlock(imageData1, "image/jpeg"),
    core.NewImageBlock(imageData2, "image/jpeg"),
}
```

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- WebP (`.webp`)
- GIF (`.gif`)

## Key Components

### ContentBlock Types
- `core.FieldTypeText`: Text content
- `core.FieldTypeImage`: Image content
- `core.FieldTypeAudio`: Audio content (future support)

### Signatures
The example defines several signatures for different multimodal tasks:
- `ImageAnalysisSignature`: Basic image analysis
- `VisionQASignature`: Structured vision Q&A
- `MultiModalChatSignature`: Conversational interactions

### LLM Capabilities
Gemini supports these multimodal capabilities:
- `core.CapabilityMultimodal`: General multimodal support
- `core.CapabilityVision`: Image/vision processing
- `core.CapabilityAudio`: Audio processing

## Error Handling

The example includes comprehensive error handling for:
- Missing API keys
- Unsupported image formats
- LLM API errors
- File loading issues

## Customization

### Custom Signatures
Create your own multimodal signatures:
```go
customSignature := core.NewSignature(
    []core.InputField{
        {Field: core.NewImageField("image", core.WithDescription("Input image"))},
        {Field: core.NewTextField("prompt", core.WithDescription("Custom prompt"))},
    },
    []core.OutputField{
        {Field: core.NewTextField("result", core.WithDescription("Analysis result"))},
    },
).WithInstruction("Your custom instruction here")
```

### Image Processing
Process images from different sources:
```go
// From file
imageData, err := os.ReadFile("path/to/image.jpg")
imageBlock := core.NewImageBlock(imageData, "image/jpeg")

// From base64
decodedData, err := base64.StdEncoding.DecodeString(base64String)
imageBlock := core.NewImageBlock(decodedData, "image/png")

// From URL (you'd need to fetch it first)
resp, err := http.Get("https://example.com/image.jpg")
imageData, err := io.ReadAll(resp.Body)
imageBlock := core.NewImageBlock(imageData, "image/jpeg")
```

## Performance Tips

1. **Image Size**: Optimize image sizes for better performance
2. **Batch Processing**: Process multiple images in batches when possible
3. **Streaming**: Use streaming for long-running analysis tasks
4. **Caching**: Cache analysis results for repeated queries

## Advanced Usage

### Chain of Thought with Images
```go
cotSignature := core.NewSignature(
    []core.InputField{
        {Field: core.NewImageField("image", core.WithDescription("Image to analyze"))},
        {Field: core.NewTextField("question", core.WithDescription("Question about the image"))},
    },
    []core.OutputField{
        {Field: core.NewTextField("reasoning", core.WithDescription("Step-by-step reasoning"))},
        {Field: core.NewTextField("answer", core.WithDescription("Final answer"))},
    },
).WithInstruction("Analyze the image step by step, showing your reasoning process")
```

### Multi-turn Conversations
```go
// Maintain conversation context with images
conversation := []core.ContentBlock{
    core.NewTextBlock("Previous conversation context..."),
    core.NewImageBlock(imageData, "image/jpeg"),
    core.NewTextBlock("User: What do you see in this image?"),
    core.NewTextBlock("Assistant: I can see..."),
    core.NewTextBlock("User: Can you tell me more about the colors?"),
}
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Make sure your `GEMINI_API_KEY` is set correctly
   - Verify your API key has the necessary permissions

2. **Image Loading Issues**:
   - Check file paths and permissions
   - Ensure image formats are supported
   - Verify image files aren't corrupted

3. **Memory Issues**:
   - Large images may cause memory issues
   - Consider resizing images before processing

4. **Rate Limiting**:
   - Implement retry logic for API calls
   - Add delays between requests if needed

### Debug Mode
Enable debug logging to see detailed request/response information:
```go
// Initialize logging with DEBUG level
import "github.com/XiaoConstantine/dspy-go/pkg/logging"

output := logging.NewConsoleOutput(true, logging.WithColor(true))
logger := logging.NewLogger(logging.Config{
    Severity: logging.DEBUG,
    Outputs:  []logging.Output{output},
})
logging.SetLogger(logger)
```

This will show detailed information about:
- Multimodal content generation
- LLM completion responses
- Response parsing and field extraction
- Token usage statistics

## Next Steps

- Try different image types and formats
- Experiment with different prompt styles
- Combine multimodal analysis with other DSPy modules
- Build applications that use multimodal capabilities

For more examples and documentation, check out the main DSPy-Go repository.
