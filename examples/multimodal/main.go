package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// ImageAnalysisSignature defines the signature for image analysis.
var ImageAnalysisSignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image", core.WithDescription("The image to analyze"))},
		{Field: core.NewTextField("question", core.WithDescription("Question about the image"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("answer", core.WithDescription("Analysis result"))},
	},
).WithInstruction("Analyze the provided image and answer the given question about it.")

// VisionQASignature defines signature for visual question answering.
var VisionQASignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image", core.WithDescription("The image to examine"))},
		{Field: core.NewTextField("task", core.WithDescription("What to look for in the image"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("observation", core.WithDescription("What you observe in the image"))},
		{Field: core.NewTextField("answer", core.WithDescription("Direct answer to the task"))},
	},
).WithInstruction("Examine the image carefully and provide detailed observations, then answer the specific task.")

// MultiModalChatSignature for conversational interactions with images.
var MultiModalChatSignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image", core.WithDescription("The image in the conversation"))},
		{Field: core.NewTextField("message", core.WithDescription("User's message or question"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("response", core.WithDescription("Assistant's response"))},
	},
).WithInstruction("You are a helpful assistant that can see and analyze images. Respond naturally to the user's message while considering the provided image.")

func main() {
	// Initialize logging with DEBUG level
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	// Check if GEMINI_API_KEY is set
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is required")
	}

	// Create Gemini LLM instance
	llm, err := llms.NewGeminiLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		log.Fatalf("Failed to create Gemini LLM: %v", err)
	}

	// Check if LLM supports multimodal capabilities
	capabilities := llm.Capabilities()
	hasMultimodal := false
	for _, cap := range capabilities {
		if cap == core.CapabilityMultimodal || cap == core.CapabilityVision {
			hasMultimodal = true
			break
		}
	}

	if !hasMultimodal {
		log.Fatal("LLM does not support multimodal capabilities")
	}

	fmt.Println("🎨 Multimodal Gemini Example")
	fmt.Println("Model:", llm.ModelID())
	fmt.Println("Capabilities:", capabilities)
	fmt.Println()

	ctx := core.WithExecutionState(context.Background())

	// Example 1: Basic image analysis
	fmt.Println("📸 Example 1: Basic Image Analysis")
	if err := basicImageAnalysis(ctx, llm); err != nil {
		log.Printf("Basic image analysis failed: %v", err)
	}
	fmt.Println()

	// Example 2: Vision-based Question Answering
	fmt.Println("🔍 Example 2: Vision Question Answering")
	if err := visionQuestionAnswering(ctx, llm); err != nil {
		log.Printf("Vision QA failed: %v", err)
	}
	fmt.Println()

	// Example 3: Multimodal Chat
	fmt.Println("💬 Example 3: Multimodal Chat")
	if err := multiModalChat(ctx, llm); err != nil {
		log.Printf("Multimodal chat failed: %v", err)
	}
	fmt.Println()

	// Example 4: Streaming multimodal generation
	fmt.Println("🌊 Example 4: Streaming Multimodal Generation")
	if err := streamingMultiModal(ctx, llm); err != nil {
		log.Printf("Streaming multimodal failed: %v", err)
	}
	fmt.Println()

	// Example 5: Multiple images analysis
	fmt.Println("📷 Example 5: Multiple Images Analysis")
	if err := multipleImagesAnalysis(ctx, llm); err != nil {
		log.Printf("Multiple images analysis failed: %v", err)
	}
}

func basicImageAnalysis(ctx context.Context, llm core.LLM) error {
	// Load sample image
	imageData, err := loadSampleImage("cat.jpeg")
	if err != nil {
		return fmt.Errorf("failed to load sample image: %w", err)
	}

	// Create image analysis module
	imageAnalyzer := modules.NewPredict(ImageAnalysisSignature)
	imageAnalyzer.SetLLM(llm)

	// Prepare inputs
	inputs := map[string]any{
		"image":    core.NewImageBlock(imageData, "image/jpeg"),
		"question": "What objects can you see in this image?",
	}

	// Execute analysis
	outputs, err := imageAnalyzer.Process(ctx, inputs)
	if err != nil {
		return fmt.Errorf("failed to analyze image: %w", err)
	}

	fmt.Printf("Question: %s\n", inputs["question"])
	fmt.Printf("Answer: %s\n", outputs["answer"])
	return nil
}

func visionQuestionAnswering(ctx context.Context, llm core.LLM) error {
	// Load sample image
	imageData, err := loadSampleImage("cat.jpeg")
	if err != nil {
		return fmt.Errorf("failed to load sample image: %w", err)
	}

	// Create vision QA module
	visionQA := modules.NewPredict(VisionQASignature)
	visionQA.SetLLM(llm)

	// Prepare inputs
	inputs := map[string]any{
		"image": core.NewImageBlock(imageData, "image/jpeg"),
		"task":  "Count the number of people in this image and describe what they are doing.",
	}

	// Execute vision QA
	outputs, err := visionQA.Process(ctx, inputs)
	if err != nil {
		return fmt.Errorf("failed to perform vision QA: %w", err)
	}

	fmt.Printf("Task: %s\n", inputs["task"])
	fmt.Printf("Observation: %s\n", outputs["observation"])
	fmt.Printf("Answer: %s\n", outputs["answer"])
	return nil
}

func multiModalChat(ctx context.Context, llm core.LLM) error {
	// Load sample image
	imageData, err := loadSampleImage("cat.jpeg")
	if err != nil {
		return fmt.Errorf("failed to load sample image: %w", err)
	}

	// Create multimodal chat module
	chatModule := modules.NewPredict(MultiModalChatSignature)
	chatModule.SetLLM(llm)

	// Simulate a conversation
	conversations := []struct {
		message string
		desc    string
	}{
		{"Hello! Can you tell me what you see in this image?", "Initial greeting"},
		{"What's the mood or atmosphere of this scene?", "Follow-up question"},
		{"If you were to give this image a title, what would it be?", "Creative task"},
	}

	for i, conv := range conversations {
		fmt.Printf("Turn %d (%s):\n", i+1, conv.desc)
		fmt.Printf("User: %s\n", conv.message)

		inputs := map[string]any{
			"image":   core.NewImageBlock(imageData, "image/jpeg"),
			"message": conv.message,
		}

		outputs, err := chatModule.Process(ctx, inputs)
		if err != nil {
			return fmt.Errorf("failed to process chat turn %d: %w", i+1, err)
		}

		fmt.Printf("Assistant: %s\n", outputs["response"])
		fmt.Println()
	}

	return nil
}

func streamingMultiModal(ctx context.Context, llm core.LLM) error {
	// Load sample image
	imageData, err := loadSampleImage("cat.jpeg")
	if err != nil {
		return fmt.Errorf("failed to load sample image: %w", err)
	}

	// Create content blocks
	content := []core.ContentBlock{
		core.NewTextBlock("Please describe this image in detail, including colors, objects, people, and the overall scene. Take your time to provide a comprehensive description."),
		core.NewImageBlock(imageData, "image/jpeg"),
	}

	// Start streaming
	fmt.Println("🌊 Streaming response:")
	streamResp, err := llm.StreamGenerateWithContent(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to start streaming: %w", err)
	}

	// Process streaming chunks
	for chunk := range streamResp.ChunkChannel {
		if chunk.Error != nil {
			return fmt.Errorf("streaming error: %w", chunk.Error)
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()

	return nil
}

func multipleImagesAnalysis(ctx context.Context, llm core.LLM) error {
	// For this example, we'll use the same image multiple times
	// In practice, you'd load different images
	imageData, err := loadSampleImage("cat.jpeg")
	if err != nil {
		return fmt.Errorf("failed to load sample image: %w", err)
	}

	// Create content with multiple images and text
	content := []core.ContentBlock{
		core.NewTextBlock("I'm showing you two images. Please compare them and tell me:"),
		core.NewTextBlock("1. What similarities do you notice?"),
		core.NewTextBlock("2. What differences do you see?"),
		core.NewTextBlock("3. Which image do you find more interesting and why?"),
		core.NewTextBlock("First image:"),
		core.NewImageBlock(imageData, "image/jpeg"),
		core.NewTextBlock("Second image:"),
		core.NewImageBlock(imageData, "image/jpeg"), // Same image for demo
	}

	// Generate response
	fmt.Println("📊 Analyzing multiple images...")
	resp, err := llm.GenerateWithContent(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to analyze multiple images: %w", err)
	}

	fmt.Printf("Analysis:\n%s\n", resp.Content)
	if resp.Usage != nil {
		fmt.Printf("Token usage: %d prompt + %d completion = %d total\n",
			resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	}

	return nil
}

func loadSampleImage(filename string) ([]byte, error) {
	// Try to load from current directory first
	if data, err := os.ReadFile(filename); err == nil {
		return data, nil
	}

	// Try to load from examples/multimodal directory
	fullPath := filepath.Join("examples", "multimodal", filename)
	if data, err := os.ReadFile(fullPath); err == nil {
		return data, nil
	}

	// Generate a simple placeholder image if no sample found
	fmt.Printf("Warning: Could not find sample image '%s'. Creating placeholder.\n", filename)
	return createPlaceholderImage(), nil
}

func createPlaceholderImage() []byte {
	// Create a simple base64 encoded 1x1 pixel JPEG for demonstration
	// In practice, you'd provide actual image files
	return []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xD9}
}


// Usage examples and helper functions

func ExampleWithBase64Image() {
	// Example of working with base64 encoded images
	base64Image := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	// Convert base64 to bytes (you'd typically use encoding/base64 package)
	// imageData, _ := base64.StdEncoding.DecodeString(base64Image)

	// Create image block
	// imageBlock := core.NewImageBlock(imageData, "image/png")

	fmt.Printf("Base64 image example (length: %d)\n", len(base64Image))
}

func ExampleWithFileUpload() {
	// Example of handling file uploads
	fileUploadHandler := func(w io.Writer, r io.Reader) error {
		// Read uploaded file
		data, err := io.ReadAll(r)
		if err != nil {
			return err
		}

		// Detect MIME type (you'd use a library like http.DetectContentType)
		mimeType := "image/jpeg" // Simplified for example

		// Create image block
		imageBlock := core.NewImageBlock(data, mimeType)

		// Use the image block in your multimodal pipeline
		fmt.Printf("Processed uploaded image: %s\n", imageBlock.String())
		return nil
	}

	fmt.Printf("File upload handler example: %v\n", fileUploadHandler != nil)
}

func ExampleErrorHandling() {
	// Example of proper error handling for multimodal operations
	examples := []struct {
		name        string
		description string
		handleError func(error) string
	}{
		{
			name:        "Unsupported image format",
			description: "Handle cases where image format is not supported",
			handleError: func(err error) string {
				return fmt.Sprintf("Image format error: %v", err)
			},
		},
		{
			name:        "Image too large",
			description: "Handle cases where image exceeds size limits",
			handleError: func(err error) string {
				return fmt.Sprintf("Image size error: %v", err)
			},
		},
		{
			name:        "LLM API error",
			description: "Handle LLM API errors gracefully",
			handleError: func(err error) string {
				return fmt.Sprintf("LLM API error: %v", err)
			},
		},
	}

	for _, example := range examples {
		fmt.Printf("Error handling example: %s - %s\n", example.name, example.description)
	}
}
