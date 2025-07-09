package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// ImageUtils provides utility functions for image processing.
type ImageUtils struct{}

// DetectMimeType detects the MIME type of image data.
func (u *ImageUtils) DetectMimeType(data []byte) string {
	contentType := http.DetectContentType(data)
	return contentType
}

// ResizeImage resizes an image to fit within max dimensions while maintaining aspect ratio.
func (u *ImageUtils) ResizeImage(data []byte, maxWidth, maxHeight int) ([]byte, error) {
	// Decode the image
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Get current dimensions
	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	// Check if resizing is needed
	if width <= maxWidth && height <= maxHeight {
		return data, nil // No resizing needed
	}

	// Calculate new dimensions maintaining aspect ratio
	ratio := float64(width) / float64(height)
	newWidth := maxWidth
	newHeight := int(float64(newWidth) / ratio)

	if newHeight > maxHeight {
		newHeight = maxHeight
		newWidth = int(float64(newHeight) * ratio)
	}

	// For this example, we'll return the original data
	// In a real implementation, you'd use an image resizing library
	fmt.Printf("Would resize image from %dx%d to %dx%d\n", width, height, newWidth, newHeight)

	// Re-encode the image
	var buf bytes.Buffer
	switch format {
	case "jpeg":
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
	case "png":
		err = png.Encode(&buf, img)
	default:
		return data, nil // Return original for unsupported formats
	}

	if err != nil {
		return data, nil // Return original on error
	}

	return buf.Bytes(), nil
}

// ValidateImageFormat checks if the image format is supported.
func (u *ImageUtils) ValidateImageFormat(mimeType string) bool {
	supportedFormats := []string{
		"image/jpeg",
		"image/jpg",
		"image/png",
		"image/webp",
		"image/gif",
	}

	for _, format := range supportedFormats {
		if strings.EqualFold(mimeType, format) {
			return true
		}
	}
	return false
}

// MultiModalPipeline demonstrates a pipeline for processing multiple images.
type MultiModalPipeline struct {
	llm   core.LLM
	utils *ImageUtils
}

func NewMultiModalPipeline(llm core.LLM) *MultiModalPipeline {
	return &MultiModalPipeline{
		llm:   llm,
		utils: &ImageUtils{},
	}
}

// ProcessImageBatch processes multiple images in a batch.
func (p *MultiModalPipeline) ProcessImageBatch(ctx context.Context, images []ImageInput, prompt string) ([]ImageAnalysisResult, error) {
	var results []ImageAnalysisResult

	for i, img := range images {
		result, err := p.processSingleImage(ctx, img, prompt)
		if err != nil {
			result = ImageAnalysisResult{
				ImageID: img.ID,
				Error:   err.Error(),
			}
		}
		results = append(results, result)

		// Log progress
		fmt.Printf("Processed image %d/%d: %s\n", i+1, len(images), img.ID)
	}

	return results, nil
}

func (p *MultiModalPipeline) processSingleImage(ctx context.Context, img ImageInput, prompt string) (ImageAnalysisResult, error) {
	// Validate image format
	if !p.utils.ValidateImageFormat(img.MimeType) {
		return ImageAnalysisResult{}, fmt.Errorf("unsupported image format: %s", img.MimeType)
	}

	// Resize if needed
	processedData, err := p.utils.ResizeImage(img.Data, 1024, 1024)
	if err != nil {
		return ImageAnalysisResult{}, fmt.Errorf("failed to resize image: %w", err)
	}

	// Create content blocks
	content := []core.ContentBlock{
		core.NewTextBlock(prompt),
		core.NewImageBlock(processedData, img.MimeType),
	}

	// Generate response
	resp, err := p.llm.GenerateWithContent(ctx, content)
	if err != nil {
		return ImageAnalysisResult{}, fmt.Errorf("failed to generate response: %w", err)
	}

	return ImageAnalysisResult{
		ImageID:     img.ID,
		Analysis:    resp.Content,
		TokenUsage:  resp.Usage,
		ProcessedAt: img.Metadata["timestamp"],
	}, nil
}

// ImageInput represents an input image for processing.
type ImageInput struct {
	ID       string
	Data     []byte
	MimeType string
	Metadata map[string]interface{}
}

// ImageAnalysisResult represents the result of image analysis.
type ImageAnalysisResult struct {
	ImageID     string          `json:"image_id"`
	Analysis    string          `json:"analysis"`
	TokenUsage  *core.TokenInfo `json:"token_usage,omitempty"`
	ProcessedAt interface{}     `json:"processed_at,omitempty"`
	Error       string          `json:"error,omitempty"`
}

// Advanced multimodal signatures

// ImageComparisonSignature compares multiple images.
var ImageComparisonSignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image1", core.WithDescription("First image to compare"))},
		{Field: core.NewImageField("image2", core.WithDescription("Second image to compare"))},
		{Field: core.NewTextField("comparison_criteria", core.WithDescription("What aspects to compare"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("similarities", core.WithDescription("Similarities between images"))},
		{Field: core.NewTextField("differences", core.WithDescription("Differences between images"))},
		{Field: core.NewTextField("conclusion", core.WithDescription("Overall comparison conclusion"))},
	},
).WithInstruction("Compare the two images based on the specified criteria and provide detailed analysis of similarities and differences.")

// ImageCaptioningSignature generates captions for images.
var ImageCaptioningSignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image", core.WithDescription("Image to caption"))},
		{Field: core.NewTextField("style", core.WithDescription("Caption style (e.g., descriptive, creative, technical)"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("caption", core.WithDescription("Generated caption"))},
		{Field: core.NewTextField("alt_text", core.WithDescription("Alternative text for accessibility"))},
	},
).WithInstruction("Generate an appropriate caption and alt text for the image based on the specified style.")

// OCRSignature extracts text from images.
var OCRSignature = core.NewSignature(
	[]core.InputField{
		{Field: core.NewImageField("image", core.WithDescription("Image containing text"))},
		{Field: core.NewTextField("language", core.WithDescription("Expected language of text (optional)"))},
	},
	[]core.OutputField{
		{Field: core.NewTextField("extracted_text", core.WithDescription("Text extracted from image"))},
		{Field: core.NewTextField("confidence", core.WithDescription("Confidence level of extraction"))},
	},
).WithInstruction("Extract all visible text from the image and provide a confidence assessment.")

// Advanced usage examples

// CompareImages demonstrates image comparison functionality.
func CompareImages(ctx context.Context, llm core.LLM, image1Data, image2Data []byte) error {
	fmt.Println("🔍 Comparing two images...")

	comparator := modules.NewPredict(ImageComparisonSignature)
	comparator.SetLLM(llm)

	inputs := map[string]any{
		"image1":              core.NewImageBlock(image1Data, "image/jpeg"),
		"image2":              core.NewImageBlock(image2Data, "image/jpeg"),
		"comparison_criteria": "visual composition, color scheme, subject matter, and overall mood",
	}

	outputs, err := comparator.Process(ctx, inputs)
	if err != nil {
		return fmt.Errorf("failed to compare images: %w", err)
	}

	fmt.Printf("Similarities: %s\n", outputs["similarities"])
	fmt.Printf("Differences: %s\n", outputs["differences"])
	fmt.Printf("Conclusion: %s\n", outputs["conclusion"])

	return nil
}

// GenerateImageCaptions demonstrates image captioning.
func GenerateImageCaptions(ctx context.Context, llm core.LLM, imageData []byte) error {
	fmt.Println("📝 Generating image captions...")

	captioner := modules.NewPredict(ImageCaptioningSignature)
	captioner.SetLLM(llm)

	styles := []string{"descriptive", "creative", "technical"}

	for _, style := range styles {
		inputs := map[string]any{
			"image": core.NewImageBlock(imageData, "image/jpeg"),
			"style": style,
		}

		outputs, err := captioner.Process(ctx, inputs)
		if err != nil {
			fmt.Printf("Failed to generate %s caption: %v\n", style, err)
			continue
		}

		titleCase := cases.Title(language.English)
		fmt.Printf("%s Caption: %s\n", titleCase.String(style), outputs["caption"])
		fmt.Printf("Alt Text: %s\n", outputs["alt_text"])
		fmt.Println()
	}

	return nil
}

// ExtractTextFromImage demonstrates OCR functionality.
func ExtractTextFromImage(ctx context.Context, llm core.LLM, imageData []byte) error {
	fmt.Println("🔤 Extracting text from image...")

	ocr := modules.NewPredict(OCRSignature)
	ocr.SetLLM(llm)

	inputs := map[string]any{
		"image":    core.NewImageBlock(imageData, "image/jpeg"),
		"language": "English",
	}

	outputs, err := ocr.Process(ctx, inputs)
	if err != nil {
		return fmt.Errorf("failed to extract text: %w", err)
	}

	fmt.Printf("Extracted Text: %s\n", outputs["extracted_text"])
	fmt.Printf("Confidence: %s\n", outputs["confidence"])

	return nil
}

// Utility functions for different image sources

// LoadImageFromURL loads an image from a URL.
func LoadImageFromURL(url string) ([]byte, string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("failed to fetch image: status %d", resp.StatusCode)
	}

	// Read the image data
	data := make([]byte, resp.ContentLength)
	_, err = resp.Body.Read(data)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read image data: %w", err)
	}

	contentType := resp.Header.Get("Content-Type")
	return data, contentType, nil
}

// LoadImageFromBase64 loads an image from base64 string.
func LoadImageFromBase64(base64String string) ([]byte, error) {
	// Remove data URL prefix if present
	if strings.HasPrefix(base64String, "data:") {
		parts := strings.SplitN(base64String, ",", 2)
		if len(parts) == 2 {
			base64String = parts[1]
		}
	}

	data, err := base64.StdEncoding.DecodeString(base64String)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64: %w", err)
	}

	return data, nil
}

// GetImageMimeType determines the MIME type from file extension.
func GetImageMimeType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	default:
		return "image/jpeg" // Default fallback
	}
}

// BatchProcessingExample demonstrates batch processing of images.
func BatchProcessingExample(ctx context.Context, llm core.LLM) error {
	fmt.Println("📦 Batch processing example...")

	// Create sample images for batch processing
	sampleImage := createPlaceholderImage()

	images := []ImageInput{
		{
			ID:       "image1",
			Data:     sampleImage,
			MimeType: "image/jpeg",
			Metadata: map[string]interface{}{"timestamp": "2024-01-01T00:00:00Z"},
		},
		{
			ID:       "image2",
			Data:     sampleImage,
			MimeType: "image/jpeg",
			Metadata: map[string]interface{}{"timestamp": "2024-01-01T00:01:00Z"},
		},
	}

	pipeline := NewMultiModalPipeline(llm)
	results, err := pipeline.ProcessImageBatch(ctx, images, "Describe what you see in this image in one sentence.")
	if err != nil {
		return fmt.Errorf("batch processing failed: %w", err)
	}

	// Print results
	for _, result := range results {
		fmt.Printf("Image %s: %s\n", result.ImageID, result.Analysis)
		if result.Error != "" {
			fmt.Printf("  Error: %s\n", result.Error)
		}
	}

	return nil
}
