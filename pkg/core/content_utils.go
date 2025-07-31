package core

import (
	"encoding/base64"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// Content creation utilities for multimodal content handling.
// These utilities provide convenient ways to create ContentBlocks from various sources.

// NewImageFromFile creates an image ContentBlock from a file path.
// It reads the file, detects the MIME type, and creates a properly formatted ContentBlock.
func NewImageFromFile(path string) (*ContentBlock, error) {
	if path == "" {
		return nil, errors.New(errors.InvalidInput, "file path cannot be empty")
	}

	// Check if file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("image file not found: %s", path))
	}

	// Read the file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to read image file: %s", path))
	}

	// Detect MIME type
	mimeType := detectImageMimeType(data, path)
	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported image format: %s", path))
	}

	block := NewImageBlock(data, mimeType)
	block.Metadata = map[string]interface{}{
		"source": "file",
		"path":   path,
		"size":   len(data),
	}

	return &block, nil
}

// NewImageFromURL creates an image ContentBlock from a URL.
// It downloads the image, detects the MIME type, and creates a properly formatted ContentBlock.
func NewImageFromURL(url string) (*ContentBlock, error) {
	if url == "" {
		return nil, errors.New(errors.InvalidInput, "URL cannot be empty")
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Download the image
	resp, err := client.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to download image from URL: %s", url))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(errors.ResourceNotFound, fmt.Sprintf("failed to download image: HTTP %d from URL: %s", resp.StatusCode, url))
	}

	// Read the response body
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to read image data from URL: %s", url))
	}

	// Detect MIME type from response headers or content
	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" || strings.Contains(mimeType, "text/") {
		mimeType = detectImageMimeType(data, url)
	}

	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported image format from URL: %s", url))
	}

	block := NewImageBlock(data, mimeType)
	block.Metadata = map[string]interface{}{
		"source":      "url",
		"url":         url,
		"size":        len(data),
		"content_type": resp.Header.Get("Content-Type"),
	}

	return &block, nil
}

// NewImageFromBase64 creates an image ContentBlock from base64 encoded data.
// It decodes the base64 data and creates a properly formatted ContentBlock.
func NewImageFromBase64(data string, mimeType string) (*ContentBlock, error) {
	if data == "" {
		return nil, errors.New(errors.InvalidInput, "base64 data cannot be empty")
	}

	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, "MIME type cannot be empty")
	}

	// Validate MIME type
	if !isValidImageMimeType(mimeType) {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported image MIME type: %s", mimeType))
	}

	// Remove data URL prefix if present
	if strings.HasPrefix(data, "data:") {
		parts := strings.SplitN(data, ",", 2)
		if len(parts) == 2 {
			data = parts[1]
		}
	}

	// Decode base64 data
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to decode base64 image data")
	}

	block := NewImageBlock(decoded, mimeType)
	block.Metadata = map[string]interface{}{
		"source": "base64",
		"size":   len(decoded),
	}

	return &block, nil
}

// NewAudioFromFile creates an audio ContentBlock from a file path.
// It reads the file, detects the MIME type, and creates a properly formatted ContentBlock.
func NewAudioFromFile(path string) (*ContentBlock, error) {
	if path == "" {
		return nil, errors.New(errors.InvalidInput, "file path cannot be empty")
	}

	// Check if file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("audio file not found: %s", path))
	}

	// Read the file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to read audio file: %s", path))
	}

	// Detect MIME type
	mimeType := detectAudioMimeType(data, path)
	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported audio format: %s", path))
	}

	block := NewAudioBlock(data, mimeType)
	block.Metadata = map[string]interface{}{
		"source": "file",
		"path":   path,
		"size":   len(data),
	}

	return &block, nil
}

// NewAudioFromURL creates an audio ContentBlock from a URL.
// It downloads the audio, detects the MIME type, and creates a properly formatted ContentBlock.
func NewAudioFromURL(url string) (*ContentBlock, error) {
	if url == "" {
		return nil, errors.New(errors.InvalidInput, "URL cannot be empty")
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 60 * time.Second, // Longer timeout for audio files
	}

	// Download the audio
	resp, err := client.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to download audio from URL: %s", url))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(errors.ResourceNotFound, fmt.Sprintf("failed to download audio: HTTP %d from URL: %s", resp.StatusCode, url))
	}

	// Read the response body
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, errors.ResourceNotFound, fmt.Sprintf("failed to read audio data from URL: %s", url))
	}

	// Detect MIME type from response headers or content
	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" || strings.Contains(mimeType, "text/") {
		mimeType = detectAudioMimeType(data, url)
	}

	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported audio format from URL: %s", url))
	}

	block := NewAudioBlock(data, mimeType)
	block.Metadata = map[string]interface{}{
		"source":      "url",
		"url":         url,
		"size":        len(data),
		"content_type": resp.Header.Get("Content-Type"),
	}

	return &block, nil
}

// Helper functions for MIME type detection and validation

// detectImageMimeType detects the MIME type of image data.
func detectImageMimeType(data []byte, filename string) string {
	// First try to detect from file content
	if len(data) >= 8 {
		// PNG signature
		if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
			return "image/png"
		}
		// JPEG signature
		if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
			return "image/jpeg"
		}
		// GIF signature
		if len(data) >= 6 && string(data[0:6]) == "GIF87a" || string(data[0:6]) == "GIF89a" {
			return "image/gif"
		}
		// WebP signature
		if len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
			return "image/webp"
		}
	}

	// Fallback to file extension
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	case ".tiff", ".tif":
		return "image/tiff"
	}

	// Try using Go's built-in detection
	detected := mime.TypeByExtension(ext)
	if detected != "" && isValidImageMimeType(detected) {
		return detected
	}

	return ""
}

// detectAudioMimeType detects the MIME type of audio data.
func detectAudioMimeType(data []byte, filename string) string {
	// First try to detect from file content
	if len(data) >= 4 {
		// MP3 signature
		if (data[0] == 0xFF && (data[1]&0xE0) == 0xE0) || // MPEG audio frame
			(len(data) >= 3 && string(data[0:3]) == "ID3") { // ID3 tag
			return "audio/mpeg"
		}
		// WAV signature
		if len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
			return "audio/wav"
		}
		// OGG signature
		if len(data) >= 4 && string(data[0:4]) == "OggS" {
			return "audio/ogg"
		}
		// FLAC signature
		if len(data) >= 4 && string(data[0:4]) == "fLaC" {
			return "audio/flac"
		}
	}

	// Fallback to file extension
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".mp3":
		return "audio/mpeg"
	case ".wav":
		return "audio/wav"
	case ".ogg":
		return "audio/ogg"
	case ".flac":
		return "audio/flac"
	case ".aac":
		return "audio/aac"
	case ".m4a":
		return "audio/mp4"
	case ".wma":
		return "audio/x-ms-wma"
	}

	// Try using Go's built-in detection
	detected := mime.TypeByExtension(ext)
	if detected != "" && isValidAudioMimeType(detected) {
		return detected
	}

	return ""
}

// isValidImageMimeType checks if a MIME type is a valid image type.
func isValidImageMimeType(mimeType string) bool {
	validTypes := []string{
		"image/jpeg",
		"image/png",
		"image/gif",
		"image/webp",
		"image/bmp",
		"image/tiff",
		"image/svg+xml",
	}

	for _, validType := range validTypes {
		if mimeType == validType {
			return true
		}
	}
	return false
}

// isValidAudioMimeType checks if a MIME type is a valid audio type.
func isValidAudioMimeType(mimeType string) bool {
	validTypes := []string{
		"audio/mpeg",
		"audio/wav",
		"audio/wave", // Alternative WAV MIME type
		"audio/ogg",
		"audio/flac",
		"audio/aac",
		"audio/mp4",
		"audio/x-ms-wma",
	}

	for _, validType := range validTypes {
		if mimeType == validType {
			return true
		}
	}
	return false
}

// NewAudioFromBase64 creates an audio ContentBlock from base64 encoded data.
// It decodes the base64 data and creates a properly formatted ContentBlock.
func NewAudioFromBase64(data string, mimeType string) (*ContentBlock, error) {
	if data == "" {
		return nil, errors.New(errors.InvalidInput, "base64 data cannot be empty")
	}

	if mimeType == "" {
		return nil, errors.New(errors.InvalidInput, "MIME type cannot be empty")
	}

	// Validate MIME type
	if !isValidAudioMimeType(mimeType) {
		return nil, errors.New(errors.InvalidInput, fmt.Sprintf("unsupported audio MIME type: %s", mimeType))
	}

	// Remove data URL prefix if present
	if strings.HasPrefix(data, "data:") {
		parts := strings.SplitN(data, ",", 2)
		if len(parts) == 2 {
			data = parts[1]
		}
	}

	// Decode base64 data
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to decode base64 audio data")
	}

	block := NewAudioBlock(decoded, mimeType)
	block.Metadata = map[string]interface{}{
		"source": "base64",
		"size":   len(decoded),
	}

	return &block, nil
}
