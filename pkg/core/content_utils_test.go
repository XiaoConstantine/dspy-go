package core

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewImageFromFile(t *testing.T) {
	t.Run("Valid PNG file", func(t *testing.T) {
		// Create a temporary PNG file
		tempDir := t.TempDir()
		pngPath := filepath.Join(tempDir, "test.png")

		// PNG signature + minimal data
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}
		err := os.WriteFile(pngPath, pngData, 0644)
		require.NoError(t, err)

		block, err := NewImageFromFile(pngPath)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeImage, block.Type)
		assert.Equal(t, "image/png", block.MimeType)
		assert.Equal(t, pngData, block.Data)
		assert.Equal(t, "file", block.Metadata["source"])
		assert.Equal(t, pngPath, block.Metadata["path"])
		assert.Equal(t, len(pngData), block.Metadata["size"])
	})

	t.Run("Valid JPEG file", func(t *testing.T) {
		tempDir := t.TempDir()
		jpegPath := filepath.Join(tempDir, "test.jpg")

		// JPEG signature + minimal data
		jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46}
		err := os.WriteFile(jpegPath, jpegData, 0644)
		require.NoError(t, err)

		block, err := NewImageFromFile(jpegPath)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeImage, block.Type)
		assert.Equal(t, "image/jpeg", block.MimeType)
		assert.Equal(t, jpegData, block.Data)
	})

	t.Run("File not found", func(t *testing.T) {
		block, err := NewImageFromFile("/nonexistent/file.png")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "image file not found")
	})

	t.Run("Empty file path", func(t *testing.T) {
		block, err := NewImageFromFile("")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "file path cannot be empty")
	})

	t.Run("Unsupported file format", func(t *testing.T) {
		tempDir := t.TempDir()
		txtPath := filepath.Join(tempDir, "test.txt")

		err := os.WriteFile(txtPath, []byte("not an image"), 0644)
		require.NoError(t, err)

		block, err := NewImageFromFile(txtPath)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported image format")
	})
}

func TestNewImageFromURL(t *testing.T) {
	t.Run("Valid PNG from URL", func(t *testing.T) {
		// Create test server
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(pngData)
		}))
		defer server.Close()

		block, err := NewImageFromURL(server.URL)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeImage, block.Type)
		assert.Equal(t, "image/png", block.MimeType)
		assert.Equal(t, pngData, block.Data)
		assert.Equal(t, "url", block.Metadata["source"])
		assert.Equal(t, server.URL, block.Metadata["url"])
	})

	t.Run("Valid JPEG from URL with content detection", func(t *testing.T) {
		// Create test server that doesn't set Content-Type
		jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write(jpegData)
		}))
		defer server.Close()

		block, err := NewImageFromURL(server.URL + "/test.jpg")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeImage, block.Type)
		assert.Equal(t, "image/jpeg", block.MimeType)
		assert.Equal(t, jpegData, block.Data)
	})

	t.Run("Empty URL", func(t *testing.T) {
		block, err := NewImageFromURL("")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "URL cannot be empty")
	})

	t.Run("HTTP error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		block, err := NewImageFromURL(server.URL)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to download image: HTTP 404")
	})

	t.Run("Invalid URL", func(t *testing.T) {
		block, err := NewImageFromURL("not-a-url")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to download image from URL")
	})

	t.Run("Unsupported content type", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("not an image"))
		}))
		defer server.Close()

		block, err := NewImageFromURL(server.URL)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported image format")
	})
}

func TestNewImageFromBase64(t *testing.T) {
	t.Run("Valid base64 PNG", func(t *testing.T) {
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}
		base64Data := base64.StdEncoding.EncodeToString(pngData)

		block, err := NewImageFromBase64(base64Data, "image/png")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeImage, block.Type)
		assert.Equal(t, "image/png", block.MimeType)
		assert.Equal(t, pngData, block.Data)
		assert.Equal(t, "base64", block.Metadata["source"])
		assert.Equal(t, len(pngData), block.Metadata["size"])
	})

	t.Run("Valid base64 with data URL prefix", func(t *testing.T) {
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}
		base64Data := base64.StdEncoding.EncodeToString(pngData)
		dataURL := "data:image/png;base64," + base64Data

		block, err := NewImageFromBase64(dataURL, "image/png")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, pngData, block.Data)
	})

	t.Run("Empty base64 data", func(t *testing.T) {
		block, err := NewImageFromBase64("", "image/png")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "base64 data cannot be empty")
	})

	t.Run("Empty MIME type", func(t *testing.T) {
		block, err := NewImageFromBase64("dGVzdA==", "")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "MIME type cannot be empty")
	})

	t.Run("Invalid MIME type", func(t *testing.T) {
		block, err := NewImageFromBase64("dGVzdA==", "text/plain")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported image MIME type")
	})

	t.Run("Invalid base64 data", func(t *testing.T) {
		block, err := NewImageFromBase64("invalid-base64!", "image/png")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to decode base64 image data")
	})
}

func TestNewAudioFromFile(t *testing.T) {
	t.Run("Valid MP3 file", func(t *testing.T) {
		tempDir := t.TempDir()
		mp3Path := filepath.Join(tempDir, "test.mp3")

		// MP3 signature + minimal data
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00}
		err := os.WriteFile(mp3Path, mp3Data, 0644)
		require.NoError(t, err)

		block, err := NewAudioFromFile(mp3Path)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeAudio, block.Type)
		assert.Equal(t, "audio/mpeg", block.MimeType)
		assert.Equal(t, mp3Data, block.Data)
		assert.Equal(t, "file", block.Metadata["source"])
		assert.Equal(t, mp3Path, block.Metadata["path"])
		assert.Equal(t, len(mp3Data), block.Metadata["size"])
	})

	t.Run("Valid WAV file", func(t *testing.T) {
		tempDir := t.TempDir()
		wavPath := filepath.Join(tempDir, "test.wav")

		// WAV signature + minimal data
		wavData := []byte{'R', 'I', 'F', 'F', 0x00, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E'}
		err := os.WriteFile(wavPath, wavData, 0644)
		require.NoError(t, err)

		block, err := NewAudioFromFile(wavPath)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeAudio, block.Type)
		assert.Equal(t, "audio/wav", block.MimeType)
		assert.Equal(t, wavData, block.Data)
	})

	t.Run("File not found", func(t *testing.T) {
		block, err := NewAudioFromFile("/nonexistent/file.mp3")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "audio file not found")
	})

	t.Run("Empty file path", func(t *testing.T) {
		block, err := NewAudioFromFile("")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "file path cannot be empty")
	})

	t.Run("Unsupported file format", func(t *testing.T) {
		tempDir := t.TempDir()
		txtPath := filepath.Join(tempDir, "test.txt")

		err := os.WriteFile(txtPath, []byte("not an audio"), 0644)
		require.NoError(t, err)

		block, err := NewAudioFromFile(txtPath)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported audio format")
	})
}

func TestNewAudioFromURL(t *testing.T) {
	t.Run("Valid MP3 from URL", func(t *testing.T) {
		// Create test server
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = w.Write(mp3Data)
		}))
		defer server.Close()

		block, err := NewAudioFromURL(server.URL)
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeAudio, block.Type)
		assert.Equal(t, "audio/mpeg", block.MimeType)
		assert.Equal(t, mp3Data, block.Data)
		assert.Equal(t, "url", block.Metadata["source"])
		assert.Equal(t, server.URL, block.Metadata["url"])
	})

	t.Run("Valid WAV from URL with content detection", func(t *testing.T) {
		// Create test server that doesn't set Content-Type
		wavData := []byte{'R', 'I', 'F', 'F', 0x00, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E'}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write(wavData)
		}))
		defer server.Close()

		block, err := NewAudioFromURL(server.URL + "/test.wav")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeAudio, block.Type)
		// Accept either audio/wav or audio/wave (Go's built-in detection returns audio/wave)
		assert.Contains(t, []string{"audio/wav", "audio/wave"}, block.MimeType)
		assert.Equal(t, wavData, block.Data)
	})

	t.Run("Empty URL", func(t *testing.T) {
		block, err := NewAudioFromURL("")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "URL cannot be empty")
	})

	t.Run("HTTP error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		block, err := NewAudioFromURL(server.URL)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to download audio: HTTP 404")
	})

	t.Run("Invalid URL", func(t *testing.T) {
		block, err := NewAudioFromURL("not-a-url")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to download audio from URL")
	})

	t.Run("Unsupported content type", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("not an audio"))
		}))
		defer server.Close()

		block, err := NewAudioFromURL(server.URL)
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported audio format")
	})
}

func TestNewAudioFromBase64(t *testing.T) {
	t.Run("Valid base64 MP3", func(t *testing.T) {
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00}
		base64Data := base64.StdEncoding.EncodeToString(mp3Data)

		block, err := NewAudioFromBase64(base64Data, "audio/mpeg")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, FieldTypeAudio, block.Type)
		assert.Equal(t, "audio/mpeg", block.MimeType)
		assert.Equal(t, mp3Data, block.Data)
		assert.Equal(t, "base64", block.Metadata["source"])
		assert.Equal(t, len(mp3Data), block.Metadata["size"])
	})

	t.Run("Valid base64 with data URL prefix", func(t *testing.T) {
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00}
		base64Data := base64.StdEncoding.EncodeToString(mp3Data)
		dataURL := "data:audio/mpeg;base64," + base64Data

		block, err := NewAudioFromBase64(dataURL, "audio/mpeg")
		require.NoError(t, err)
		assert.NotNil(t, block)
		assert.Equal(t, mp3Data, block.Data)
	})

	t.Run("Empty base64 data", func(t *testing.T) {
		block, err := NewAudioFromBase64("", "audio/mpeg")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "base64 data cannot be empty")
	})

	t.Run("Empty MIME type", func(t *testing.T) {
		block, err := NewAudioFromBase64("dGVzdA==", "")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "MIME type cannot be empty")
	})

	t.Run("Invalid MIME type", func(t *testing.T) {
		block, err := NewAudioFromBase64("dGVzdA==", "text/plain")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "unsupported audio MIME type")
	})

	t.Run("Invalid base64 data", func(t *testing.T) {
		block, err := NewAudioFromBase64("invalid-base64!", "audio/mpeg")
		assert.Error(t, err)
		assert.Nil(t, block)
		assert.Contains(t, err.Error(), "failed to decode base64 audio data")
	})
}

func TestDetectImageMimeType(t *testing.T) {
	t.Run("PNG detection", func(t *testing.T) {
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
		mimeType := detectImageMimeType(pngData, "test.png")
		assert.Equal(t, "image/png", mimeType)
	})

	t.Run("JPEG detection", func(t *testing.T) {
		jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10}
		mimeType := detectImageMimeType(jpegData, "test.jpg")
		assert.Equal(t, "image/jpeg", mimeType)
	})

	t.Run("GIF detection", func(t *testing.T) {
		gifData := []byte("GIF87a")
		mimeType := detectImageMimeType(gifData, "test.gif")
		assert.Equal(t, "image/gif", mimeType)
	})

	t.Run("WebP detection", func(t *testing.T) {
		webpData := []byte("RIFF____WEBP")
		mimeType := detectImageMimeType(webpData, "test.webp")
		assert.Equal(t, "image/webp", mimeType)
	})

	t.Run("Extension fallback", func(t *testing.T) {
		data := []byte("not image data")
		mimeType := detectImageMimeType(data, "test.bmp")
		assert.Equal(t, "image/bmp", mimeType)
	})

	t.Run("Unknown format", func(t *testing.T) {
		data := []byte("not image data")
		mimeType := detectImageMimeType(data, "test.xyz")
		assert.Equal(t, "", mimeType)
	})
}

func TestDetectAudioMimeType(t *testing.T) {
	t.Run("MP3 detection", func(t *testing.T) {
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00}
		mimeType := detectAudioMimeType(mp3Data, "test.mp3")
		assert.Equal(t, "audio/mpeg", mimeType)
	})

	t.Run("WAV detection", func(t *testing.T) {
		wavData := []byte("RIFF____WAVE")
		mimeType := detectAudioMimeType(wavData, "test.wav")
		assert.Equal(t, "audio/wav", mimeType)
	})

	t.Run("OGG detection", func(t *testing.T) {
		oggData := []byte("OggS")
		mimeType := detectAudioMimeType(oggData, "test.ogg")
		assert.Equal(t, "audio/ogg", mimeType)
	})

	t.Run("FLAC detection", func(t *testing.T) {
		flacData := []byte("fLaC")
		mimeType := detectAudioMimeType(flacData, "test.flac")
		assert.Equal(t, "audio/flac", mimeType)
	})

	t.Run("Extension fallback", func(t *testing.T) {
		data := []byte("not audio data")
		mimeType := detectAudioMimeType(data, "test.aac")
		assert.Equal(t, "audio/aac", mimeType)
	})

	t.Run("Unknown format", func(t *testing.T) {
		data := []byte("not audio data")
		mimeType := detectAudioMimeType(data, "test.xyz")
		assert.Equal(t, "", mimeType)
	})
}

func TestIsValidImageMimeType(t *testing.T) {
	validTypes := []string{
		"image/jpeg",
		"image/png",
		"image/gif",
		"image/webp",
		"image/bmp",
		"image/tiff",
		"image/svg+xml",
	}

	for _, mimeType := range validTypes {
		assert.True(t, isValidImageMimeType(mimeType), fmt.Sprintf("Should accept %s", mimeType))
	}

	invalidTypes := []string{
		"text/plain",
		"audio/mpeg",
		"video/mp4",
		"application/json",
		"image/xyz",
	}

	for _, mimeType := range invalidTypes {
		assert.False(t, isValidImageMimeType(mimeType), fmt.Sprintf("Should reject %s", mimeType))
	}
}

func TestIsValidAudioMimeType(t *testing.T) {
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

	for _, mimeType := range validTypes {
		assert.True(t, isValidAudioMimeType(mimeType), fmt.Sprintf("Should accept %s", mimeType))
	}

	invalidTypes := []string{
		"text/plain",
		"image/jpeg",
		"video/mp4",
		"application/json",
		"audio/xyz",
	}

	for _, mimeType := range invalidTypes {
		assert.False(t, isValidAudioMimeType(mimeType), fmt.Sprintf("Should reject %s", mimeType))
	}
}

func TestContentUtilsIntegration(t *testing.T) {
	t.Run("End-to-end image workflow", func(t *testing.T) {
		// Create a PNG file
		tempDir := t.TempDir()
		pngPath := filepath.Join(tempDir, "test.png")
		pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D}
		err := os.WriteFile(pngPath, pngData, 0644)
		require.NoError(t, err)

		// Load from file
		block1, err := NewImageFromFile(pngPath)
		require.NoError(t, err)

		// Convert to base64
		base64Data := base64.StdEncoding.EncodeToString(pngData)

		// Load from base64
		block2, err := NewImageFromBase64(base64Data, "image/png")
		require.NoError(t, err)

		// Compare blocks
		assert.Equal(t, block1.Type, block2.Type)
		assert.Equal(t, block1.MimeType, block2.MimeType)
		assert.Equal(t, block1.Data, block2.Data)

		// Metadata should be different
		assert.Equal(t, "file", block1.Metadata["source"])
		assert.Equal(t, "base64", block2.Metadata["source"])
	})

	t.Run("End-to-end audio workflow", func(t *testing.T) {
		// Create an MP3 file
		tempDir := t.TempDir()
		mp3Path := filepath.Join(tempDir, "test.mp3")
		mp3Data := []byte{0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00}
		err := os.WriteFile(mp3Path, mp3Data, 0644)
		require.NoError(t, err)

		// Load from file
		block1, err := NewAudioFromFile(mp3Path)
		require.NoError(t, err)

		// Convert to base64
		base64Data := base64.StdEncoding.EncodeToString(mp3Data)

		// Load from base64
		block2, err := NewAudioFromBase64(base64Data, "audio/mpeg")
		require.NoError(t, err)

		// Compare blocks
		assert.Equal(t, block1.Type, block2.Type)
		assert.Equal(t, block1.MimeType, block2.MimeType)
		assert.Equal(t, block1.Data, block2.Data)

		// Metadata should be different
		assert.Equal(t, "file", block1.Metadata["source"])
		assert.Equal(t, "base64", block2.Metadata["source"])
	})
}
