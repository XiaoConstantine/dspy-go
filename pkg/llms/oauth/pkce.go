package oauth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
)

// GeneratePKCE generates a PKCE verifier and challenge pair.
// The verifier is a random string, and the challenge is its SHA256 hash.
func GeneratePKCE() (verifier, challenge string, err error) {
	// Generate 32 random bytes for verifier
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", "", err
	}
	verifier = base64URLEncode(bytes)

	// SHA256 hash of verifier for challenge
	hash := sha256.Sum256([]byte(verifier))
	challenge = base64URLEncode(hash[:])

	return verifier, challenge, nil
}

func base64URLEncode(data []byte) string {
	return base64.RawURLEncoding.EncodeToString(data)
}
