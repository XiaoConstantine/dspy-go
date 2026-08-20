// Package jsonv2test provides reusable JSON v2 boundary contract checks.
package jsonv2test

import (
	"bytes"
	"testing"
)

// Contract describes the externally visible defaults expected at JSON v2
// decoding boundaries. Duplicate names and invalid UTF-8 are rejected, while
// case-mismatched and otherwise unknown names are ignored unless a caller opts
// into a stricter unknown-member policy.
type Contract[T any] struct {
	Valid           []byte
	DuplicateMember []byte
	InvalidUTF8     []byte
	CaseMismatch    []byte
	UnknownMember   []byte

	CheckValid            func(testing.TB, T)
	CheckDuplicateError   func(testing.TB, T, error)
	CheckInvalidUTF8Error func(testing.TB, T, error)
	CheckCaseMismatch     func(testing.TB, T)
	CheckUnknownMember    func(testing.TB, T)
}

// Check runs a common strictness and compatibility matrix through decode.
// Callers supply boundary-specific payloads so custom unmarshalers, streaming
// parsers, and HTTP response decoders can share the same policy.
func Check[T any](t *testing.T, decode func([]byte) (T, error), contract Contract[T]) {
	t.Helper()

	tests := []struct {
		name       string
		input      []byte
		wantError  bool
		check      func(testing.TB, T)
		checkError func(testing.TB, T, error)
	}{
		{name: "valid", input: contract.Valid, check: contract.CheckValid},
		{name: "duplicate member", input: contract.DuplicateMember, wantError: true, checkError: contract.CheckDuplicateError},
		{name: "invalid UTF-8", input: contract.InvalidUTF8, wantError: true, checkError: contract.CheckInvalidUTF8Error},
		{name: "case-mismatched member", input: contract.CaseMismatch, check: contract.CheckCaseMismatch},
		{name: "unknown member", input: contract.UnknownMember, check: contract.CheckUnknownMember},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if len(test.input) == 0 {
				t.Fatal("JSON v2 contract fixture is empty")
			}

			got, err := decode(test.input)
			if test.wantError {
				if err == nil {
					t.Fatal("decode unexpectedly accepted strict JSON v2 violation")
				}
				if test.checkError != nil {
					test.checkError(t, got, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("decode rejected compatible JSON v2 input: %v", err)
			}
			if test.check != nil {
				test.check(t, got)
			}
		})
	}
}

// InvalidUTF8 returns a JSON byte sequence containing an invalid byte between
// prefix and suffix. A string helper would risk normalizing the byte.
func InvalidUTF8(prefix, suffix string) []byte {
	data := make([]byte, 0, len(prefix)+1+len(suffix))
	data = append(data, prefix...)
	data = append(data, 0xff)
	return append(data, suffix...)
}

// WithObjectMembers appends members to a JSON object fixture without decoding
// and re-encoding it, preserving malformed byte sequences used by strictness
// tests.
func WithObjectMembers(object, members []byte) []byte {
	object = bytes.TrimSpace(object)
	if len(object) < 2 || object[0] != '{' || object[len(object)-1] != '}' {
		panic("JSON contract fixture must be an object")
	}

	result := make([]byte, 0, len(object)+len(members)+1)
	result = append(result, object[:len(object)-1]...)
	if len(bytes.TrimSpace(object[1:len(object)-1])) != 0 {
		result = append(result, ',')
	}
	result = append(result, members...)
	return append(result, '}')
}
