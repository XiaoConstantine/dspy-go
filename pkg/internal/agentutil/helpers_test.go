package agentutil

import "testing"

func TestTruncateStringPreservesUTF8(t *testing.T) {
	if got := TruncateString("héllo", 4); got != "h..." {
		t.Fatalf("TruncateString() = %q, want %q", got, "h...")
	}
	if got := TruncateString("你好世界", 3); got != "你好世" {
		t.Fatalf("TruncateString() = %q, want %q", got, "你好世")
	}
}
