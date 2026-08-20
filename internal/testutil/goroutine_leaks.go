package testutil

import (
	"bytes"
	"runtime/pprof"
	"testing"
)

// CheckGoroutineLeaks reports goroutines that become unreachable while blocked
// after the test and its other cleanup functions have completed. It is intended
// for focused lifecycle tests, not as a package-wide substitute for explicit
// synchronization and shutdown assertions.
func CheckGoroutineLeaks(t testing.TB) {
	t.Helper()

	profile := pprof.Lookup("goroutineleak")
	if profile == nil {
		t.Fatal("runtime/pprof goroutineleak profile is unavailable")
	}

	// Register this before the test creates resources. Cleanup functions run in
	// last-added, first-called order, so resource cleanup registered later gets
	// a chance to finish before the profile triggers leak detection.
	t.Cleanup(func() {
		t.Helper()

		var report bytes.Buffer
		if err := profile.WriteTo(&report, 1); err != nil {
			t.Errorf("write goroutine leak profile: %v", err)
			return
		}
		if count := profile.Count(); count != 0 {
			t.Errorf("detected %d unreachable blocked goroutine(s):\n%s", count, report.String())
		}
	})
}
