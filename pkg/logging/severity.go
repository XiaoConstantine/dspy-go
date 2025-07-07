package logging

// Severity represents log levels with clear mapping to different stages of LLM operations.
type Severity int32

const (
	DEBUG Severity = iota
	INFO
	WARN
	ERROR
	FATAL
)

// String provides human-readable severity levels.
func (s Severity) String() string {
	return [...]string{"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}[s]
}

// ParseSeverity converts a string to a Severity level.
// Returns INFO level for unknown strings.
func ParseSeverity(level string) Severity {
	switch level {
	case "DEBUG":
		return DEBUG
	case "INFO":
		return INFO
	case "WARN":
		return WARN
	case "ERROR":
		return ERROR
	case "FATAL":
		return FATAL
	default:
		return INFO
	}
}
