package logging

// Severity represents log levels with clear mapping to different stages of LLM operations
type Severity int32

const (
	DEBUG Severity = iota
	INFO
	WARN
	ERROR
	FATAL
)

// String provides human-readable severity levels
func (s Severity) String() string {
	return [...]string{"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}[s]
}
