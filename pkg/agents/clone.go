package agents

import (
	"reflect"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// MessageValueCloner lets metadata values with package-specific types define
// their ownership-safe copy without coupling pkg/agents to that package. The
// returned value should be assignable to the original concrete type; nil is
// represented by that type's zero value.
type MessageValueCloner interface {
	CloneMessageValue() any
}

type cloneVisit struct {
	typeOf   reflect.Type
	kind     reflect.Kind
	ptr      uintptr
	length   int
	capacity int
}

type cloneState map[cloneVisit]reflect.Value

// CloneMessages returns copies of the messages whose mutable fields can be
// retained independently. A nil input produces a nil result.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	cloned := make([]Message, len(messages))
	for i, message := range messages {
		cloned[i] = message.Clone()
	}
	return cloned
}

// Clone returns a copy of the message whose mutable slices, maps, content
// blocks, tool calls, and tool result can be retained independently.
func (m Message) Clone() Message {
	state := make(cloneState)
	cloned := Message{
		ID:        m.ID,
		Role:      m.Role,
		Content:   cloneContentBlocksWithState(m.Content, state),
		ToolCalls: cloneToolCallsWithState(m.ToolCalls, state),
		Metadata:  cloneAnyMapWithState(m.Metadata, state),
	}
	if m.ToolResult != nil {
		result := m.ToolResult.cloneWithState(state)
		cloned.ToolResult = &result
	}
	return cloned
}

// Clone returns a copy of the tool result whose mutable fields can be retained
// independently.
func (r MessageToolResult) Clone() MessageToolResult {
	return r.cloneWithState(make(cloneState))
}

func (r MessageToolResult) cloneWithState(state cloneState) MessageToolResult {
	return MessageToolResult{
		ToolCallID:     r.ToolCallID,
		Name:           r.Name,
		Content:        cloneContentBlocksWithState(r.Content, state),
		DisplayContent: cloneContentBlocksWithState(r.DisplayContent, state),
		Details:        cloneAnyMapWithState(r.Details, state),
		IsError:        r.IsError,
		Synthetic:      r.Synthetic,
		Redacted:       r.Redacted,
		Truncated:      r.Truncated,
		Raw:            cloneAnyMapWithState(r.Raw, state),
	}
}

func cloneContentBlocks(blocks []core.ContentBlock) []core.ContentBlock {
	return cloneContentBlocksWithState(blocks, make(cloneState))
}

func cloneContentBlocksWithState(blocks []core.ContentBlock, state cloneState) []core.ContentBlock {
	if blocks == nil {
		return nil
	}
	cloned := make([]core.ContentBlock, len(blocks))
	for i, block := range blocks {
		cloned[i] = block
		cloned[i].Data = cloneBytes(block.Data)
		cloned[i].Metadata = cloneAnyMapWithState(block.Metadata, state)
	}
	return cloned
}

func cloneToolCalls(calls []core.ToolCall) []core.ToolCall {
	return cloneToolCallsWithState(calls, make(cloneState))
}

func cloneToolCallsWithState(calls []core.ToolCall, state cloneState) []core.ToolCall {
	if calls == nil {
		return nil
	}
	cloned := make([]core.ToolCall, len(calls))
	for i, call := range calls {
		cloned[i] = core.ToolCall{
			ID:        call.ID,
			Name:      call.Name,
			Arguments: cloneAnyMapWithState(call.Arguments, state),
			Metadata:  cloneAnyMapWithState(call.Metadata, state),
		}
	}
	return cloned
}

func cloneTypedMap[K comparable, V any](values map[K]V) map[K]V {
	if values == nil {
		return nil
	}
	cloned := cloneReflectValue(reflect.ValueOf(values), make(cloneState))
	return cloned.Interface().(map[K]V)
}

func cloneAnyMap(values map[string]any) map[string]any {
	return cloneAnyMapWithState(values, make(cloneState))
}

func cloneAnyMapWithState(values map[string]any, state cloneState) map[string]any {
	if values == nil {
		return nil
	}
	cloned := cloneReflectValue(reflect.ValueOf(values), state)
	if !cloned.IsValid() {
		return nil
	}
	return cloned.Interface().(map[string]any)
}

func cloneAnyValue(value any) any {
	cloned := cloneReflectValue(reflect.ValueOf(value), make(cloneState))
	if !cloned.IsValid() {
		return nil
	}
	return cloned.Interface()
}

func cloneReflectValue(value reflect.Value, state cloneState) reflect.Value {
	if !value.IsValid() {
		return value
	}

	if value.CanInterface() {
		if cloner, ok := value.Interface().(MessageValueCloner); ok && !isNilValue(cloner) {
			cloned := cloner.CloneMessageValue()
			if cloned == nil {
				return reflect.Zero(value.Type())
			}
			return reflect.ValueOf(cloned)
		}
	}

	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneReflectValue(value.Elem(), state)
		result := reflect.New(value.Type()).Elem()
		if cloned.IsValid() && cloned.Type().AssignableTo(value.Elem().Type()) {
			result.Set(cloned)
		} else if cloned.IsValid() && cloned.Type().AssignableTo(value.Type()) {
			result.Set(cloned)
		} else if cloned.IsValid() && cloned.Type().Implements(value.Type()) {
			result.Set(cloned)
		} else {
			result.Set(value)
		}
		return result
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := cloneVisit{typeOf: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if cloned, ok := state[visit]; ok {
			return cloned
		}
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		state[visit] = cloned
		iterator := value.MapRange()
		for iterator.Next() {
			item := cloneReflectValue(iterator.Value(), state)
			if item.IsValid() && item.Type().AssignableTo(value.Type().Elem()) {
				cloned.SetMapIndex(iterator.Key(), item)
			} else {
				cloned.SetMapIndex(iterator.Key(), reflect.Zero(value.Type().Elem()))
			}
		}
		return cloned
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := cloneVisit{
			typeOf:   value.Type(),
			kind:     value.Kind(),
			ptr:      value.Pointer(),
			length:   value.Len(),
			capacity: value.Cap(),
		}
		if cloned, ok := state[visit]; ok {
			return cloned
		}
		fullSource := value.Slice(0, value.Cap())
		fullClone := reflect.MakeSlice(value.Type(), value.Cap(), value.Cap())
		cloned := fullClone.Slice(0, value.Len())
		state[visit] = cloned
		for i := range value.Cap() {
			item := cloneReflectValue(fullSource.Index(i), state)
			if item.IsValid() && item.Type().AssignableTo(value.Type().Elem()) {
				fullClone.Index(i).Set(item)
			} else {
				fullClone.Index(i).Set(reflect.Zero(value.Type().Elem()))
			}
		}
		return cloned
	case reflect.Array:
		cloned := reflect.New(value.Type()).Elem()
		for i := range value.Len() {
			item := cloneReflectValue(value.Index(i), state)
			if item.IsValid() && item.Type().AssignableTo(value.Type().Elem()) {
				cloned.Index(i).Set(item)
			} else {
				cloned.Index(i).Set(reflect.Zero(value.Type().Elem()))
			}
		}
		return cloned
	default:
		return value
	}
}

func cloneBytes(value []byte) []byte {
	if value == nil {
		return nil
	}
	cloned := make([]byte, len(value))
	copy(cloned, value)
	return cloned
}

func isNilValue(value any) bool {
	reflected := reflect.ValueOf(value)
	return isNilableKind(reflected.Kind()) && reflected.IsNil()
}

func isNilableKind(kind reflect.Kind) bool {
	switch kind {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return true
	default:
		return false
	}
}
