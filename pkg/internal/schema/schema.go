// Package schema compiles a signature's outputs into an ordered JSON Schema,
// validates replies against it, and coerces values to the fields' Go types. It
// is internal so the adapters that use it can change it freely.
package schema

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

const compiledSchemaURL = "https://dspy-go.invalid/output-schema.json"

// CompiledSchema is an ordered JSON Schema compiled from a Signature's outputs.
type CompiledSchema struct {
	Raw              json.RawMessage
	StrictCompatible bool
	PropertyOrder    []string
}

// CompileSignatureSchema builds an ordered JSON Schema for signature.Outputs in
// exact signature declaration order, never round-tripping properties through a
// Go map. StrictCompatible is false if any field uses an open-ended mapping
// (such as map[string]any) that strict structured-output APIs forbid.
func CompileSignatureSchema(signature core.Signature) (CompiledSchema, error) {
	var buf bytes.Buffer
	buf.WriteString(`{"type":"object","properties":{`)

	order := make([]string, 0, len(signature.Outputs))
	strictOK := true
	seen := make(map[string]struct{}, len(signature.Outputs))
	for i, field := range signature.Outputs {
		if field.Name == "" {
			return CompiledSchema{}, fmt.Errorf("output field %d has empty name", i)
		}
		if _, dup := seen[field.Name]; dup {
			return CompiledSchema{}, fmt.Errorf("duplicate output field %q", field.Name)
		}
		seen[field.Name] = struct{}{}
		order = append(order, field.Name)

		fieldSchema, fieldStrict, err := CompileFieldSchema(field)
		if err != nil {
			return CompiledSchema{}, fmt.Errorf("output field %q schema: %w", field.Name, err)
		}
		if !fieldStrict {
			strictOK = false
		}
		if i > 0 {
			buf.WriteByte(',')
		}
		writeJSONString(&buf, field.Name)
		buf.WriteByte(':')
		buf.Write(fieldSchema)
	}

	buf.WriteString(`},"required":[`)
	for i, name := range order {
		if i > 0 {
			buf.WriteByte(',')
		}
		writeJSONString(&buf, name)
	}
	buf.WriteString(`],"additionalProperties":false}`)

	raw := append(json.RawMessage(nil), buf.Bytes()...)
	if !jsontext.Value(raw).IsValid() {
		return CompiledSchema{}, fmt.Errorf("compiled signature schema is not valid JSON")
	}
	return CompiledSchema{
		Raw:              raw,
		StrictCompatible: strictOK,
		PropertyOrder:    order,
	}, nil
}

// CompileFieldSchema compiles one OutputField into ordered JSON Schema bytes
// and reports whether the resulting schema is compatible with strict provider
// structured-output modes.
func CompileFieldSchema(field core.OutputField) (json.RawMessage, bool, error) {
	if len(field.Schema) > 0 {
		trimmed := bytes.TrimSpace([]byte(field.Schema))
		if !jsontext.Value(trimmed).IsValid() {
			return nil, false, fmt.Errorf("explicit field schema is not valid strict JSON")
		}
		return append(json.RawMessage(nil), trimmed...), true, nil
	}

	nullable := field.Optional
	goType := field.GoType
	if goType != nil && goType.Kind() == reflect.Pointer {
		nullable = true
		goType = goType.Elem()
	}

	var buf bytes.Buffer
	strictOK, err := writeTypeSchema(&buf, goType, field.Description, field.EnumValues(), nullable, map[reflect.Type]bool{})
	if err != nil {
		return nil, false, err
	}
	return append(json.RawMessage(nil), buf.Bytes()...), strictOK, nil
}

func writeTypeSchema(
	buf *bytes.Buffer,
	t reflect.Type,
	description string,
	enum []string,
	nullable bool,
	visiting map[reflect.Type]bool,
) (bool, error) {
	// Default legacy untyped field -> string.
	if t == nil {
		return writePrimitiveSchema(buf, "string", description, enum, nullable), nil
	}

	for t.Kind() == reflect.Pointer {
		nullable = true
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return writePrimitiveSchema(buf, "string", description, enum, nullable), nil
	case reflect.Bool:
		return writePrimitiveSchema(buf, "boolean", description, nil, nullable), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return writePrimitiveSchema(buf, "integer", description, nil, nullable), nil
	case reflect.Float32, reflect.Float64:
		return writePrimitiveSchema(buf, "number", description, nil, nullable), nil
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return writePrimitiveSchema(buf, "string", description, nil, nullable), nil
		}
		var itemBuf bytes.Buffer
		itemStrict, err := writeTypeSchema(&itemBuf, t.Elem(), "", nil, false, visiting)
		if err != nil {
			return false, err
		}
		writeObjectSchemaOpen(buf, "array", description, nullable)
		buf.WriteString(`,"items":`)
		buf.Write(itemBuf.Bytes())
		buf.WriteByte('}')
		return itemStrict, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return false, fmt.Errorf("map key type %s must be string for JSON schema", t.Key())
		}
		var valBuf bytes.Buffer
		_, err := writeTypeSchema(&valBuf, t.Elem(), "", nil, false, visiting)
		if err != nil {
			return false, err
		}
		writeObjectSchemaOpen(buf, "object", description, nullable)
		buf.WriteString(`,"additionalProperties":`)
		buf.Write(valBuf.Bytes())
		buf.WriteByte('}')
		// Open-ended mappings are incompatible with strict structured outputs.
		return false, nil
	case reflect.Struct:
		if visiting[t] {
			return false, fmt.Errorf("recursive struct type %s is not supported without explicit schema", t)
		}
		visiting[t] = true
		defer delete(visiting, t)

		var propsBuf bytes.Buffer
		var reqNames []string
		strictOK := true
		first := true
		for i := 0; i < t.NumField(); i++ {
			sf := t.Field(i)
			if !sf.IsExported() {
				continue
			}
			name, skip := structFieldJSONName(sf)
			if skip {
				continue
			}
			fieldOptional := sf.Type.Kind() == reflect.Pointer
			if dspyTag := sf.Tag.Get("dspy"); dspyTag != "" {
				for _, part := range strings.Split(dspyTag, ",")[1:] {
					switch strings.TrimSpace(part) {
					case "optional":
						fieldOptional = true
					case "required":
						fieldOptional = false
					}
				}
			}
			var fieldEnum []string
			if enumTag := sf.Tag.Get("enum"); enumTag != "" {
				for _, item := range strings.Split(enumTag, ",") {
					if trimmed := strings.TrimSpace(item); trimmed != "" {
						fieldEnum = append(fieldEnum, trimmed)
					}
				}
			}
			var propBuf bytes.Buffer
			propStrict, err := writeTypeSchema(&propBuf, sf.Type, sf.Tag.Get("description"), fieldEnum, fieldOptional, visiting)
			if err != nil {
				return false, fmt.Errorf("struct field %s: %w", sf.Name, err)
			}
			if !propStrict {
				strictOK = false
			}
			if !first {
				propsBuf.WriteByte(',')
			}
			first = false
			writeJSONString(&propsBuf, name)
			propsBuf.WriteByte(':')
			propsBuf.Write(propBuf.Bytes())
			reqNames = append(reqNames, name)
		}
		writeObjectSchemaOpen(buf, "object", description, nullable)
		buf.WriteString(`,"properties":{`)
		buf.Write(propsBuf.Bytes())
		buf.WriteString(`},"required":[`)
		for i, name := range reqNames {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeJSONString(buf, name)
		}
		buf.WriteString(`],"additionalProperties":false}`)
		return strictOK, nil
	case reflect.Interface:
		// Open any value: incompatible with strict schema mode.
		buf.WriteByte('{')
		if description != "" {
			buf.WriteString(`"description":`)
			writeJSONString(buf, description)
		}
		buf.WriteByte('}')
		return false, nil
	default:
		return false, fmt.Errorf("unsupported Go type %s for JSON schema", t)
	}
}

func structFieldJSONName(sf reflect.StructField) (string, bool) {
	if dspyTag := sf.Tag.Get("dspy"); dspyTag != "" {
		parts := strings.Split(dspyTag, ",")
		if parts[0] == "-" {
			return "", true
		}
		if parts[0] != "" {
			return parts[0], false
		}
	}
	if jsonTag := sf.Tag.Get("json"); jsonTag != "" {
		parts := strings.Split(jsonTag, ",")
		if parts[0] == "-" {
			return "", true
		}
		if parts[0] != "" {
			return parts[0], false
		}
	}
	return strings.ToLower(sf.Name), false
}

func writePrimitiveSchema(buf *bytes.Buffer, jsonType, description string, enum []string, nullable bool) bool {
	writeObjectSchemaOpen(buf, jsonType, description, nullable)
	if len(enum) > 0 {
		buf.WriteString(`,"enum":[`)
		for i, v := range enum {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeJSONString(buf, v)
		}
		if nullable {
			buf.WriteString(`,null`)
		}
		buf.WriteByte(']')
	}
	buf.WriteByte('}')
	return true
}

func writeObjectSchemaOpen(buf *bytes.Buffer, jsonType, description string, nullable bool) {
	buf.WriteString(`{"type":`)
	if nullable {
		buf.WriteByte('[')
		writeJSONString(buf, jsonType)
		buf.WriteString(`,"null"]`)
	} else {
		writeJSONString(buf, jsonType)
	}
	if description != "" {
		buf.WriteString(`,"description":`)
		writeJSONString(buf, description)
	}
}

func writeJSONString(buf *bytes.Buffer, s string) {
	encoded, _ := json.Marshal(s)
	buf.Write(encoded)
}

// MarshalOrderedObject serializes values into a JSON object whose keys follow
// exact fieldOrder order (never alphabetical map order).
func MarshalOrderedObject(fieldOrder []string, values map[string]any) (json.RawMessage, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	first := true
	for _, name := range fieldOrder {
		val, exists := values[name]
		if !exists {
			continue
		}
		encoded, err := json.Marshal(val)
		if err != nil {
			return nil, fmt.Errorf("marshal field %q: %w", name, err)
		}
		if !first {
			buf.WriteByte(',')
		}
		first = false
		writeJSONString(&buf, name)
		buf.WriteByte(':')
		buf.Write(encoded)
	}
	buf.WriteByte('}')
	return append(json.RawMessage(nil), buf.Bytes()...), nil
}

// compiledSchemaCache caches compiled jsonschema.Schema values keyed by raw schema bytes.
const maxCompiledSchemaCacheBytes = 1 << 20

var compiledSchemaCache struct {
	sync.RWMutex
	entries map[string]*jsonschema.Schema
	bytes   int
}

// ValidateJSON validates raw strict JSON value against schema. It rejects
// duplicate object keys, invalid UTF-8, trailing tokens, and schema violations.
// Matches the contract agreed for llm-go's upcoming llm.ValidateJSON.
func ValidateJSON(schema, value json.RawMessage) error {
	if len(value) == 0 || !jsontext.Value(value).IsValid() {
		return fmt.Errorf("value must be strict JSON")
	}
	compiled, err := compileCachedJSONSchema(schema)
	if err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	decoded, err := jsonschema.UnmarshalJSON(bytes.NewReader(value))
	if err != nil {
		return fmt.Errorf("decode strict JSON: %w", err)
	}
	return compiled.Validate(decoded)
}

func compileCachedJSONSchema(raw []byte) (*jsonschema.Schema, error) {
	key := string(raw)
	compiledSchemaCache.RLock()
	cached, ok := compiledSchemaCache.entries[key]
	compiledSchemaCache.RUnlock()
	if ok {
		return cached, nil
	}

	if len(raw) == 0 || !jsontext.Value(raw).IsValid() {
		return nil, fmt.Errorf("schema must be strict JSON")
	}
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	applySchemaNullableExtension(doc)

	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(offlineSchemaLoader{})
	if err := compiler.AddResource(compiledSchemaURL, doc); err != nil {
		return nil, err
	}
	compiled, err := compiler.Compile(compiledSchemaURL)
	if err != nil {
		return nil, err
	}

	if len(raw) <= maxCompiledSchemaCacheBytes {
		compiledSchemaCache.Lock()
		if compiledSchemaCache.entries == nil || compiledSchemaCache.bytes+len(raw) > maxCompiledSchemaCacheBytes {
			compiledSchemaCache.entries = make(map[string]*jsonschema.Schema)
			compiledSchemaCache.bytes = 0
		}
		compiledSchemaCache.entries[key] = compiled
		compiledSchemaCache.bytes += len(raw)
		compiledSchemaCache.Unlock()
	}
	return compiled, nil
}

type offlineSchemaLoader struct{}

func (offlineSchemaLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("external schema reference %q is disabled", url)
}

func applySchemaNullableExtension(value any) {
	if list, ok := value.([]any); ok {
		for _, item := range list {
			applySchemaNullableExtension(item)
		}
		return
	}
	obj, ok := value.(map[string]any)
	if !ok {
		return
	}
	if nullable, ok := obj["nullable"].(bool); ok && nullable {
		switch t := obj["type"].(type) {
		case string:
			if t != "null" {
				obj["type"] = []any{t, "null"}
			}
		case []any:
			hasNull := false
			for _, item := range t {
				if item == "null" {
					hasNull = true
					break
				}
			}
			if !hasNull {
				obj["type"] = append(t, "null")
			}
		}
	}
	for _, k := range []string{"additionalProperties", "items", "contains", "not", "if", "then", "else"} {
		applySchemaNullableExtension(obj[k])
	}
	for _, k := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
		if children, ok := obj[k].([]any); ok {
			for _, child := range children {
				applySchemaNullableExtension(child)
			}
		}
	}
	for _, k := range []string{"properties", "patternProperties", "$defs", "definitions"} {
		if children, ok := obj[k].(map[string]any); ok {
			for _, child := range children {
				applySchemaNullableExtension(child)
			}
		}
	}
}

// IsNonScalarField reports whether field expects structured JSON (struct, slice,
// array, map, or explicit object/array schema) rather than a scalar string/number/bool.
func IsNonScalarField(field core.OutputField) bool {
	if len(field.Schema) > 0 {
		var meta struct {
			Type any `json:"type"`
		}
		if json.Unmarshal([]byte(field.Schema), &meta) == nil {
			switch t := meta.Type.(type) {
			case string:
				return t == "object" || t == "array"
			case []any:
				for _, item := range t {
					if s, ok := item.(string); ok && (s == "object" || s == "array") {
						return true
					}
				}
			}
		}
		return false
	}
	t := field.GoType
	if t == nil {
		return false
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Struct, reflect.Map, reflect.Array:
		return true
	case reflect.Slice:
		return t.Elem().Kind() != reflect.Uint8
	default:
		return false
	}
}

// CoerceFieldValue converts a parsed string or JSON value into a Go value
// compatible with field.GoType and field.Optional, returning an error on
// invalid conversions.
func CoerceFieldValue(field core.OutputField, raw any) (any, error) {
	if raw == nil {
		if field.Optional || (field.GoType != nil && field.GoType.Kind() == reflect.Pointer) {
			return nil, nil
		}
		return nil, errors.New("required field cannot be null")
	}

	t := field.GoType
	if t == nil {
		// Untyped field defaults to string.
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("expected string, got %T", raw)
		}
		if err := validateEnum(field.EnumValues(), s); err != nil {
			return nil, err
		}
		return s, nil
	}

	isPtr := false
	elemType := t
	if elemType.Kind() == reflect.Pointer {
		isPtr = true
		elemType = elemType.Elem()
	}

	val, err := coerceToType(elemType, field.EnumValues(), raw)
	if err != nil {
		return nil, err
	}
	if !isPtr {
		return val, nil
	}
	ptr := reflect.New(elemType)
	rv := reflect.ValueOf(val)
	if rv.Type().AssignableTo(elemType) {
		ptr.Elem().Set(rv)
	} else if rv.Type().ConvertibleTo(elemType) {
		ptr.Elem().Set(rv.Convert(elemType))
	} else {
		return nil, fmt.Errorf("cannot assign %T to *%s", val, elemType)
	}
	return ptr.Interface(), nil
}

func coerceToType(t reflect.Type, enum []string, raw any) (any, error) {
	switch t.Kind() {
	case reflect.String:
		s, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("expected string, got %T", raw)
		}
		if err := validateEnum(enum, s); err != nil {
			return nil, err
		}
		if t != reflect.TypeOf("") {
			return reflect.ValueOf(s).Convert(t).Interface(), nil
		}
		return s, nil
	case reflect.Bool:
		switch v := raw.(type) {
		case bool:
			return v, nil
		case string:
			b, err := strconv.ParseBool(strings.TrimSpace(v))
			if err != nil {
				return nil, fmt.Errorf("invalid boolean %q", v)
			}
			return b, nil
		default:
			return nil, fmt.Errorf("expected bool, got %T", raw)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		n, err := parseStrictInt(raw, t.Bits())
		if err != nil {
			return nil, err
		}
		out := reflect.New(t).Elem()
		out.SetInt(n)
		return out.Interface(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		n, err := parseStrictUint(raw, t.Bits())
		if err != nil {
			return nil, err
		}
		out := reflect.New(t).Elem()
		out.SetUint(n)
		return out.Interface(), nil
	case reflect.Float32, reflect.Float64:
		f, err := parseStrictFloat(raw, t.Bits())
		if err != nil {
			return nil, err
		}
		out := reflect.New(t).Elem()
		out.SetFloat(f)
		return out.Interface(), nil
	case reflect.Struct, reflect.Slice, reflect.Array, reflect.Map:
		var encoded []byte
		if s, ok := raw.(string); ok {
			encoded = []byte(strings.TrimSpace(s))
		} else {
			var err error
			encoded, err = json.Marshal(raw)
			if err != nil {
				return nil, err
			}
		}
		target := reflect.New(t)
		dec := json.NewDecoder(bytes.NewReader(encoded))
		if err := dec.Decode(target.Interface()); err != nil {
			return nil, fmt.Errorf("decode %s: %w", t, err)
		}
		return target.Elem().Interface(), nil
	case reflect.Interface:
		return raw, nil
	default:
		return nil, fmt.Errorf("unsupported target type %s", t)
	}
}

func validateEnum(enum []string, value string) error {
	if len(enum) == 0 {
		return nil
	}
	for _, allowed := range enum {
		if value == allowed {
			return nil
		}
	}
	return fmt.Errorf("value %q is not one of [%s]", value, strings.Join(enum, ", "))
}

func parseStrictInt(raw any, bitSize int) (int64, error) {
	switch v := raw.(type) {
	case int:
		return int64(v), nil
	case int64:
		return v, nil
	case int32:
		return int64(v), nil
	case json.Number:
		return strconv.ParseInt(v.String(), 10, bitSize)
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) || math.Trunc(v) != v {
			return 0, fmt.Errorf("expected integer, got %v", v)
		}
		return strconv.ParseInt(strconv.FormatFloat(v, 'f', 0, 64), 10, bitSize)
	case string:
		return strconv.ParseInt(strings.TrimSpace(v), 10, bitSize)
	default:
		return 0, fmt.Errorf("expected integer, got %T", raw)
	}
}

func parseStrictUint(raw any, bitSize int) (uint64, error) {
	switch v := raw.(type) {
	case uint:
		return uint64(v), nil
	case uint64:
		return v, nil
	case int:
		if v < 0 {
			return 0, fmt.Errorf("expected unsigned integer, got %d", v)
		}
		return uint64(v), nil
	case int64:
		if v < 0 {
			return 0, fmt.Errorf("expected unsigned integer, got %d", v)
		}
		return uint64(v), nil
	case json.Number:
		return strconv.ParseUint(v.String(), 10, bitSize)
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || math.Trunc(v) != v {
			return 0, fmt.Errorf("expected unsigned integer, got %v", v)
		}
		return strconv.ParseUint(strconv.FormatFloat(v, 'f', 0, 64), 10, bitSize)
	case string:
		return strconv.ParseUint(strings.TrimSpace(v), 10, bitSize)
	default:
		return 0, fmt.Errorf("expected unsigned integer, got %T", raw)
	}
}

func parseStrictFloat(raw any, bitSize int) (float64, error) {
	switch v := raw.(type) {
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("expected finite number, got %v", v)
		}
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case json.Number:
		return strconv.ParseFloat(v.String(), bitSize)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(v), bitSize)
		if err != nil {
			return 0, fmt.Errorf("invalid number %q", v)
		}
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return 0, fmt.Errorf("expected finite number, got %q", v)
		}
		return f, nil
	default:
		return 0, fmt.Errorf("expected number, got %T", raw)
	}
}
