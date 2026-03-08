package core

// ShallowCopyMap returns a shallow copy of the provided map.
func ShallowCopyMap[V any](input map[string]V) map[string]V {
	if input == nil {
		return nil
	}

	result := make(map[string]V, len(input))
	for k, v := range input {
		result[k] = v
	}
	return result
}
