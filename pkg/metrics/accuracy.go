package metrics

import (
	"reflect"
	"strings"
)

// ExactMatch checks if the predicted answer exactly matches the expected answer for all fields.
func ExactMatch(expected, actual map[string]interface{}) float64 {
	for key, expectedValue := range expected {
		if actualValue, ok := actual[key]; !ok || !reflect.DeepEqual(expectedValue, actualValue) {
			return 0.0
		}
	}
	return 1.0
}

// AnyMatch checks if any of the predicted answers match the expected answer for all fields.
func AnyMatch(expected, actual map[string]interface{}) float64 {
	for key, expectedValue := range expected {
		actualValue, ok := actual[key]
		if !ok {
			return 0.0
		}

		if reflect.TypeOf(actualValue).Kind() == reflect.Slice {
			found := false
			slice := reflect.ValueOf(actualValue)
			for i := 0; i < slice.Len(); i++ {
				if reflect.DeepEqual(expectedValue, slice.Index(i).Interface()) {
					found = true
					break
				}
			}
			if !found {
				return 0.0
			}
		} else if !reflect.DeepEqual(expectedValue, actualValue) {
			return 0.0
		}
	}
	return 1.0
}

// F1Score calculates the F1 score between the expected and actual answers.
func F1Score(expected, actual map[string]interface{}) float64 {
	var totalF1 float64
	var count int

	for key, expectedValue := range expected {
		actualValue, ok := actual[key]
		if !ok {
			continue
		}

		expectedStr, expectedOk := expectedValue.(string)
		actualStr, actualOk := actualValue.(string)
		if !expectedOk || !actualOk {
			continue
		}

		expectedTokens := tokenize(expectedStr)
		actualTokens := tokenize(actualStr)

		if len(expectedTokens) == 0 && len(actualTokens) == 0 {
			totalF1 += 1.0
			count++
			continue
		}

		if len(expectedTokens) == 0 || len(actualTokens) == 0 {
			count++
			continue
		}

		intersection := intersection(expectedTokens, actualTokens)
		precision := float64(len(intersection)) / float64(len(actualTokens))
		recall := float64(len(intersection)) / float64(len(expectedTokens))

		if precision+recall > 0 {
			f1 := 2 * precision * recall / (precision + recall)
			totalF1 += f1
			count++
		} else {
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	return totalF1 / float64(count)
}

// Helper functions.
func tokenize(s string) []string {
	return strings.Fields(s)
}

func intersection(a, b []string) []string {
	set := make(map[string]bool)
	for _, item := range a {
		set[item] = true
	}

	var result []string
	for _, item := range b {
		if set[item] {
			result = append(result, item)
			delete(set, item)
		}
	}
	return result
}

// MetricFunc is a type alias for metric functions.
type MetricFunc func(expected, actual map[string]interface{}) float64

// Accuracy is a struct that can be used to create customizable accuracy metrics.
type Accuracy struct {
	MetricFunc MetricFunc
}

// NewAccuracy creates a new Accuracy metric with the specified metric function.
func NewAccuracy(metricFunc MetricFunc) *Accuracy {
	return &Accuracy{MetricFunc: metricFunc}
}

// Evaluate applies the metric function to the expected and actual outputs.
func (a *Accuracy) Evaluate(expected, actual map[string]interface{}) float64 {
	return a.MetricFunc(expected, actual)
}
