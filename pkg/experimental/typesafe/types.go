package typesafe

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// QuestionType identifies a System One question and answer kind.
type QuestionType string

const (
	QuestionTypeNoul   QuestionType = "noul"
	QuestionTypeChoice QuestionType = "choice"
	QuestionTypeScore  QuestionType = "score"
)

// Question is a closed System One question definition. Construct values with
// NoulQuestion, ChoiceQuestion, or ScoreQuestion.
type Question interface {
	questionType() QuestionType
	validate(name string) error
}

// NoulCriteria describes what counts as true and false. A nil criterion is
// encoded as JSON null and leaves that outcome undescribed.
type NoulCriteria struct {
	True  any `json:"true"`
	False any `json:"false"`
}

// NoulQuestion asks a yes/no question.
type NoulQuestion struct {
	Instructions any           `json:"instructions,omitempty"`
	Criteria     *NoulCriteria `json:"criteria,omitempty"`
}

func (NoulQuestion) questionType() QuestionType { return QuestionTypeNoul }

func (q NoulQuestion) validate(name string) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return validationError("questions."+name+".instructions", err.Error())
		}
	}
	if q.Criteria != nil {
		for _, criterion := range []struct {
			label string
			value any
		}{{"true", q.Criteria.True}, {"false", q.Criteria.False}} {
			if criterion.value != nil {
				if err := validateJSONContent(criterion.value); err != nil {
					return validationError("questions."+name+".criteria."+criterion.label, err.Error())
				}
			}
		}
	}
	return nil
}

// MarshalJSON adds the wire discriminator to a Noul question.
func (q NoulQuestion) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type         QuestionType  `json:"type"`
		Instructions any           `json:"instructions,omitempty"`
		Criteria     *NoulCriteria `json:"criteria,omitempty"`
	}
	return json.Marshal(wire{Type: QuestionTypeNoul, Instructions: q.Instructions, Criteria: q.Criteria})
}

// ChoiceQuestion selects one named criterion. Criteria values may be strings,
// JSON objects, JSON arrays, or nil when a label needs no description.
type ChoiceQuestion struct {
	Instructions any            `json:"instructions,omitempty"`
	Criteria     map[string]any `json:"criteria"`
}

func (ChoiceQuestion) questionType() QuestionType { return QuestionTypeChoice }

func (q ChoiceQuestion) validate(name string) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return validationError("questions."+name+".instructions", err.Error())
		}
	}
	if len(q.Criteria) == 0 {
		return validationError("questions."+name+".criteria", "at least one choice is required")
	}
	for _, label := range sortedKeys(q.Criteria) {
		value := q.Criteria[label]
		if label == "" {
			return validationError("questions."+name+".criteria", "choice labels must not be empty")
		}
		if value != nil {
			if err := validateJSONContent(value); err != nil {
				return validationError("questions."+name+".criteria."+label, err.Error())
			}
		}
	}
	return nil
}

// MarshalJSON adds the wire discriminator to a Choice question.
func (q ChoiceQuestion) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type         QuestionType   `json:"type"`
		Instructions any            `json:"instructions,omitempty"`
		Criteria     map[string]any `json:"criteria"`
	}
	return json.Marshal(wire{Type: QuestionTypeChoice, Instructions: q.Instructions, Criteria: q.Criteria})
}

// ScoreQuestion assigns an ordered score. Criteria position is the provider's
// zero-based score level.
type ScoreQuestion struct {
	Instructions any   `json:"instructions,omitempty"`
	Criteria     []any `json:"criteria"`
}

func (ScoreQuestion) questionType() QuestionType { return QuestionTypeScore }

func (q ScoreQuestion) validate(name string) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return validationError("questions."+name+".instructions", err.Error())
		}
	}
	if len(q.Criteria) == 0 {
		return validationError("questions."+name+".criteria", "at least one score level is required")
	}
	for i, value := range q.Criteria {
		if value == nil {
			return validationError(fmt.Sprintf("questions.%s.criteria.%d", name, i), "score descriptions must not be null")
		}
		if err := validateJSONContent(value); err != nil {
			return validationError(fmt.Sprintf("questions.%s.criteria.%d", name, i), err.Error())
		}
	}
	return nil
}

// MarshalJSON adds the wire discriminator to a Score question.
func (q ScoreQuestion) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type         QuestionType `json:"type"`
		Instructions any          `json:"instructions,omitempty"`
		Criteria     []any        `json:"criteria"`
	}
	return json.Marshal(wire{Type: QuestionTypeScore, Instructions: q.Instructions, Criteria: q.Criteria})
}

// SystemOneRequest is the request body for POST /v1/systemone. An empty Model
// uses the client's configured default.
type SystemOneRequest struct {
	State     any                 `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]Question `json:"questions"`
}

func (r SystemOneRequest) validate(defaultModel string) error {
	if err := validateJSONContent(r.State); err != nil {
		return validationError("state", err.Error())
	}
	model := r.Model
	if model == "" {
		model = defaultModel
	}
	if strings.TrimSpace(model) == "" {
		return validationError("model", "must not be empty")
	}
	if len(r.Questions) == 0 {
		return validationError("questions", "at least one question is required")
	}
	for _, name := range sortedKeys(r.Questions) {
		question := r.Questions[name]
		if strings.TrimSpace(name) == "" {
			return validationError("questions", "question names must not be empty")
		}
		if question == nil || isNilInterface(question) {
			return validationError("questions."+name, "question must not be nil")
		}
		if err := question.validate(name); err != nil {
			return err
		}
	}
	return nil
}

// Answer is a tagged answer returned by System One.
type Answer interface {
	AnswerType() QuestionType
	answer()
}

// NoulAnswer contains the probability of a true answer.
type NoulAnswer struct {
	Probability float64
}

func (NoulAnswer) AnswerType() QuestionType { return QuestionTypeNoul }
func (NoulAnswer) answer()                  {}

// MarshalJSON encodes a Noul answer using the TypeSafe wire names.
func (a NoulAnswer) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type QuestionType `json:"type"`
		Noul float64      `json:"noul"`
	}{Type: QuestionTypeNoul, Noul: a.Probability})
}

// ChoiceAnswer contains the provider's selection and its evidence.
type ChoiceAnswer struct {
	Choice        string
	Confidence    float64
	Probabilities map[string]float64
}

func (ChoiceAnswer) AnswerType() QuestionType { return QuestionTypeChoice }
func (ChoiceAnswer) answer()                  {}

// MarshalJSON encodes a Choice answer using the TypeSafe wire names.
func (a ChoiceAnswer) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type          QuestionType       `json:"type"`
		Choice        string             `json:"choice"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	}{Type: QuestionTypeChoice, Choice: a.Choice, Confidence: a.Confidence, Probabilities: a.Probabilities})
}

// ScoreAnswer contains the expected provider score and its evidence. Legend and
// probability keys are converted from JSON object keys to integer levels.
type ScoreAnswer struct {
	Score         float64
	Confidence    float64
	Legend        map[int]any
	Probabilities map[int]float64
}

func (ScoreAnswer) AnswerType() QuestionType { return QuestionTypeScore }
func (ScoreAnswer) answer()                  {}

// MarshalJSON encodes a Score answer using the TypeSafe wire names.
func (a ScoreAnswer) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type          QuestionType    `json:"type"`
		Score         float64         `json:"score"`
		Confidence    float64         `json:"confidence"`
		Legend        map[int]any     `json:"legend"`
		Probabilities map[int]float64 `json:"probabilities"`
	}{
		Type: QuestionTypeScore, Score: a.Score, Confidence: a.Confidence,
		Legend: a.Legend, Probabilities: a.Probabilities,
	})
}

// UnknownAnswer preserves an answer kind introduced after this client version.
// Callers that require a known answer should reject it explicitly.
type UnknownAnswer struct {
	Type QuestionType
	Raw  json.RawMessage
}

func (a UnknownAnswer) AnswerType() QuestionType { return a.Type }
func (UnknownAnswer) answer()                    {}

// MarshalJSON returns the original unknown answer payload.
func (a UnknownAnswer) MarshalJSON() ([]byte, error) {
	return bytes.Clone(a.Raw), nil
}

// Usage reports token counts for one System One request.
type Usage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// SystemOneResponse is a validated response from POST /v1/systemone.
type SystemOneResponse struct {
	Model     string            `json:"model"`
	Answers   map[string]Answer `json:"answers"`
	Usage     Usage             `json:"usage"`
	RequestID string            `json:"-"`
}

// UnmarshalJSON validates required response fields and decodes the tagged
// answer union. Unknown answer types are retained as UnknownAnswer values.
func (r *SystemOneResponse) UnmarshalJSON(data []byte) error {
	var envelope struct {
		Model   *string                    `json:"model"`
		Answers map[string]json.RawMessage `json:"answers"`
		Usage   *struct {
			InputTokens  *int64 `json:"input_tokens"`
			OutputTokens *int64 `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := decodeJSON(data, &envelope); err != nil {
		return responseValidationError("", "response is not valid JSON: "+err.Error())
	}
	if envelope.Model == nil || strings.TrimSpace(*envelope.Model) == "" {
		return responseValidationError("model", "missing or empty string")
	}
	if len(envelope.Answers) == 0 {
		return responseValidationError("answers", "missing or empty object")
	}
	if envelope.Usage == nil {
		return responseValidationError("usage", "missing object")
	}
	if envelope.Usage.InputTokens == nil || *envelope.Usage.InputTokens < 0 {
		return responseValidationError("usage.input_tokens", "missing or negative integer")
	}
	if envelope.Usage.OutputTokens == nil || *envelope.Usage.OutputTokens < 0 {
		return responseValidationError("usage.output_tokens", "missing or negative integer")
	}

	answers := make(map[string]Answer, len(envelope.Answers))
	for _, name := range sortedKeys(envelope.Answers) {
		answer, err := decodeAnswer(name, envelope.Answers[name])
		if err != nil {
			return err
		}
		answers[name] = answer
	}

	r.Model = *envelope.Model
	r.Answers = answers
	r.Usage = Usage{InputTokens: *envelope.Usage.InputTokens, OutputTokens: *envelope.Usage.OutputTokens}
	r.RequestID = ""
	return nil
}

// MarshalJSON encodes the response's tagged answers and omits transport-only
// metadata such as RequestID.
func (r SystemOneResponse) MarshalJSON() ([]byte, error) {
	type wire struct {
		Model   string            `json:"model"`
		Answers map[string]Answer `json:"answers"`
		Usage   Usage             `json:"usage"`
	}
	return json.Marshal(wire{Model: r.Model, Answers: r.Answers, Usage: r.Usage})
}

func decodeAnswer(name string, raw json.RawMessage) (Answer, error) {
	var header struct {
		Type string `json:"type"`
	}
	if err := decodeJSON(raw, &header); err != nil {
		return nil, responseValidationError("answers."+name, "must be an object")
	}
	if header.Type == "" {
		return nil, responseValidationError("answers."+name+".type", "missing or empty string")
	}

	switch QuestionType(header.Type) {
	case QuestionTypeNoul:
		var value struct {
			Noul *float64 `json:"noul"`
		}
		if err := decodeJSON(raw, &value); err != nil || value.Noul == nil || !isFinite(*value.Noul) {
			return nil, responseValidationError("answers."+name+".noul", "missing or non-finite number")
		}
		return NoulAnswer{Probability: *value.Noul}, nil
	case QuestionTypeChoice:
		var value struct {
			Choice        *string            `json:"choice"`
			Confidence    *float64           `json:"confidence"`
			Probabilities map[string]float64 `json:"probabilities"`
		}
		if err := decodeJSON(raw, &value); err != nil {
			return nil, responseValidationError("answers."+name, "invalid choice answer")
		}
		if value.Choice == nil {
			return nil, responseValidationError("answers."+name+".choice", "missing string")
		}
		if value.Confidence == nil || !isFinite(*value.Confidence) {
			return nil, responseValidationError("answers."+name+".confidence", "missing or non-finite number")
		}
		if value.Probabilities == nil {
			return nil, responseValidationError("answers."+name+".probabilities", "missing object")
		}
		for label, probability := range value.Probabilities {
			if !isFinite(probability) {
				return nil, responseValidationError("answers."+name+".probabilities."+label, "non-finite number")
			}
		}
		return ChoiceAnswer{Choice: *value.Choice, Confidence: *value.Confidence, Probabilities: value.Probabilities}, nil
	case QuestionTypeScore:
		var value struct {
			Score         *float64           `json:"score"`
			Confidence    *float64           `json:"confidence"`
			Legend        map[string]any     `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
		}
		if err := decodeJSON(raw, &value); err != nil {
			return nil, responseValidationError("answers."+name, "invalid score answer")
		}
		if value.Score == nil || !isFinite(*value.Score) {
			return nil, responseValidationError("answers."+name+".score", "missing or non-finite number")
		}
		if value.Confidence == nil || !isFinite(*value.Confidence) {
			return nil, responseValidationError("answers."+name+".confidence", "missing or non-finite number")
		}
		if value.Legend == nil {
			return nil, responseValidationError("answers."+name+".legend", "missing object")
		}
		if value.Probabilities == nil {
			return nil, responseValidationError("answers."+name+".probabilities", "missing object")
		}
		legend, err := integerKeyMap(name, "legend", value.Legend)
		if err != nil {
			return nil, err
		}
		probabilities, err := integerProbabilityMap(name, value.Probabilities)
		if err != nil {
			return nil, err
		}
		return ScoreAnswer{
			Score: *value.Score, Confidence: *value.Confidence,
			Legend: legend, Probabilities: probabilities,
		}, nil
	default:
		return UnknownAnswer{Type: QuestionType(header.Type), Raw: bytes.Clone(raw)}, nil
	}
}

func validateResponseForRequest(response *SystemOneResponse, request SystemOneRequest) *ResponseValidationError {
	if len(response.Answers) != len(request.Questions) {
		return responseValidationError("answers", "answer names do not match request questions")
	}
	for _, name := range sortedKeys(request.Questions) {
		question := request.Questions[name]
		answer, found := response.Answers[name]
		if !found {
			return responseValidationError("answers."+name, "missing answer")
		}
		if answer == nil || isNilInterface(answer) {
			return responseValidationError("answers."+name, "nil answer")
		}
		if answer.AnswerType() != question.questionType() {
			return responseValidationError("answers."+name+".type", fmt.Sprintf("got %q, want %q", answer.AnswerType(), question.questionType()))
		}
		switch question.questionType() {
		case QuestionTypeNoul:
			answer := answer.(NoulAnswer)
			if err := validateUnitInterval(answer.Probability); err != nil {
				return responseValidationError("answers."+name+".noul", err.Error())
			}
		case QuestionTypeChoice:
			criteria := choiceCriteria(question)
			answer := answer.(ChoiceAnswer)
			if err := validateUnitInterval(answer.Confidence); err != nil {
				return responseValidationError("answers."+name+".confidence", err.Error())
			}
			if _, found := criteria[answer.Choice]; !found {
				return responseValidationError("answers."+name+".choice", "selected label is not in the requested criteria")
			}
			if err := validateChoiceDistribution(answer.Probabilities, criteria); err != nil {
				return responseValidationError("answers."+name+".probabilities", err.Error())
			}
		case QuestionTypeScore:
			levels := len(scoreCriteria(question))
			answer := answer.(ScoreAnswer)
			if err := validateUnitInterval(answer.Confidence); err != nil {
				return responseValidationError("answers."+name+".confidence", err.Error())
			}
			if answer.Score < 0 || answer.Score > float64(levels-1) {
				return responseValidationError("answers."+name+".score", "outside the requested rubric index range")
			}
			if err := validateScoreDistribution(answer.Probabilities, levels); err != nil {
				return responseValidationError("answers."+name+".probabilities", err.Error())
			}
			if err := validateScoreLegend(answer.Legend, levels); err != nil {
				return responseValidationError("answers."+name+".legend", err.Error())
			}
		}
	}
	return nil
}

func choiceCriteria(question Question) map[string]any {
	switch question := question.(type) {
	case ChoiceQuestion:
		return question.Criteria
	case *ChoiceQuestion:
		return question.Criteria
	default:
		return nil
	}
}

func scoreCriteria(question Question) []any {
	switch question := question.(type) {
	case ScoreQuestion:
		return question.Criteria
	case *ScoreQuestion:
		return question.Criteria
	default:
		return nil
	}
}

func validateUnitInterval(value float64) error {
	if !isFinite(value) || value < 0 || value > 1 {
		return fmt.Errorf("must be a finite number in [0, 1]")
	}
	return nil
}

func validateChoiceDistribution(probabilities map[string]float64, criteria map[string]any) error {
	if len(probabilities) != len(criteria) {
		return fmt.Errorf("labels do not match the requested criteria")
	}
	var mass float64
	for _, label := range sortedKeys(criteria) {
		probability, found := probabilities[label]
		if !found {
			return fmt.Errorf("missing label %q", label)
		}
		if err := validateUnitInterval(probability); err != nil {
			return fmt.Errorf("label %q %w", label, err)
		}
		mass += probability
	}
	return validateDistributionMass(mass)
}

func validateScoreDistribution(probabilities map[int]float64, levels int) error {
	if len(probabilities) != levels {
		return fmt.Errorf("levels do not match the requested criteria")
	}
	var mass float64
	for level := range levels {
		probability, found := probabilities[level]
		if !found {
			return fmt.Errorf("missing level %d", level)
		}
		if err := validateUnitInterval(probability); err != nil {
			return fmt.Errorf("level %d %w", level, err)
		}
		mass += probability
	}
	return validateDistributionMass(mass)
}

func validateScoreLegend(legend map[int]any, levels int) error {
	if len(legend) != levels {
		return fmt.Errorf("levels do not match the requested criteria")
	}
	for level := range levels {
		if _, found := legend[level]; !found {
			return fmt.Errorf("missing level %d", level)
		}
	}
	return nil
}

func validateDistributionMass(mass float64) error {
	const tolerance = 0.02
	if !isFinite(mass) || mass <= 0 || math.Abs(mass-1) > tolerance {
		return fmt.Errorf("probability mass must sum to 1 within %.2f (got %g)", tolerance, mass)
	}
	return nil
}

func integerKeyMap(name, field string, source map[string]any) (map[int]any, error) {
	result := make(map[int]any, len(source))
	for _, key := range sortedKeys(source) {
		value := source[key]
		index, err := strconv.Atoi(key)
		if err != nil || index < 0 {
			return nil, responseValidationError("answers."+name+"."+field+"."+key, "key must be a non-negative integer")
		}
		result[index] = value
	}
	return result, nil
}

func integerProbabilityMap(name string, source map[string]float64) (map[int]float64, error) {
	result := make(map[int]float64, len(source))
	for _, key := range sortedKeys(source) {
		value := source[key]
		index, err := strconv.Atoi(key)
		if err != nil || index < 0 {
			return nil, responseValidationError("answers."+name+".probabilities."+key, "key must be a non-negative integer")
		}
		if !isFinite(value) {
			return nil, responseValidationError("answers."+name+".probabilities."+key, "non-finite number")
		}
		result[index] = value
	}
	return result, nil
}

func validateJSONContent(value any) error {
	if value == nil {
		return fmt.Errorf("must be a string, JSON object, or JSON array")
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("is not JSON encodable: %w", err)
	}
	var decoded any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(&decoded); err != nil {
		return fmt.Errorf("is not valid JSON: %w", err)
	}
	switch decoded.(type) {
	case string, []any, map[string]any:
		return nil
	default:
		return fmt.Errorf("must be a string, JSON object, or JSON array")
	}
}

func decodeJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	return decoder.Decode(target)
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
