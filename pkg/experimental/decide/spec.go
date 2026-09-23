package decide

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

const probabilityMassTolerance = 0.02

type outputKind string

const (
	outputNoul   outputKind = "noul"
	outputScore  outputKind = "score"
	outputChoice outputKind = "choice"
)

// Output declares one closed-set output for New. Implementations are sealed so
// an output always has a validated answer-space representation.
type Output interface {
	outputName() string
	kind() outputKind
	validate() error
	clone() Output
	question(instructions any) typesafe.Question
	provenance() outputProvenance
	decode(typesafe.Answer) (Decision, any, error)
	parameter() any
	setParameter(any) error
}

type noulOutput struct {
	name      string
	threshold float64
}

// Noul declares a Boolean output with an initial threshold of 0.5.
func Noul(name string) Output {
	return &noulOutput{name: name, threshold: 0.5}
}

func (o *noulOutput) outputName() string { return o.name }
func (*noulOutput) kind() outputKind     { return outputNoul }
func (o *noulOutput) clone() Output      { copy := *o; return &copy }

func (o *noulOutput) validate() error {
	if err := validateOutputName(o.name); err != nil {
		return err
	}
	return validateThreshold(o.threshold)
}

func (o *noulOutput) question(instructions any) typesafe.Question {
	return typesafe.NoulQuestion{Instructions: instructions}
}

func (o *noulOutput) provenance() outputProvenance {
	return outputProvenance{Name: o.name, Kind: outputNoul}
}

func (o *noulOutput) decode(raw typesafe.Answer) (Decision, any, error) {
	answer, ok := asNoulAnswer(raw)
	if !ok {
		return nil, nil, fmt.Errorf("output %q: expected noul answer, got %q", o.name, answerType(raw))
	}
	if err := validateProbability(answer.Probability); err != nil {
		return nil, nil, fmt.Errorf("output %q noul probability: %w", o.name, err)
	}
	denominator := math.Max(o.threshold, 1-o.threshold)
	decision := NoulDecision{
		Value:       answer.Probability >= o.threshold,
		Probability: answer.Probability,
		Threshold:   o.threshold,
		Confidence:  math.Abs(answer.Probability-o.threshold) / denominator,
	}
	return decision, decision.Value, nil
}

func (o *noulOutput) parameter() any { return o.threshold }

func (o *noulOutput) setParameter(value any) error {
	threshold, ok := value.(float64)
	if !ok {
		return fmt.Errorf("threshold must be a number")
	}
	if err := validateThreshold(threshold); err != nil {
		return err
	}
	o.threshold = threshold
	return nil
}

// ScoreLevel pairs a local numeric anchor with the description sent to the
// provider for that zero-based rubric level.
type ScoreLevel struct {
	value       float64
	description string
}

// Level constructs one Score rubric level.
func Level(value float64, description string) ScoreLevel {
	return ScoreLevel{value: value, description: description}
}

type scoreOutput struct {
	name    string
	levels  []ScoreLevel
	anchors []float64
}

// Score declares an ordinal output. At least two and at most ten finite,
// strictly increasing anchors are required.
func Score(name string, levels ...ScoreLevel) Output {
	cloned := append([]ScoreLevel(nil), levels...)
	anchors := make([]float64, len(cloned))
	for i, level := range cloned {
		anchors[i] = level.value
	}
	return &scoreOutput{name: name, levels: cloned, anchors: anchors}
}

func (o *scoreOutput) outputName() string { return o.name }
func (*scoreOutput) kind() outputKind     { return outputScore }

func (o *scoreOutput) clone() Output {
	return &scoreOutput{
		name:    o.name,
		levels:  append([]ScoreLevel(nil), o.levels...),
		anchors: append([]float64(nil), o.anchors...),
	}
}

func (o *scoreOutput) validate() error {
	if err := validateOutputName(o.name); err != nil {
		return err
	}
	if len(o.levels) < 2 || len(o.levels) > 10 {
		return fmt.Errorf("output %q: Score requires between 2 and 10 levels", o.name)
	}
	for i, level := range o.levels {
		if !isFinite(level.value) {
			return fmt.Errorf("output %q: Score anchor %d must be finite", o.name, i)
		}
		if i > 0 && o.levels[i-1].value >= level.value {
			return fmt.Errorf("output %q: Score anchors must be strictly increasing", o.name)
		}
	}
	return validateScoreAnchors(o.anchors, o.levels)
}

func (o *scoreOutput) question(instructions any) typesafe.Question {
	criteria := make([]any, len(o.levels))
	for i, level := range o.levels {
		criteria[i] = level.description
	}
	return typesafe.ScoreQuestion{Instructions: instructions, Criteria: criteria}
}

func (o *scoreOutput) provenance() outputProvenance {
	levels := make([]scoreLevelProvenance, len(o.levels))
	for i, level := range o.levels {
		levels[i] = scoreLevelProvenance{Value: level.value, Description: level.description}
	}
	return outputProvenance{Name: o.name, Kind: outputScore, ScoreLevels: levels}
}

func (o *scoreOutput) decode(raw typesafe.Answer) (Decision, any, error) {
	answer, ok := asScoreAnswer(raw)
	if !ok {
		return nil, nil, fmt.Errorf("output %q: expected score answer, got %q", o.name, answerType(raw))
	}
	if !isFinite(answer.Score) || answer.Score < 0 || answer.Score > float64(len(o.levels)-1) {
		return nil, nil, fmt.Errorf("output %q: provider score must be within the rubric index range", o.name)
	}
	if err := validateProbability(answer.Confidence); err != nil {
		return nil, nil, fmt.Errorf("output %q provider confidence: %w", o.name, err)
	}
	if err := validateIndexedDistribution(answer.Probabilities, len(o.levels)); err != nil {
		return nil, nil, fmt.Errorf("output %q: %w", o.name, err)
	}

	var weighted, mass float64
	for i, anchor := range o.anchors {
		probability := answer.Probabilities[i]
		weighted += anchor * probability
		mass += probability
	}
	value := weighted / mass
	decision := ScoreDecision{
		Value:              value,
		ProviderScore:      answer.Score,
		ProviderConfidence: answer.Confidence,
		probabilities:      cloneIntMap(answer.Probabilities),
		anchors:            append([]float64(nil), o.anchors...),
	}
	return decision, value, nil
}

func (o *scoreOutput) parameter() any {
	return append([]float64(nil), o.anchors...)
}

func (o *scoreOutput) setParameter(value any) error {
	anchors, ok := value.([]float64)
	if !ok {
		return fmt.Errorf("Score anchors must be a numeric list")
	}
	if err := validateScoreAnchors(anchors, o.levels); err != nil {
		return err
	}
	o.anchors = append([]float64(nil), anchors...)
	return nil
}

// ChoiceOption pairs a typed application value with its provider description.
type ChoiceOption[T any] struct {
	value       T
	description string
}

// Option constructs one typed Choice option.
func Option[T any](value T, description string) ChoiceOption[T] {
	return ChoiceOption[T]{value: value, description: description}
}

type labeledChoice[T any] struct {
	label       string
	value       T
	description string
}

type choiceOutput[T any] struct {
	name        string
	options     []labeledChoice[T]
	multipliers map[string]float64
	buildErr    error
}

// Choice declares a typed closed set. String, Boolean, signed integer, and
// unsigned integer values (including named aliases) are supported.
func Choice[T any](name string, options ...ChoiceOption[T]) Output {
	result := &choiceOutput[T]{name: name, multipliers: make(map[string]float64, len(options))}
	for _, option := range options {
		label, err := choiceLabel(option.value)
		if err != nil && result.buildErr == nil {
			result.buildErr = err
		}
		result.options = append(result.options, labeledChoice[T]{
			label: label, value: option.value, description: option.description,
		})
		result.multipliers[label] = 1
	}
	return result
}

func (o *choiceOutput[T]) outputName() string { return o.name }
func (*choiceOutput[T]) kind() outputKind     { return outputChoice }

func (o *choiceOutput[T]) clone() Output {
	return &choiceOutput[T]{
		name:        o.name,
		options:     append([]labeledChoice[T](nil), o.options...),
		multipliers: cloneStringMap(o.multipliers),
		buildErr:    o.buildErr,
	}
}

func (o *choiceOutput[T]) validate() error {
	if err := validateOutputName(o.name); err != nil {
		return err
	}
	if o.buildErr != nil {
		return fmt.Errorf("output %q: %w", o.name, o.buildErr)
	}
	if len(o.options) == 0 || len(o.options) > 255 {
		return fmt.Errorf("output %q: Choice requires between 1 and 255 options", o.name)
	}
	labels := make(map[string]struct{}, len(o.options))
	for _, option := range o.options {
		if option.label == "" {
			return fmt.Errorf("output %q: Choice labels must not be empty", o.name)
		}
		if _, duplicate := labels[option.label]; duplicate {
			return fmt.Errorf("output %q: Choice values have colliding provider label %q", o.name, option.label)
		}
		labels[option.label] = struct{}{}
	}
	return validateChoiceMultipliers(o.multipliers, labels)
}

func (o *choiceOutput[T]) question(instructions any) typesafe.Question {
	criteria := make(map[string]any, len(o.options))
	for _, option := range o.options {
		if option.description == "" {
			criteria[option.label] = nil
		} else {
			criteria[option.label] = option.description
		}
	}
	return typesafe.ChoiceQuestion{Instructions: instructions, Criteria: criteria}
}

func (o *choiceOutput[T]) provenance() outputProvenance {
	options := make([]choiceOptionProvenance, len(o.options))
	for i, option := range o.options {
		valueType := reflect.TypeOf(option.value)
		typeName := "<nil>"
		if valueType != nil {
			typeName = valueType.PkgPath() + ":" + valueType.String()
		}
		options[i] = choiceOptionProvenance{
			Label:       option.label,
			Description: option.description,
			ValueType:   typeName,
		}
	}
	return outputProvenance{Name: o.name, Kind: outputChoice, ChoiceOptions: options}
}

func (o *choiceOutput[T]) decode(raw typesafe.Answer) (Decision, any, error) {
	answer, ok := asChoiceAnswer(raw)
	if !ok {
		return nil, nil, fmt.Errorf("output %q: expected choice answer, got %q", o.name, answerType(raw))
	}
	if err := validateProbability(answer.Confidence); err != nil {
		return nil, nil, fmt.Errorf("output %q provider confidence: %w", o.name, err)
	}
	labels := make([]string, len(o.options))
	byLabel := make(map[string]T, len(o.options))
	for i, option := range o.options {
		labels[i] = option.label
		byLabel[option.label] = option.value
	}
	providerValue, found := byLabel[answer.Choice]
	if !found {
		return nil, nil, fmt.Errorf("output %q: provider selected unknown label %q", o.name, answer.Choice)
	}
	if err := validateLabeledDistribution(answer.Probabilities, labels); err != nil {
		return nil, nil, fmt.Errorf("output %q: %w", o.name, err)
	}

	selected := answer.Choice
	weighted := false
	for _, multiplier := range o.multipliers {
		if multiplier != 1 {
			weighted = true
			break
		}
	}
	if weighted {
		bestScore := -1.0
		for _, option := range o.options {
			score := answer.Probabilities[option.label] * o.multipliers[option.label]
			if score > bestScore || score == bestScore && option.label == answer.Choice {
				bestScore = score
				selected = option.label
			}
		}
		if bestScore <= 0 {
			return nil, nil, fmt.Errorf("output %q: Choice multipliers leave no positive probability mass", o.name)
		}
	}

	decision := ChoiceDecision[T]{
		Value:              byLabel[selected],
		ProviderValue:      providerValue,
		LocalLabel:         selected,
		ProviderLabel:      answer.Choice,
		ProviderConfidence: answer.Confidence,
		probabilities:      cloneStringMap(answer.Probabilities),
		multipliers:        cloneStringMap(o.multipliers),
	}
	return decision, decision.Value, nil
}

func (o *choiceOutput[T]) parameter() any {
	return cloneStringMap(o.multipliers)
}

func (o *choiceOutput[T]) setParameter(value any) error {
	multipliers, ok := value.(map[string]float64)
	if !ok {
		return fmt.Errorf("Choice multipliers must be a string-to-number map")
	}
	labels := make(map[string]struct{}, len(o.options))
	for _, option := range o.options {
		labels[option.label] = struct{}{}
	}
	if err := validateChoiceMultipliers(multipliers, labels); err != nil {
		return err
	}
	effective := make(map[string]float64, len(labels))
	for label := range labels {
		effective[label] = 1
	}
	for label, multiplier := range multipliers {
		effective[label] = multiplier
	}
	o.multipliers = effective
	return nil
}

func validateOutputName(name string) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("decision output name must not be empty")
	}
	return nil
}

func validateThreshold(value float64) error {
	if !isFinite(value) || value < 0 || value > 1 {
		return fmt.Errorf("threshold must be a finite number in [0, 1]")
	}
	return nil
}

func validateScoreAnchors(anchors []float64, levels []ScoreLevel) error {
	if len(anchors) != len(levels) {
		return fmt.Errorf("Score anchors must contain exactly %d values", len(levels))
	}
	for i, anchor := range anchors {
		if !isFinite(anchor) {
			return fmt.Errorf("Score anchor %d must be finite", i)
		}
		if anchor < levels[0].value || anchor > levels[len(levels)-1].value {
			return fmt.Errorf("Score anchor %d must remain within the declared range", i)
		}
		if i > 0 && anchors[i-1] >= anchor {
			return fmt.Errorf("Score anchors must be strictly increasing")
		}
	}
	return nil
}

func validateChoiceMultipliers(multipliers map[string]float64, labels map[string]struct{}) error {
	for label, multiplier := range multipliers {
		if _, found := labels[label]; !found {
			return fmt.Errorf("Choice multiplier has unknown label %q", label)
		}
		if !isFinite(multiplier) || multiplier < 0 {
			return fmt.Errorf("Choice multiplier for %q must be finite and non-negative", label)
		}
	}
	for label := range labels {
		if multiplier, found := multipliers[label]; !found || multiplier > 0 {
			return nil
		}
	}
	return fmt.Errorf("Choice multipliers must leave at least one option enabled")
}

func validateProbability(value float64) error {
	if !isFinite(value) || value < 0 || value > 1 {
		return fmt.Errorf("must be a finite number in [0, 1]")
	}
	return nil
}

func validateIndexedDistribution(probabilities map[int]float64, count int) error {
	if len(probabilities) != count {
		return fmt.Errorf("Score distribution must contain exactly %d levels", count)
	}
	labels := make([]int, count)
	for i := range count {
		labels[i] = i
	}
	var mass float64
	for _, index := range labels {
		probability, found := probabilities[index]
		if !found {
			return fmt.Errorf("Score distribution is missing level %d", index)
		}
		if err := validateProbability(probability); err != nil {
			return fmt.Errorf("Score probability at level %d %w", index, err)
		}
		mass += probability
	}
	return validateProbabilityMass(mass)
}

func validateLabeledDistribution(probabilities map[string]float64, labels []string) error {
	if len(probabilities) != len(labels) {
		return fmt.Errorf("Choice distribution labels do not match the declared options")
	}
	var mass float64
	for _, label := range labels {
		probability, found := probabilities[label]
		if !found {
			return fmt.Errorf("Choice distribution is missing label %q", label)
		}
		if err := validateProbability(probability); err != nil {
			return fmt.Errorf("Choice probability for %q %w", label, err)
		}
		mass += probability
	}
	return validateProbabilityMass(mass)
}

func validateProbabilityMass(mass float64) error {
	if !isFinite(mass) || mass <= 0 || math.Abs(mass-1) > probabilityMassTolerance {
		return fmt.Errorf("probability mass must sum to 1 within %.2f (got %g)", probabilityMassTolerance, mass)
	}
	return nil
}

func choiceLabel(value any) (string, error) {
	if value == nil {
		return "", fmt.Errorf("Choice values must be strings, Booleans, or integers")
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.String:
		return reflected.String(), nil
	case reflect.Bool:
		return strconv.FormatBool(reflected.Bool()), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(reflected.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(reflected.Uint(), 10), nil
	default:
		return "", fmt.Errorf("Choice values must be strings, Booleans, or integers")
	}
}

func asNoulAnswer(raw typesafe.Answer) (typesafe.NoulAnswer, bool) {
	switch answer := raw.(type) {
	case typesafe.NoulAnswer:
		return answer, true
	case *typesafe.NoulAnswer:
		if answer != nil {
			return *answer, true
		}
	}
	return typesafe.NoulAnswer{}, false
}

func asScoreAnswer(raw typesafe.Answer) (typesafe.ScoreAnswer, bool) {
	switch answer := raw.(type) {
	case typesafe.ScoreAnswer:
		return answer, true
	case *typesafe.ScoreAnswer:
		if answer != nil {
			return *answer, true
		}
	}
	return typesafe.ScoreAnswer{}, false
}

func asChoiceAnswer(raw typesafe.Answer) (typesafe.ChoiceAnswer, bool) {
	switch answer := raw.(type) {
	case typesafe.ChoiceAnswer:
		return answer, true
	case *typesafe.ChoiceAnswer:
		if answer != nil {
			return *answer, true
		}
	}
	return typesafe.ChoiceAnswer{}, false
}

func answerType(raw typesafe.Answer) typesafe.QuestionType {
	if raw == nil {
		return "<nil>"
	}
	value := reflect.ValueOf(raw)
	if value.Kind() == reflect.Pointer && value.IsNil() {
		return "<nil>"
	}
	return raw.AnswerType()
}

func cloneStringMap(source map[string]float64) map[string]float64 {
	result := make(map[string]float64, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func cloneIntMap(source map[int]float64) map[int]float64 {
	result := make(map[int]float64, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
