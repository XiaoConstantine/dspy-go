package samples

import "github.com/XiaoConstantine/dspy-go/pkg/core"

// SampleDataset represents a small dataset for quick experimentation
type SampleDataset struct {
	Name        string
	Description string
	Examples    []core.Example
}

// GetGSM8KSample returns a small subset of GSM8K for quick testing
func GetGSM8KSample() SampleDataset {
	return SampleDataset{
		Name:        "GSM8K Sample",
		Description: "Grade School Math 8K - Sample math word problems",
		Examples: []core.Example{
			{
				Inputs: map[string]interface{}{
					"question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
				},
				Outputs: map[string]interface{}{
					"answer": "18",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber are needed?",
				},
				Outputs: map[string]interface{}{
					"answer": "3",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
				},
				Outputs: map[string]interface{}{
					"answer": "70000",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
				},
				Outputs: map[string]interface{}{
					"answer": "540",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms, and vegetables. She gives the chickens their feed in three separate meals. If she has 20 chickens, how many cups of feed does she need in the morning?",
				},
				Outputs: map[string]interface{}{
					"answer": "20",
				},
			},
		},
	}
}

// GetHotPotQASample returns a small subset of HotPotQA for quick testing
func GetHotPotQASample() SampleDataset {
	return SampleDataset{
		Name:        "HotPotQA Sample",
		Description: "Multi-hop reasoning questions requiring multiple steps",
		Examples: []core.Example{
			{
				Inputs: map[string]interface{}{
					"question": "Which magazine was started first Arthur's Magazine or First for Women?",
				},
				Outputs: map[string]interface{}{
					"answer": "Arthur's Magazine",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "The Oberoi family is part of a hotel company that has a head office in which city?",
				},
				Outputs: map[string]interface{}{
					"answer": "Delhi",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was held?",
				},
				Outputs: map[string]interface{}{
					"answer": "6.213 km",
				},
			},
		},
	}
}

// GetQASample returns a simple Q&A dataset for basic testing
func GetQASample() SampleDataset {
	return SampleDataset{
		Name:        "Simple Q&A",
		Description: "Basic question-answering for testing simple patterns",
		Examples: []core.Example{
			{
				Inputs: map[string]interface{}{
					"question": "What is the capital of France?",
				},
				Outputs: map[string]interface{}{
					"answer": "Paris",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "What color do you get when you mix red and blue?",
				},
				Outputs: map[string]interface{}{
					"answer": "Purple",
				},
			},
			{
				Inputs: map[string]interface{}{
					"question": "How many sides does a triangle have?",
				},
				Outputs: map[string]interface{}{
					"answer": "3",
				},
			},
		},
	}
}

// GetSampleDataset returns a dataset by name
func GetSampleDataset(name string) (SampleDataset, bool) {
	switch name {
	case "gsm8k":
		return GetGSM8KSample(), true
	case "hotpotqa":
		return GetHotPotQASample(), true
	case "qa", "simple":
		return GetQASample(), true
	default:
		return SampleDataset{}, false
	}
}

// ListAvailableDatasets returns all available sample datasets
func ListAvailableDatasets() []string {
	return []string{"gsm8k", "hotpotqa", "qa"}
}
