package runner

import "testing"

func TestAnswersMatch(t *testing.T) {
	cases := []struct {
		expected, predicted string
		want                bool
	}{
		{"Paris", "Paris", true},
		{"Paris", "Lyon", false},
		{"70000", "Josh made a profit of $70,000.", true},
		{"18", "Janet makes $18 every day.", true},
		{"3", "- 3 bolts of fiber are needed for one robe.", true},
		{"540", "He runs 3 sprints of 60 meters, so 540 meters.", true},
		{"540", "He runs 540 meters, not 600.", false},
		{"20", "No number here.", false},
	}
	for _, c := range cases {
		if got := answersMatch(c.expected, c.predicted); got != c.want {
			t.Errorf("answersMatch(%q, %q) = %v, want %v", c.expected, c.predicted, got, c.want)
		}
	}
}
