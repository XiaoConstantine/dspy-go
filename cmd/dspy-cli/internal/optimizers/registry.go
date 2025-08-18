package optimizers

import "fmt"

type OptimizerInfo struct {
	Name          string
	Description   string
	UseCase       string
	Complexity    string
	ComputeCost   string
	Convergence   string
	BestFor       []string
	Example       string
}

var Registry = map[string]OptimizerInfo{
	"bootstrap": {
		Name:        "BootstrapFewShot ✅",
		Description: "Automatically selects high-quality examples for few-shot learning",
		UseCase:     "Quick improvements with minimal computational cost",
		Complexity:  "Low",
		ComputeCost: "Low (fast convergence)",
		Convergence: "Fast",
		BestFor: []string{
			"Getting started with optimization",
			"Simple tasks with clear patterns",
			"When you need quick results",
			"Limited computational budget",
		},
		Example: "Simple Q&A tasks, basic classification",
	},
	"mipro": {
		Name:        "MIPRO ✅",
		Description: "Multi-step Interactive Prompt Optimization using Tree-structured Parzen Estimator",
		UseCase:     "Systematic optimization with good balance of cost and performance",
		Complexity:  "Medium",
		ComputeCost: "Medium (systematic search)",
		Convergence: "Moderate",
		BestFor: []string{
			"Complex reasoning tasks",
			"When you need systematic optimization",
			"Multi-step problems",
			"Balanced cost/performance requirements",
		},
		Example: "Math problem solving, multi-hop reasoning",
	},
	"simba": {
		Name:        "SIMBA ✅",
		Description: "Stochastic Introspective Mini-Batch Ascent with self-improving capabilities",
		UseCase:     "Advanced optimization with introspective learning",
		Complexity:  "High",
		ComputeCost: "Medium-High (introspective analysis)",
		Convergence: "Adaptive",
		BestFor: []string{
			"Complex tasks requiring self-reflection",
			"Adaptive learning scenarios",
			"When optimization needs to improve itself",
			"Advanced reasoning patterns",
		},
		Example: "Complex reasoning, adaptive problem solving",
	},
	"gepa": {
		Name:        "GEPA ✅",
		Description: "Generative Evolutionary Prompt Adaptation with multi-objective Pareto optimization",
		UseCase:     "State-of-the-art optimization for the most demanding tasks",
		Complexity:  "Very High",
		ComputeCost: "High (evolutionary + Pareto optimization)",
		Convergence: "Sophisticated",
		BestFor: []string{
			"Cutting-edge performance requirements",
			"Multi-objective optimization",
			"Complex semantic understanding",
			"Research and experimentation",
		},
		Example: "Advanced NLP tasks, research-grade optimization",
	},
	"copro": {
		Name:        "COPRO ✅",
		Description: "Collaborative Prompt Optimization for multi-module scenarios",
		UseCase:     "Optimizing multiple modules working together",
		Complexity:  "Medium-High",
		ComputeCost: "Medium (collaborative approach)",
		Convergence: "Coordinated",
		BestFor: []string{
			"Multi-module systems",
			"Collaborative optimization",
			"Complex pipeline optimization",
			"Coordinated reasoning tasks",
		},
		Example: "Multi-step workflows, complex pipelines",
	},
}

func GetOptimizer(name string) (OptimizerInfo, error) {
	if info, exists := Registry[name]; exists {
		return info, nil
	}
	return OptimizerInfo{}, fmt.Errorf("optimizer '%s' not found", name)
}

func ListAll() []string {
	var names []string
	for name := range Registry {
		names = append(names, name)
	}
	return names
}

func GetRecommendation(useCase string) []string {
	recommendations := []string{}

	switch useCase {
	case "beginner", "simple", "quick":
		recommendations = append(recommendations, "bootstrap")
	case "balanced", "moderate", "systematic":
		recommendations = append(recommendations, "bootstrap", "simba")
	case "advanced", "complex", "research":
		recommendations = append(recommendations, "simba", "copro")
	case "multi-module", "pipeline", "collaborative":
		recommendations = append(recommendations, "copro")
	default:
		recommendations = append(recommendations, "bootstrap", "simba")
	}

	return recommendations
}
