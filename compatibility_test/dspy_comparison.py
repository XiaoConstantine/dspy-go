#!/usr/bin/env python3
# /// script
# dependencies = [
#     "dspy-ai>=2.4.0",
#     "google-generativeai>=0.3.0",
#     "numpy>=1.21.0",
# ]
# ///
"""
DSPy Python Reference Implementation for Compatibility Testing
This script provides a reference implementation using the official DSPy Python package
for side-by-side comparison with dspy-go optimizers.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2, SIMBA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonMetrics:
    """Metrics collection for comparison purposes"""
    
    def __init__(self):
        self.scores = []
        self.execution_times = []
        self.token_usage = []
        
    def add_score(self, score: float):
        self.scores.append(score)
        
    def add_execution_time(self, time: float):
        self.execution_times.append(time)
        
    def add_token_usage(self, tokens: int):
        self.token_usage.append(tokens)
        
    def get_average_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
        
    def get_total_tokens(self) -> int:
        return sum(self.token_usage)
        
    def get_total_time(self) -> float:
        return sum(self.execution_times)


class SimpleQA(dspy.Signature):
    """Simple Question Answering signature for testing"""
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer to the question")


class BasicProgram(dspy.Module):
    """Basic program for testing optimizer compatibility"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleQA)
        
    def forward(self, question: str) -> str:
        return self.predictor(question=question).answer


class OptimizerComparison:
    """Main comparison class for testing optimizer compatibility"""
    
    def __init__(self, model_name: str = "gemini/gemini-2.0-flash"):
        self.model_name = model_name
        # Disable retries to reduce API calls when service is overloaded
        self.lm = dspy.LM(model=model_name, max_tokens=8192, max_retries=0)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize metrics
        self.bootstrap_metrics = ComparisonMetrics()
        self.mipro_metrics = ComparisonMetrics()
        self.simba_metrics = ComparisonMetrics()
        
    def create_sample_dataset(self, size: int = 20) -> List[dspy.Example]:
        """Create a sample dataset for testing"""
        sample_data = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What color is the sky?", "answer": "Blue"},
            {"question": "What is the largest planet?", "answer": "Jupiter"},
            {"question": "What is the smallest prime number?", "answer": "2"},
            {"question": "What is the chemical symbol for water?", "answer": "H2O"},
            {"question": "What is the speed of light?", "answer": "299,792,458 m/s"},
            {"question": "What year did World War II end?", "answer": "1945"},
            {"question": "What is the square root of 16?", "answer": "4"},
            {"question": "What is the boiling point of water?", "answer": "100°C"},
        ]
        
        # Duplicate and extend to reach desired size
        examples = []
        for i in range(size):
            data = sample_data[i % len(sample_data)]
            examples.append(dspy.Example(question=data["question"], answer=data["answer"]).with_inputs("question"))
            
        return examples
        
    def accuracy_metric(self, example: dspy.Example, prediction: str, trace=None) -> float:
        """Simple accuracy metric for evaluation"""
        try:
            # Handle None predictions
            if prediction is None:
                logger.warning("Received None prediction")
                return 0.0
                
            # Handle case where prediction is not a string
            if not isinstance(prediction, str):
                logger.warning(f"Prediction is not a string: {type(prediction)}")
                return 0.0
                
            expected = example.answer.lower().strip()
            predicted = prediction.lower().strip()
            
            # Simple substring matching for demo purposes
            if expected in predicted or predicted in expected:
                return 1.0
            return 0.0
        except Exception as e:
            logger.error(f"Error in accuracy metric: {e}")
            return 0.0
        
    def test_bootstrap_fewshot(self, dataset: List[dspy.Example], 
                              max_bootstrapped_demos: int = 4,
                              max_labeled_demos: int = 4) -> Tuple[dspy.Module, Dict[str, Any]]:
        """Test BootstrapFewShot optimizer"""
        logger.info("Testing BootstrapFewShot optimizer")
        
        program = BasicProgram()
        # Split dataset to match Go's approach: 3/4 for training, 1/4 for validation
        dataset_size = len(dataset)
        train_size = min(dataset_size * 3 // 4, dataset_size - 1)
        if train_size < 1:
            train_size = 1
        trainset = dataset[:train_size]
        valset = dataset[train_size:]
        
        # Create optimizer
        teleprompter = BootstrapFewShot(
            metric=self.accuracy_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos
        )
        
        # Compile program
        import time
        start_time = time.time()
        optimized_program = teleprompter.compile(program, trainset=trainset)
        compilation_time = time.time() - start_time
        
        # Evaluate on validation set
        total_score = 0.0
        for example in valset:
            try:
                prediction = optimized_program(question=example.question)
                score = self.accuracy_metric(example, prediction)
                total_score += score
                self.bootstrap_metrics.add_score(score)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                
        avg_score = total_score / len(valset) if valset else 0.0
        
        results = {
            "optimizer": "BootstrapFewShot",
            "compilation_time": compilation_time,
            "average_score": avg_score,
            "total_examples": len(valset),
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "demonstrations": self._extract_demonstrations(optimized_program)
        }
        
        logger.info(f"BootstrapFewShot results: {results}")
        return optimized_program, results
        
    def test_mipro_v2(self, dataset: List[dspy.Example],
                     num_trials: int = 5,
                     max_bootstrapped_demos: int = 3,
                     max_labeled_demos: int = 3) -> Tuple[dspy.Module, Dict[str, Any]]:
        """Test MIPROv2 optimizer"""
        logger.info("Testing MIPROv2 optimizer")
        
        program = BasicProgram()
        # Split dataset to match Go's approach: 3/4 for training, 1/4 for validation
        dataset_size = len(dataset)
        train_size = min(dataset_size * 3 // 4, dataset_size - 1)
        if train_size < 1:
            train_size = 1
        trainset = dataset[:train_size]
        valset = dataset[train_size:]
        
        # Create optimizer
        teleprompter = MIPROv2(
            metric=self.accuracy_metric,
            auto="light",  # Use light mode for faster testing
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            verbose=True
        )
        
        # Compile program (don't set num_trials when auto is set)
        import time
        start_time = time.time()
        optimized_program = teleprompter.compile(
            program,
            trainset=trainset,
            minibatch=True,
            minibatch_size=min(10, len(trainset)),
            requires_permission_to_run=False
        )
        compilation_time = time.time() - start_time
        
        # Evaluate on validation set
        total_score = 0.0
        for example in valset:
            try:
                prediction = optimized_program(question=example.question)
                score = self.accuracy_metric(example, prediction)
                total_score += score
                self.mipro_metrics.add_score(score)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                
        avg_score = total_score / len(valset) if valset else 0.0
        
        results = {
            "optimizer": "MIPROv2",
            "compilation_time": compilation_time,
            "average_score": avg_score,
            "total_examples": len(valset),
            "num_trials": num_trials,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "demonstrations": self._extract_demonstrations(optimized_program)
        }
        
        logger.info(f"MIPROv2 results: {results}")
        return optimized_program, results

    def test_simba(self, dataset: List[dspy.Example],
                   batch_size: int = 4,
                   max_steps: int = 6,
                   num_candidates: int = 4,
                   sampling_temperature: float = 0.2) -> Tuple[dspy.Module, Dict[str, Any]]:
        """Test SIMBA optimizer"""
        logger.info("Testing SIMBA optimizer")
        
        program = BasicProgram()
        # Split dataset to match Go's approach: 3/4 for training, 1/4 for validation
        dataset_size = len(dataset)
        train_size = min(dataset_size * 3 // 4, dataset_size - 1)
        if train_size < 1:
            train_size = 1
        trainset = dataset[:train_size]
        valset = dataset[train_size:]
        
        # Create optimizer with correct DSPy SIMBA parameter names
        teleprompter = SIMBA(
            metric=self.accuracy_metric,
            bsize=batch_size,
            max_steps=max_steps,
            num_candidates=num_candidates,
            temperature_for_sampling=sampling_temperature
        )
        
        # Compile program
        import time
        start_time = time.time()
        optimized_program = teleprompter.compile(program, trainset=trainset)
        compilation_time = time.time() - start_time
        
        # Evaluate on validation set
        total_score = 0.0
        for example in valset:
            try:
                prediction = optimized_program(question=example.question)
                score = self.accuracy_metric(example, prediction)
                total_score += score
                self.simba_metrics.add_score(score)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                # Continue with next example instead of failing
                total_score += 0.0
                self.simba_metrics.add_score(0.0)
                
        avg_score = total_score / len(valset) if valset else 0.0
        
        results = {
            "optimizer": "SIMBA",
            "compilation_time": compilation_time,
            "average_score": avg_score,
            "total_examples": len(valset),
            "batch_size": batch_size,
            "max_steps": max_steps,
            "num_candidates": num_candidates,
            "sampling_temperature": sampling_temperature,
            "demonstrations": self._extract_demonstrations(optimized_program)
        }
        
        logger.info(f"SIMBA results: {results}")
        return optimized_program, results
        
    def _extract_demonstrations(self, program: dspy.Module) -> List[Dict[str, Any]]:
        """Extract demonstrations from optimized program"""
        demonstrations = []
        
        # Try to extract demos from the predictor
        if hasattr(program, 'predictor') and hasattr(program.predictor, 'demos'):
            for demo in program.predictor.demos:
                demo_dict = {}
                try:
                    # Try to get inputs and outputs safely
                    if hasattr(demo, '_inputs') and demo._inputs is not None:
                        demo_dict["inputs"] = demo._inputs
                    elif hasattr(demo, 'inputs'):
                        try:
                            demo_dict["inputs"] = demo.inputs()
                        except ValueError:
                            demo_dict["inputs"] = {}
                    else:
                        demo_dict["inputs"] = {}
                        
                    if hasattr(demo, '_outputs') and demo._outputs is not None:
                        demo_dict["outputs"] = demo._outputs
                    elif hasattr(demo, 'outputs'):
                        try:
                            demo_dict["outputs"] = demo.outputs()
                        except ValueError:
                            demo_dict["outputs"] = {}
                    else:
                        demo_dict["outputs"] = {}
                        
                    # If we still have empty inputs/outputs, try to extract from the demo object itself
                    if not demo_dict["inputs"] and not demo_dict["outputs"]:
                        demo_dict = {"inputs": {}, "outputs": {}, "raw": str(demo)}
                        
                except Exception as e:
                    logger.warning(f"Failed to extract demo: {e}")
                    demo_dict = {"inputs": {}, "outputs": {}, "error": str(e)}
                    
                demonstrations.append(demo_dict)
                
        return demonstrations
        
    def run_comparison(self, dataset_size: int = 20) -> Dict[str, Any]:
        """Run full comparison between optimizers"""
        logger.info("Starting optimizer comparison")
        
        # Create dataset
        dataset = self.create_sample_dataset(dataset_size)
        
        # Test optimizers
        bootstrap_program, bootstrap_results = self.test_bootstrap_fewshot(dataset)
        mipro_program, mipro_results = self.test_mipro_v2(dataset)
        
        # Compare results
        comparison_results = {
            "dataset_size": dataset_size,
            "model": self.model_name,
            "bootstrap_fewshot": bootstrap_results,
            "mipro_v2": mipro_results,
            "comparison": {
                "score_difference": mipro_results["average_score"] - bootstrap_results["average_score"],
                "time_difference": mipro_results["compilation_time"] - bootstrap_results["compilation_time"],
                "better_optimizer": "MIPROv2" if mipro_results["average_score"] > bootstrap_results["average_score"] else "BootstrapFewShot"
            }
        }
        
        return comparison_results
        
    def save_results(self, results: Dict[str, Any], filename: str = "dspy_comparison_results.json"):
        """Save comparison results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
        

def main():
    """Main function for running the comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DSPy Optimizer Comparison')
    parser.add_argument('--optimizer', choices=['bootstrap', 'mipro', 'simba', 'all'], 
                       default='all', help='Optimizer to test (default: all)')
    parser.add_argument('--dataset-size', type=int, default=20, 
                       help='Dataset size for testing (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize comparison with Gemini 2.0 Flash
    comparison = OptimizerComparison("gemini/gemini-2.0-flash")
    
    # Create dataset
    dataset = comparison.create_sample_dataset(args.dataset_size)
    
    results = {
        "dataset_size": args.dataset_size,
        "model": comparison.model_name,
        "optimizer_tested": args.optimizer
    }
    
    if args.optimizer == 'bootstrap' or args.optimizer == 'all':
        print("Testing BootstrapFewShot...")
        _, bootstrap_results = comparison.test_bootstrap_fewshot(dataset)
        results["bootstrap_fewshot"] = bootstrap_results
        
    if args.optimizer == 'mipro' or args.optimizer == 'all':
        print("Testing MIPROv2...")
        _, mipro_results = comparison.test_mipro_v2(dataset)
        results["mipro_v2"] = mipro_results
    
    if args.optimizer == 'simba' or args.optimizer == 'all':
        print("Testing SIMBA...")
        _, simba_results = comparison.test_simba(dataset)
        results["simba"] = simba_results
    
    # Add comparison section if testing multiple optimizers
    if args.optimizer == 'all':
        bootstrap_score = results["bootstrap_fewshot"]["average_score"]
        mipro_score = results["mipro_v2"]["average_score"] 
        simba_score = results["simba"]["average_score"]
        bootstrap_time = results["bootstrap_fewshot"]["compilation_time"]
        mipro_time = results["mipro_v2"]["compilation_time"]
        simba_time = results["simba"]["compilation_time"]
        
        # Find best optimizer
        best_optimizer = "BootstrapFewShot"
        best_score = bootstrap_score
        if mipro_score > best_score:
            best_optimizer = "MIPROv2"
            best_score = mipro_score
        if simba_score > best_score:
            best_optimizer = "SIMBA"
            best_score = simba_score
            
        results["comparison"] = {
            "bootstrap_vs_mipro_score_diff": mipro_score - bootstrap_score,
            "bootstrap_vs_simba_score_diff": simba_score - bootstrap_score,
            "mipro_vs_simba_score_diff": simba_score - mipro_score,
            "bootstrap_vs_mipro_time_diff": mipro_time - bootstrap_time,
            "bootstrap_vs_simba_time_diff": simba_time - bootstrap_time,
            "mipro_vs_simba_time_diff": simba_time - mipro_time,
            "best_optimizer": best_optimizer,
            "best_score": best_score
        }
    
    # Save results
    comparison.save_results(results)
    
    # Print summary
    print(f"\n=== DSPy Optimizer Comparison Results ===")
    print(f"Dataset size: {results['dataset_size']}")
    print(f"Model: {results['model']}")
    print(f"Optimizer tested: {results['optimizer_tested']}")
    
    if "bootstrap_fewshot" in results:
        print(f"\nBootstrapFewShot:")
        print(f"  - Average score: {results['bootstrap_fewshot']['average_score']:.3f}")
        print(f"  - Compilation time: {results['bootstrap_fewshot']['compilation_time']:.2f}s")
        print(f"  - Demonstrations: {len(results['bootstrap_fewshot']['demonstrations'])}")
    
    if "mipro_v2" in results:
        print(f"\nMIPROv2:")
        print(f"  - Average score: {results['mipro_v2']['average_score']:.3f}")
        print(f"  - Compilation time: {results['mipro_v2']['compilation_time']:.2f}s")
        print(f"  - Demonstrations: {len(results['mipro_v2']['demonstrations'])}")
    
    if "simba" in results:
        print(f"\nSIMBA:")
        if "error" in results["simba"]:
            print(f"  - Error: {results['simba']['error']}")
        else:
            print(f"  - Average score: {results['simba']['average_score']:.3f}")
            print(f"  - Compilation time: {results['simba']['compilation_time']:.2f}s")
            print(f"  - Demonstrations: {len(results['simba']['demonstrations'])}")
    
    if "comparison" in results:
        print(f"\nBest optimizer: {results['comparison']['best_optimizer']} ({results['comparison']['best_score']:.3f})")


if __name__ == "__main__":
    main()
