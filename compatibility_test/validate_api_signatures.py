#!/usr/bin/env python3
# /// script
# dependencies = [
#     "dspy-ai>=2.4.0",
#     "google-generativeai>=0.3.0",
#     "numpy>=1.21.0",
# ]
# ///
"""
API Signature Validation Script
Validates that the Go dspy-go implementation has compatible API signatures with Python DSPy
"""

import inspect
import json
from typing import Dict, Any, List, Optional

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2


class APISignatureValidator:
    """Validates API signatures between Python DSPy and Go dspy-go"""
    
    def __init__(self):
        self.validation_results = {
            "bootstrap_fewshot": {},
            "mipro_v2": {},
            "summary": {}
        }
        
    def extract_python_signatures(self) -> Dict[str, Any]:
        """Extract Python DSPy optimizer signatures"""
        signatures = {}
        
        # BootstrapFewShot
        bootstrap_init = inspect.signature(BootstrapFewShot.__init__)
        bootstrap_compile = inspect.signature(BootstrapFewShot.compile)
        
        signatures["bootstrap_fewshot"] = {
            "init": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    for param in bootstrap_init.parameters.values()
                    if param.name != "self"
                ]
            },
            "compile": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    for param in bootstrap_compile.parameters.values()
                    if param.name != "self"
                ]
            }
        }
        
        # MIPROv2
        mipro_init = inspect.signature(MIPROv2.__init__)
        mipro_compile = inspect.signature(MIPROv2.compile)
        
        signatures["mipro_v2"] = {
            "init": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    for param in mipro_init.parameters.values()
                    if param.name != "self"
                ]
            },
            "compile": {
                "parameters": [
                    {
                        "name": param.name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    for param in mipro_compile.parameters.values()
                    if param.name != "self"
                ]
            }
        }
        
        return signatures
        
    def define_go_signatures(self) -> Dict[str, Any]:
        """Define expected Go dspy-go signatures based on codebase analysis"""
        signatures = {}
        
        # BootstrapFewShot (based on dspy-go code analysis)
        signatures["bootstrap_fewshot"] = {
            "init": {
                "parameters": [
                    {
                        "name": "metric",
                        "type": "func(example map[string]interface{}, prediction map[string]interface{}, ctx context.Context) bool",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "maxBootstrapped",
                        "type": "int",
                        "default": None,
                        "required": True
                    }
                ]
            },
            "compile": {
                "parameters": [
                    {
                        "name": "ctx",
                        "type": "context.Context",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "student",
                        "type": "core.Program",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "teacher",
                        "type": "core.Program",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "trainset",
                        "type": "[]map[string]interface{}",
                        "default": None,
                        "required": True
                    }
                ]
            }
        }
        
        # MIPRO (based on dspy-go code analysis)
        signatures["mipro"] = {
            "init": {
                "parameters": [
                    {
                        "name": "metric",
                        "type": "func(example, prediction map[string]interface{}, ctx context.Context) float64",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "opts",
                        "type": "...MIPROOption",
                        "default": None,
                        "required": False
                    }
                ]
            },
            "compile": {
                "parameters": [
                    {
                        "name": "ctx",
                        "type": "context.Context",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "program",
                        "type": "core.Program",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "dataset",
                        "type": "core.Dataset",
                        "default": None,
                        "required": True
                    },
                    {
                        "name": "metric",
                        "type": "core.Metric",
                        "default": None,
                        "required": True
                    }
                ]
            }
        }
        
        return signatures
        
    def validate_bootstrap_fewshot(self, python_sig: Dict[str, Any], go_sig: Dict[str, Any]) -> Dict[str, Any]:
        """Validate BootstrapFewShot API compatibility"""
        validation = {
            "init_compatibility": self._validate_method_compatibility(
                python_sig["init"], go_sig["init"]
            ),
            "compile_compatibility": self._validate_method_compatibility(
                python_sig["compile"], go_sig["compile"]
            )
        }
        
        # Check core parameter compatibility
        core_params = {
            "metric": "Required metric function",
            "max_bootstrapped_demos": "Maximum bootstrapped demonstrations",
            "max_labeled_demos": "Maximum labeled demonstrations"
        }
        
        validation["core_parameter_mapping"] = self._check_core_parameters(
            python_sig, go_sig, core_params
        )
        
        return validation
        
    def validate_mipro(self, python_sig: Dict[str, Any], go_sig: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MIPRO API compatibility"""
        validation = {
            "init_compatibility": self._validate_method_compatibility(
                python_sig["init"], go_sig["init"]
            ),
            "compile_compatibility": self._validate_method_compatibility(
                python_sig["compile"], go_sig["compile"]
            )
        }
        
        # Check core parameter compatibility
        core_params = {
            "metric": "Required metric function",
            "num_trials": "Number of optimization trials",
            "max_bootstrapped_demos": "Maximum bootstrapped demonstrations",
            "max_labeled_demos": "Maximum labeled demonstrations"
        }
        
        validation["core_parameter_mapping"] = self._check_core_parameters(
            python_sig, go_sig, core_params
        )
        
        return validation
        
    def _validate_method_compatibility(self, python_method: Dict[str, Any], go_method: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compatibility between Python and Go method signatures"""
        compatibility = {
            "parameter_count": {
                "python": len(python_method["parameters"]),
                "go": len(go_method["parameters"]),
                "compatible": True
            },
            "required_parameters": {
                "python": [p["name"] for p in python_method["parameters"] if p["required"]],
                "go": [p["name"] for p in go_method["parameters"] if p["required"]],
                "compatible": True
            },
            "parameter_mapping": [],
            "missing_in_go": [],
            "extra_in_go": []
        }
        
        # Map parameters
        python_params = {p["name"]: p for p in python_method["parameters"]}
        go_params = {p["name"]: p for p in go_method["parameters"]}
        
        # Check for missing parameters
        for py_param in python_params:
            if py_param not in go_params:
                compatibility["missing_in_go"].append(py_param)
                
        # Check for extra parameters
        for go_param in go_params:
            if go_param not in python_params:
                compatibility["extra_in_go"].append(go_param)
                
        # Overall compatibility
        compatibility["compatible"] = (
            len(compatibility["missing_in_go"]) == 0 and
            len(compatibility["extra_in_go"]) == 0
        )
        
        return compatibility
        
    def _check_core_parameters(self, python_sig: Dict[str, Any], go_sig: Dict[str, Any], core_params: Dict[str, str]) -> Dict[str, Any]:
        """Check if core parameters are properly mapped"""
        mapping = {}
        
        # Get all parameters from both signatures
        all_python_params = {}
        all_go_params = {}
        
        for method in ["init", "compile"]:
            for param in python_sig[method]["parameters"]:
                all_python_params[param["name"]] = param
            for param in go_sig[method]["parameters"]:
                all_go_params[param["name"]] = param
                
        # Check core parameter mapping
        for param_name, description in core_params.items():
            mapping[param_name] = {
                "description": description,
                "in_python": param_name in all_python_params,
                "in_go": param_name in all_go_params or self._find_similar_param(param_name, all_go_params),
                "compatible": True
            }
            
        return mapping
        
    def _find_similar_param(self, param_name: str, go_params: Dict[str, Any]) -> bool:
        """Find similar parameter names in Go implementation"""
        # Map common parameter variations
        name_mappings = {
            "max_bootstrapped_demos": ["maxBootstrapped", "max_bootstrapped_demos"],
            "max_labeled_demos": ["maxLabeled", "max_labeled_demos"],
            "num_trials": ["numTrials", "num_trials"]
        }
        
        if param_name in name_mappings:
            for variant in name_mappings[param_name]:
                if variant in go_params:
                    return True
                    
        return False
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete API signature validation"""
        # Extract signatures
        python_signatures = self.extract_python_signatures()
        go_signatures = self.define_go_signatures()
        
        # Validate BootstrapFewShot
        self.validation_results["bootstrap_fewshot"] = self.validate_bootstrap_fewshot(
            python_signatures["bootstrap_fewshot"],
            go_signatures["bootstrap_fewshot"]
        )
        
        # Validate MIPRO (note: Go uses "mipro" key)
        self.validation_results["mipro_v2"] = self.validate_mipro(
            python_signatures["mipro_v2"],
            go_signatures["mipro"]
        )
        
        # Generate summary
        self.validation_results["summary"] = self._generate_summary()
        
        return self.validation_results
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "overall_compatible": True,
            "bootstrap_fewshot_compatible": True,
            "mipro_compatible": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check BootstrapFewShot compatibility
        bootstrap_compat = self.validation_results["bootstrap_fewshot"]
        if not bootstrap_compat["init_compatibility"]["compatible"]:
            summary["bootstrap_fewshot_compatible"] = False
            summary["issues"].append("BootstrapFewShot init signature incompatible")
            
        if not bootstrap_compat["compile_compatibility"]["compatible"]:
            summary["bootstrap_fewshot_compatible"] = False
            summary["issues"].append("BootstrapFewShot compile signature incompatible")
            
        # Check MIPRO compatibility
        mipro_compat = self.validation_results["mipro_v2"]
        if not mipro_compat["init_compatibility"]["compatible"]:
            summary["mipro_compatible"] = False
            summary["issues"].append("MIPRO init signature incompatible")
            
        if not mipro_compat["compile_compatibility"]["compatible"]:
            summary["mipro_compatible"] = False
            summary["issues"].append("MIPRO compile signature incompatible")
            
        # Overall compatibility
        summary["overall_compatible"] = (
            summary["bootstrap_fewshot_compatible"] and
            summary["mipro_compatible"]
        )
        
        # Generate recommendations
        if not summary["overall_compatible"]:
            summary["recommendations"].append("Review API signatures for compatibility")
            summary["recommendations"].append("Consider adapter patterns for incompatible signatures")
            
        return summary
        
    def print_validation_report(self):
        """Print detailed validation report"""
        results = self.run_validation()
        
        print("=" * 60)
        print("API Signature Validation Report")
        print("=" * 60)
        
        # Summary
        print("\n📊 VALIDATION SUMMARY")
        print("-" * 30)
        summary = results["summary"]
        status = "✅" if summary["overall_compatible"] else "❌"
        print(f"{status} Overall Compatible: {summary['overall_compatible']}")
        
        if summary["issues"]:
            print("\n🚨 Issues Found:")
            for issue in summary["issues"]:
                print(f"  - {issue}")
                
        if summary["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"  - {rec}")
                
        # Detailed results
        print("\n🔄 BOOTSTRAP FEWSHOT VALIDATION")
        print("-" * 35)
        self._print_optimizer_validation(results["bootstrap_fewshot"])
        
        print("\n🎯 MIPRO VALIDATION")
        print("-" * 20)
        self._print_optimizer_validation(results["mipro_v2"])
        
        print("\n" + "=" * 60)
        
    def _print_optimizer_validation(self, validation: Dict[str, Any]):
        """Print validation results for a specific optimizer"""
        init_compat = validation["init_compatibility"]
        compile_compat = validation["compile_compatibility"]
        
        print(f"Init Method: {'✅' if init_compat['compatible'] else '❌'}")
        if init_compat["missing_in_go"]:
            print(f"  Missing in Go: {init_compat['missing_in_go']}")
        if init_compat["extra_in_go"]:
            print(f"  Extra in Go: {init_compat['extra_in_go']}")
            
        print(f"Compile Method: {'✅' if compile_compat['compatible'] else '❌'}")
        if compile_compat["missing_in_go"]:
            print(f"  Missing in Go: {compile_compat['missing_in_go']}")
        if compile_compat["extra_in_go"]:
            print(f"  Extra in Go: {compile_compat['extra_in_go']}")
            
    def save_validation_report(self, filename: str = "api_validation_report.json"):
        """Save validation report to file"""
        results = self.run_validation()
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"API validation report saved to {filename}")


def main():
    """Main function"""
    validator = APISignatureValidator()
    validator.print_validation_report()
    validator.save_validation_report()


if __name__ == "__main__":
    main()