#!/usr/bin/env python3
"""
Comprehensive DART System Test Runner

This script runs all test suites and provides detailed analysis of each step
in the DART pipeline. It serves as both a testing framework and an educational
tool to understand how each component works.

Usage:
    python run_dart_tests.py [options]

Options:
    --unit          Run only unit tests
    --integration   Run only integration tests  
    --performance   Run only performance tests
    --all           Run all tests (default)
    --explain       Show detailed explanations of each step
    --verbose       Detailed test output
    --quick         Skip slow tests
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

# Add tests directory to path
tests_dir = Path(__file__).parent / 'tests'
sys.path.append(str(tests_dir))

# Import test modules
from test_data_loader import run_data_loader_tests
from test_embedding import run_embedding_tests
from test_noise import run_noise_tests
from test_reconstruction import run_reconstruction_tests
from test_integration import run_integration_tests
from test_performance import run_performance_tests

class DARTTestRunner:
    """Comprehensive test runner for DART system"""
    
    def __init__(self, verbose=False, explain=False, quick=False):
        self.verbose = verbose
        self.explain = explain
        self.quick = quick
        self.results = {}
        
    def print_header(self, title, char="="):
        """Print a formatted header"""
        width = 60
        print()
        print(char * width)
        print(f" {title} ".center(width, char))
        print(char * width)
        print()
    
    def print_explanation(self, step_name, explanation):
        """Print step explanation if enabled"""
        if self.explain:
            print(f"📚 {step_name} Explanation:")
            print(f"   {explanation}")
            print()
    
    def run_unit_tests(self):
        """Run all unit tests with explanations"""
        self.print_header("DART Unit Tests", "=")
        
        unit_results = {}
        
        # Step 1: Data Loading Tests
        self.print_explanation(
            "Data Loading",
            "Tests the ChineseDataLoader component that reads harmful prompts from CSV files "
            "and generates benign examples. This is the first step in the DART pipeline where "
            "we prepare the input data for processing."
        )
        
        print("🔍 Testing Data Loading Component...")
        try:
            unit_results['data_loader'] = run_data_loader_tests()
        except Exception as e:
            print(f"❌ Data loader tests failed: {e}")
            unit_results['data_loader'] = False
        
        # Step 2: Embedding Tests
        self.print_explanation(
            "Text Embedding",
            "Tests the ChineseEmbedding component that converts Chinese text into 512-dimensional "
            "vectors using SBERT or Unicode fallback. This step transforms variable-length text "
            "into fixed-size numerical representations that can be mathematically manipulated."
        )
        
        print("🔍 Testing Embedding Component...")
        try:
            unit_results['embedding'] = run_embedding_tests()
        except Exception as e:
            print(f"❌ Embedding tests failed: {e}")
            unit_results['embedding'] = False
        
        # Step 3: Noise Generation Tests
        self.print_explanation(
            "Noise Generation",
            "Tests the DiffusionNoise component that generates controlled Gaussian noise using "
            "Box-Muller transform with L2 norm constraints. This creates perturbations that "
            "modify embeddings while staying within semantic boundaries."
        )
        
        print("🔍 Testing Noise Generation Component...")
        try:
            unit_results['noise'] = run_noise_tests()
        except Exception as e:
            print(f"❌ Noise tests failed: {e}")
            unit_results['noise'] = False
        
        # Step 4: Reconstruction Tests
        self.print_explanation(
            "Text Reconstruction",
            "Tests the ChineseVec2Text component that converts perturbed embeddings back to "
            "Chinese text using T5 model or heuristic methods. This step completes the "
            "perturbation cycle by generating modified text that maintains semantic similarity."
        )
        
        print("🔍 Testing Reconstruction Component...")
        try:
            unit_results['reconstruction'] = run_reconstruction_tests()
        except Exception as e:
            print(f"❌ Reconstruction tests failed: {e}")
            unit_results['reconstruction'] = False
        
        self.results['unit_tests'] = unit_results
        return all(unit_results.values())
    
    def run_integration_tests(self):
        """Run integration tests with explanations"""
        self.print_header("DART Integration Tests", "=")
        
        self.print_explanation(
            "Integration Testing",
            "Tests how all DART components work together in the complete pipeline: "
            "Text → Embedding → Noise Addition → Perturbation → Reconstruction → Output. "
            "This validates the end-to-end flow and component interactions."
        )
        
        print("🔍 Testing End-to-End Integration...")
        try:
            integration_result = run_integration_tests()
            self.results['integration_tests'] = integration_result
            return integration_result
        except Exception as e:
            print(f"❌ Integration tests failed: {e}")
            self.results['integration_tests'] = False
            return False
    
    def run_performance_tests(self):
        """Run performance tests with explanations"""
        self.print_header("DART Performance Tests", "=")
        
        if self.quick:
            print("⚡ Skipping performance tests (quick mode)")
            self.results['performance_tests'] = True
            return True
        
        self.print_explanation(
            "Performance Testing",
            "Measures throughput, latency, memory usage, and scalability of each component "
            "and the complete system. This ensures the DART system can handle real-world "
            "workloads efficiently."
        )
        
        print("🔍 Testing System Performance...")
        try:
            performance_result = run_performance_tests()
            self.results['performance_tests'] = performance_result
            return performance_result
        except Exception as e:
            print(f"❌ Performance tests failed: {e}")
            self.results['performance_tests'] = False
            return False
    
    def explain_dart_pipeline(self):
        """Provide detailed explanation of the DART pipeline"""
        self.print_header("DART Pipeline Explanation", "📘")
        
        pipeline_steps = [
            {
                "step": "1. Data Loading",
                "component": "ChineseDataLoader",
                "input": "CSV file with harmful prompts",
                "output": "List of Chinese text strings",
                "purpose": "Loads test data and generates benign examples for comparison",
                "algorithm": "CSV parsing + synthetic benign prompt generation"
            },
            {
                "step": "2. Text Embedding", 
                "component": "ChineseEmbedding",
                "input": "Chinese text strings",
                "output": "512-dimensional embeddings",
                "purpose": "Converts variable-length text to fixed-size numerical vectors",
                "algorithm": "SBERT Chinese model or Unicode position encoding"
            },
            {
                "step": "3. Noise Generation",
                "component": "DiffusionNoise", 
                "input": "Original embeddings",
                "output": "Constrained noise vectors",
                "purpose": "Creates controlled perturbations within semantic boundaries",
                "algorithm": "Box-Muller Gaussian noise + L2 norm constraints"
            },
            {
                "step": "4. Perturbation Application",
                "component": "DiffusionNoise",
                "input": "Embeddings + noise vectors",
                "output": "Perturbed embeddings",
                "purpose": "Applies noise to create adversarial embeddings",
                "algorithm": "Vector addition with proximity constraints"
            },
            {
                "step": "5. Text Reconstruction",
                "component": "ChineseVec2Text",
                "input": "Perturbed embeddings + original text",
                "output": "Modified Chinese text",
                "purpose": "Converts perturbed vectors back to readable text",
                "algorithm": "T5 vec2text model or heuristic synonym replacement"
            },
            {
                "step": "6. Quality Assessment",
                "component": "DARTController",
                "input": "Original + reconstructed text",
                "output": "Similarity scores and metrics",
                "purpose": "Evaluates perturbation effectiveness and text quality",
                "algorithm": "Text similarity computation + statistical analysis"
            }
        ]
        
        for step_info in pipeline_steps:
            print(f"🔄 {step_info['step']}")
            print(f"   Component: {step_info['component']}")
            print(f"   Input: {step_info['input']}")
            print(f"   Output: {step_info['output']}")
            print(f"   Purpose: {step_info['purpose']}")
            print(f"   Algorithm: {step_info['algorithm']}")
            print()
        
        print("🎯 Overall Goal:")
        print("   The DART system tests AI robustness by creating semantically similar")
        print("   but slightly modified text that may bypass content filters while")
        print("   maintaining the original meaning. This helps identify vulnerabilities")
        print("   in AI safety systems.")
        print()
    
    def run_quick_validation(self):
        """Run a quick validation of the DART system"""
        self.print_header("Quick DART Validation", "⚡")
        
        print("🚀 Running quick system validation...")
        
        # Test basic system functionality
        try:
            import sys
            from pathlib import Path
            
            # Add dart_system to path
            dart_path = Path(__file__).parent / 'dart_system'
            sys.path.append(str(dart_path))
            
            from core.dart_controller import DARTController, DARTConfig
            
            # Quick config
            config = DARTConfig(
                csv_path="problem.csv",
                verbose=False,
                random_seed=42
            )
            
            # Test initialization
            print("  ✓ Initializing DART controller...")
            controller = DARTController(config)
            
            # Test single attack
            print("  ✓ Testing single text attack...")
            result = controller.run_attack(["測試系統是否正常運作"])
            
            if result["success"]:
                similarity = result["similarities"][0]
                print(f"  ✓ Attack successful! Similarity: {similarity:.3f}")
                return True
            else:
                print(f"  ❌ Attack failed: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.print_header("Test Report Summary", "📊")
        
        total_tests = 0
        passed_tests = 0
        
        for test_suite, results in self.results.items():
            if isinstance(results, dict):
                # Unit tests
                suite_total = len(results)
                suite_passed = sum(1 for r in results.values() if r)
                total_tests += suite_total
                passed_tests += suite_passed
                
                print(f"📋 {test_suite.replace('_', ' ').title()}:")
                for component, success in results.items():
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(f"   {component}: {status}")
            else:
                # Integration/Performance tests
                total_tests += 1
                if results:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                print(f"📋 {test_suite.replace('_', ' ').title()}: {status}")
        
        print()
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"🎯 Overall Results:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Recommendations
        if success_rate >= 90:
            print(f"   Status: 🟢 EXCELLENT - System is ready for production")
        elif success_rate >= 75:
            print(f"   Status: 🟡 GOOD - Minor issues to address")
        elif success_rate >= 50:
            print(f"   Status: 🟠 WARNING - Significant issues detected")
        else:
            print(f"   Status: 🔴 CRITICAL - Major problems need fixing")
        
        return success_rate >= 75

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive DART System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_dart_tests.py --all --explain    # Run all tests with explanations
    python run_dart_tests.py --unit --verbose   # Run only unit tests with details
    python run_dart_tests.py --quick            # Quick validation only
    python run_dart_tests.py --performance      # Performance tests only
        """
    )
    
    parser.add_argument('--unit', action='store_true', 
                      help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                      help='Run only integration tests')
    parser.add_argument('--performance', action='store_true',
                      help='Run only performance tests')
    parser.add_argument('--all', action='store_true',
                      help='Run all tests (default)')
    parser.add_argument('--explain', action='store_true',
                      help='Show detailed explanations of each step')
    parser.add_argument('--verbose', action='store_true',
                      help='Detailed test output')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick validation only')
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test type selected
    if not any([args.unit, args.integration, args.performance, args.quick]):
        args.all = True
    
    # Create test runner
    runner = DARTTestRunner(
        verbose=args.verbose,
        explain=args.explain,
        quick=args.quick
    )
    
    # Print system information
    print("🚀 DART System Comprehensive Test Suite")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    start_time = time.time()
    all_passed = True
    
    # Quick validation mode
    if args.quick:
        success = runner.run_quick_validation()
        runner.results['quick_validation'] = success
        all_passed = success
    else:
        # Explain pipeline if requested
        if args.explain:
            runner.explain_dart_pipeline()
        
        # Run selected test suites
        if args.unit or args.all:
            success = runner.run_unit_tests()
            all_passed = all_passed and success
        
        if args.integration or args.all:
            success = runner.run_integration_tests()
            all_passed = all_passed and success
        
        if args.performance or args.all:
            success = runner.run_performance_tests()
            all_passed = all_passed and success
    
    # Generate final report
    execution_time = time.time() - start_time
    
    if not args.quick:
        final_success = runner.generate_test_report()
        all_passed = all_passed and final_success
    
    # Final summary
    runner.print_header("Execution Complete", "🏁")
    print(f"⏱️  Total execution time: {execution_time:.1f} seconds")
    
    if all_passed:
        print("🎉 All tests passed! DART system is functioning correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()