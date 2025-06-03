#!/usr/bin/env python3
"""
Phase 1 Stability Validation Script

This script executes the comprehensive Phase 1 stability plan:
1. Backend stability - Database connections and error handling
2. Hardware calibration - Sensor calibration and optimization  
3. Complete integration testing - End-to-end feature validation

Usage:
    python run_phase1_stability.py [--quick] [--report-only]
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Import test modules
from test_phase1_stability import run_phase1_stability_tests, run_quick_stability_check
from ophira.core.logging import setup_logging

logger = setup_logging()

def print_banner():
    """Print Phase 1 stability banner"""
    print("=" * 80)
    print("🔧 OPHIRA AI - PHASE 1 STABILITY VALIDATION")
    print("=" * 80)
    print("Objectives:")
    print("  1. ✅ Backend Stability - Database connections & error handling")
    print("  2. 🔧 Hardware Calibration - Sensor optimization & fine-tuning")
    print("  3. 🧪 Integration Testing - End-to-end feature validation")
    print("=" * 80)
    print()

def print_quick_results(results):
    """Print quick test results in a formatted way"""
    print("🚀 QUICK STABILITY CHECK RESULTS")
    print("-" * 40)
    
    status_icon = "✅" if results["overall_status"] == "PASSED" else "❌"
    print(f"Overall Status: {status_icon} {results['overall_status']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Timestamp: {results['timestamp']}")
    print()
    
    print("Component Status:")
    for component, status in results["tests"].items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
    print()

def print_comprehensive_results(results):
    """Print comprehensive test results"""
    print("🔬 COMPREHENSIVE STABILITY TEST RESULTS")
    print("-" * 50)
    
    summary = results["test_summary"]
    status_icon = "✅" if summary["overall_status"] == "PASSED" else "❌"
    
    print(f"Overall Status: {status_icon} {summary['overall_status']}")
    print(f"Tests Passed: {summary['passed']}/{summary['total_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print()
    
    print("Test Category Results:")
    for category, result in results["test_results"].items():
        status_icon = "✅" if result["status"] == "PASSED" else "❌"
        print(f"  {status_icon} {category}: {result['status']}")
    print()
    
    if results.get("recommendations"):
        print("📋 RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
        print()
    
    if results.get("next_steps"):
        print("🎯 NEXT STEPS:")
        for i, step in enumerate(results["next_steps"], 1):
            print(f"  {i}. {step}")
        print()

def save_report(results, filename):
    """Save test results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Report saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        return False

async def run_stability_validation(quick_only=False, report_only=False):
    """Run the stability validation process"""
    
    if report_only:
        # Just load and display existing report
        report_file = "phase1_stability_report.json"
        if Path(report_file).exists():
            try:
                with open(report_file, 'r') as f:
                    results = json.load(f)
                print_comprehensive_results(results)
                return True
            except Exception as e:
                print(f"❌ Failed to load report: {e}")
                return False
        else:
            print(f"❌ Report file not found: {report_file}")
            return False
    
    print_banner()
    
    # Step 1: Quick stability check
    print("🚀 Step 1: Quick Stability Check")
    print("Performing rapid system validation...")
    print()
    
    try:
        quick_results = await run_quick_stability_check()
        print_quick_results(quick_results)
        
        # Save quick results
        save_report(quick_results, "quick_stability_check.json")
        
        if quick_results["overall_status"] != "PASSED":
            print("❌ Quick stability check failed!")
            print("   Please address the failing components before proceeding.")
            print("   Run with --report-only to view detailed results.")
            return False
        
        if quick_only:
            print("✅ Quick stability check completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Quick stability check failed with error: {e}")
        return False
    
    # Step 2: Comprehensive testing
    print("🔬 Step 2: Comprehensive Stability Testing")
    print("Running full test suite (this may take several minutes)...")
    print()
    
    try:
        comprehensive_results = await run_phase1_stability_tests()
        print_comprehensive_results(comprehensive_results)
        
        # Save comprehensive results
        save_report(comprehensive_results, "phase1_stability_report.json")
        
        if comprehensive_results["test_summary"]["overall_status"] == "PASSED":
            print("🎉 PHASE 1 STABILITY VALIDATION COMPLETE!")
            print("   All systems are stable and ready for production use.")
            print("   You may proceed to Phase 2 development.")
        else:
            print("⚠️  PHASE 1 STABILITY VALIDATION INCOMPLETE")
            print("   Some tests failed. Please review the recommendations above.")
            print("   Address the issues and re-run the validation.")
        
        return comprehensive_results["test_summary"]["overall_status"] == "PASSED"
        
    except Exception as e:
        print(f"❌ Comprehensive testing failed with error: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Phase 1 Stability Validation for Ophira AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase1_stability.py                 # Run full validation
  python run_phase1_stability.py --quick         # Run quick check only
  python run_phase1_stability.py --report-only   # Display last report
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run only quick stability check (faster)"
    )
    
    parser.add_argument(
        "--report-only",
        action="store_true", 
        help="Display the last comprehensive test report"
    )
    
    args = parser.parse_args()
    
    # Run the validation
    try:
        success = asyncio.run(run_stability_validation(
            quick_only=args.quick,
            report_only=args.report_only
        ))
        
        if success:
            print("\n✅ Validation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Validation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 