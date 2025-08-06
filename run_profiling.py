#!/usr/bin/env python3
"""
Profiling suite runner for the CHES 2025 experiment.
This script provides an easy interface to run different types of profiling.
"""

import argparse
import sys
import os
import subprocess
from typing import List

def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {command[0]}")
        return False

def check_requirements():
    """Check if required dependencies are available."""
    print("Checking requirements...")
    
    try:
        import torch
        import matplotlib
        import psutil
        import numpy
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def run_comprehensive_profiling():
    """Run comprehensive profiling of the experiment."""
    return run_command(
        ["python", "profiler/profile_experiment.py"],
        "Comprehensive profiling (CPU, memory, GPU, execution time)"
    )

def run_memory_profiling():
    """Run memory-focused profiling."""
    return run_command(
        ["python", "profiler/memory_profiler.py"],
        "Memory profiling (allocation patterns, memory usage by phase)"
    )

def run_gpu_profiling():
    """Run GPU-focused profiling."""
    return run_command(
        ["python", "profiler/gpu_profiler.py"],
        "GPU profiling (CUDA operations, GPU memory usage)"
    )

def run_flamegraph_profiling():
    """Run flame graph profiling."""
    return run_command(
        ["python", "profiler/flame_graph_profiler.py"],
        "Flame graph profiling (CPU flame graphs with py-spy)"
    )

def run_baseline_experiment():
    """Run the original experiment without profiling."""
    return run_command(
        ["python", "main.py"],
        "Baseline experiment (no profiling)"
    )

def run_quick_profile():
    """Run a quick profiling session with cProfile."""
    return run_command(
        ["python", "-m", "cProfile", "-o", "quick_profile.prof", "main.py"],
        "Quick cProfile profiling"
    )

def analyze_quick_profile():
    """Analyze the quick profile results."""
    if not os.path.exists("quick_profile.prof"):
        print("✗ Quick profile file not found. Run quick profiling first.")
        return False
    
    print("\nAnalyzing quick profile results...")
    
    try:
        import pstats
        
        # Create analysis file
        with open("quick_profile_analysis.txt", "w") as f:
            stats = pstats.Stats("quick_profile.prof", stream=f)
            f.write("Top 20 functions by cumulative time:\n")
            f.write("=" * 40 + "\n")
            stats.sort_stats('cumulative').print_stats(20)
            f.write("\n\nTop 20 functions by total time:\n")
            f.write("=" * 40 + "\n")
            stats.sort_stats('tottime').print_stats(20)
        
        print("✓ Quick profile analysis saved to quick_profile_analysis.txt")
        
        # Print summary to console
        stats = pstats.Stats("quick_profile.prof")
        print("\nTop 10 functions by cumulative time:")
        stats.sort_stats('cumulative').print_stats(10)
        
        return True
        
    except Exception as e:
        print(f"✗ Error analyzing profile: {e}")
        return False

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Profiling suite for CHES 2025 experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_profiling.py --all                 # Run all profiling types
  python run_profiling.py --comprehensive       # Run comprehensive profiling
  python run_profiling.py --memory              # Run memory profiling only
  python run_profiling.py --gpu                 # Run GPU profiling only
  python run_profiling.py --quick               # Run quick cProfile
  python run_profiling.py --baseline            # Run experiment without profiling
        """
    )
    
    # Profiling type arguments
    parser.add_argument("--all", action="store_true", 
                       help="Run all profiling types")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive profiling")
    parser.add_argument("--memory", action="store_true",
                       help="Run memory profiling")
    parser.add_argument("--gpu", action="store_true",
                       help="Run GPU profiling")
    parser.add_argument("--flamegraph", action="store_true",
                       help="Run flame graph profiling")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick cProfile profiling")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline experiment without profiling")
    parser.add_argument("--analyze-quick", action="store_true",
                       help="Analyze existing quick profile results")
    
    # Options
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip requirement checks")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Check requirements unless skipped
    if not args.skip_checks and not check_requirements():
        sys.exit(1)
    
    success_count = 0
    total_count = 0
    
    # Run requested profiling
    if args.analyze_quick:
        total_count += 1
        if analyze_quick_profile():
            success_count += 1
    
    if args.baseline:
        total_count += 1
        if run_baseline_experiment():
            success_count += 1
    
    if args.quick:
        total_count += 1
        if run_quick_profile():
            success_count += 1
            # Auto-analyze if successful
            if analyze_quick_profile():
                pass  # Don't count this as separate
    
    if args.all or args.comprehensive:
        total_count += 1
        if run_comprehensive_profiling():
            success_count += 1
    
    if args.all or args.memory:
        total_count += 1
        if run_memory_profiling():
            success_count += 1
    
    if args.all or args.gpu:
        total_count += 1
        if run_gpu_profiling():
            success_count += 1
    
    if args.all or args.flamegraph:
        total_count += 1
        if run_flamegraph_profiling():
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Profiling Summary: {success_count}/{total_count} tasks completed successfully")
    print(f"{'='*60}")
    
    if success_count < total_count:
        print("\nSome profiling tasks failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll profiling tasks completed successfully!")
        
        # Show where results are saved
        result_dirs = []
        if os.path.exists("./Profiling_Results"):
            result_dirs.append("./Profiling_Results")
        if os.path.exists("./Memory_Profiling"):
            result_dirs.append("./Memory_Profiling")
        if os.path.exists("./GPU_Profiling"):
            result_dirs.append("./GPU_Profiling")
        
        if result_dirs:
            print(f"\nResults saved to: {', '.join(result_dirs)}")

if __name__ == "__main__":
    main()
