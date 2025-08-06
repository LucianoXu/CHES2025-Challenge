#!/usr/bin/env python3
"""
GPU profiling script for the CHES 2025 experiment.
This script specifically analyzes GPU utilization and CUDA operations.
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List
import json
import sys

# Add the parent directory to Python path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the experiment modules
from src.experiment import experiment
from src.config import Config

class GPUProfiler:
    """GPU-focused profiler for the experiment."""
    
    def __init__(self, config_template: Dict):
        self.config = Config(config_template)
        self.gpu_metrics = {
            'memory_allocated': [],
            'memory_reserved': [],
            'timestamps': [],
            'phases': []
        }
        
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("CUDA not available. GPU profiling will be limited.")
    
    def record_gpu_metrics(self, phase: str):
        """Record current GPU metrics."""
        if not self.cuda_available:
            return
            
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        
        self.gpu_metrics['memory_allocated'].append(allocated)
        self.gpu_metrics['memory_reserved'].append(reserved)
        self.gpu_metrics['timestamps'].append(time.time())
        self.gpu_metrics['phases'].append(phase)
        
        print(f"[GPU-{phase}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
    
    def profile_gpu_usage(self):
        """Profile GPU usage during experiment execution."""
        if not self.cuda_available:
            print("GPU profiling requires CUDA. Running experiment without GPU profiling.")
            experiment(self.config)
            return
        
        print("Starting GPU profiling...")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
        
        # Initial state
        torch.cuda.empty_cache()
        self.record_gpu_metrics("initial")
        
        try:
            # Enable GPU profiling
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=self._save_trace,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                
                # Monitor GPU during experiment
                self._run_experiment_with_monitoring()
                
        except Exception as e:
            print(f"Error during GPU profiling: {e}")
            # Run without profiling as fallback
            experiment(self.config)
        
        finally:
            self.record_gpu_metrics("final")
            self._generate_gpu_report()
    
    def _run_experiment_with_monitoring(self):
        """Run experiment with periodic GPU monitoring."""
        # Create a monitoring thread or use a simpler approach
        self.record_gpu_metrics("experiment_start")
        
        # Run the actual experiment
        experiment(self.config)
        
        self.record_gpu_metrics("experiment_end")
    
    def _save_trace(self, prof):
        """Save profiler trace."""
        profile_dir = f"./GPU_Profiling/expr_{self.config['expr_num']}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # Save trace file
        trace_file = os.path.join(profile_dir, "gpu_trace.json")
        prof.export_chrome_trace(trace_file)
        
        # Save summary
        summary_file = os.path.join(profile_dir, "gpu_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("GPU Profiling Summary\n")
            f.write("=" * 20 + "\n\n")
            f.write("Top CUDA operations by self time:\n")
            f.write(str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)))
            f.write("\n\nTop operations by CPU time:\n")
            f.write(str(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)))
    
    def _generate_gpu_report(self):
        """Generate GPU usage report."""
        if not self.cuda_available or not self.gpu_metrics['timestamps']:
            print("No GPU metrics to report.")
            return
        
        profile_dir = f"./GPU_Profiling/expr_{self.config['expr_num']}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        phases = self.gpu_metrics['phases']
        allocated = self.gpu_metrics['memory_allocated']
        reserved = self.gpu_metrics['memory_reserved']
        
        x_positions = range(len(phases))
        
        # Memory usage plot
        ax1.plot(x_positions, allocated, 'b-o', label='Allocated Memory', linewidth=2, markersize=6)
        ax1.plot(x_positions, reserved, 'r-s', label='Reserved Memory', linewidth=2, markersize=6)
        ax1.set_xlabel('Execution Phase')
        ax1.set_ylabel('GPU Memory (MB)')
        ax1.set_title('GPU Memory Usage During Experiment')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(phases, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory efficiency
        efficiency = [a/r * 100 if r > 0 else 0 for a, r in zip(allocated, reserved)]
        ax2.plot(x_positions, efficiency, 'g-^', label='Memory Efficiency (%)', linewidth=2, markersize=6)
        ax2.set_xlabel('Execution Phase')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('GPU Memory Efficiency (Allocated/Reserved)')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(phases, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plot_file = os.path.join(profile_dir, "gpu_memory_usage.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_file = os.path.join(profile_dir, "gpu_report.txt")
        with open(report_file, 'w') as f:
            f.write("GPU Profiling Report\n")
            f.write("=" * 20 + "\n\n")
            
            f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB\n\n")
            
            f.write("GPU Memory Usage by Phase:\n")
            f.write("-" * 30 + "\n")
            for i, phase in enumerate(phases):
                alloc = allocated[i]
                reserv = reserved[i]
                eff = efficiency[i]
                f.write(f"{phase:.<20} Allocated: {alloc:>8.1f} MB, Reserved: {reserv:>8.1f} MB, Efficiency: {eff:>5.1f}%\n")
            
            f.write("\nGPU Usage Analysis:\n")
            f.write("-" * 19 + "\n")
            peak_allocated = max(allocated)
            peak_reserved = max(reserved)
            avg_efficiency = np.mean(efficiency)
            
            f.write(f"Peak allocated memory: {peak_allocated:.1f} MB\n")
            f.write(f"Peak reserved memory: {peak_reserved:.1f} MB\n")
            f.write(f"Average memory efficiency: {avg_efficiency:.1f}%\n")
            
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            utilization = (peak_allocated / total_gpu_memory) * 100
            f.write(f"Peak GPU utilization: {utilization:.1f}%\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            if utilization < 50:
                f.write("• GPU memory underutilized. Consider increasing batch size.\n")
            elif utilization > 90:
                f.write("• High GPU memory usage. Consider reducing batch size to avoid OOM.\n")
            else:
                f.write("• GPU memory utilization looks good.\n")
                
            if avg_efficiency < 70:
                f.write("• Low memory efficiency. Consider optimizing memory allocation patterns.\n")
            else:
                f.write("• Good memory efficiency.\n")
        
        print(f"\nGPU profiling report saved to: {profile_dir}")
        print(f"Peak GPU memory: {peak_allocated:.1f} MB ({utilization:.1f}% of total)")


def main():
    """Main GPU profiling function."""
    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())
    
    profiler = GPUProfiler(config_template)
    profiler.profile_gpu_usage()


if __name__ == "__main__":
    main()
