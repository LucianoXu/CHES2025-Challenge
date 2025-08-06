#!/usr/bin/env python3
"""
Memory-focused profiling script for the CHES 2025 experiment.
This script specifically analyzes memory allocation patterns and identifies memory bottlenecks.
"""

import tracemalloc
import torch
import psutil
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json
import sys

# Add the parent directory to Python path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the experiment modules
from src.experiment import experiment
from src.config import Config
from src.dataloader import load_data, SCA_Dataset
from src.trainer import trainer

class MemoryProfiler:
    """Memory-focused profiler for the experiment."""
    
    def __init__(self, config_template: Dict):
        self.config = Config(config_template)
        self.memory_snapshots = []
        self.phase_markers = []
        
    def take_memory_snapshot(self, phase_name: str):
        """Take a memory snapshot and record the current phase."""
        # Python memory
        current, peak = tracemalloc.get_traced_memory()
        
        # System memory
        process = psutil.Process()
        system_memory = process.memory_info().rss
        
        # GPU memory
        gpu_allocated = 0
        gpu_cached = 0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_cached = torch.cuda.memory_reserved()
        
        snapshot = {
            'phase': phase_name,
            'timestamp': time.time(),
            'python_current': current,
            'python_peak': peak,
            'system_memory': system_memory,
            'gpu_allocated': gpu_allocated,
            'gpu_cached': gpu_cached
        }
        
        self.memory_snapshots.append(snapshot)
        print(f"[{phase_name}] Memory: System={system_memory/1024/1024:.1f}MB, "
              f"GPU={gpu_allocated/1024/1024:.1f}MB, "
              f"Python={current/1024/1024:.1f}MB")
    
    def profile_memory_usage(self):
        """Profile memory usage during experiment execution."""
        print("Starting memory profiling...")
        
        # Start memory tracking
        tracemalloc.start()
        
        # Initial memory state
        self.take_memory_snapshot("initial")
        
        try:
            # Profile data loading phase
            self._profile_data_loading()
            
            # Profile training phase
            self._profile_training()
            
        finally:
            # Final memory state
            self.take_memory_snapshot("final")
            tracemalloc.stop()
            
            # Generate memory report
            self._generate_memory_report()
    
    def _profile_data_loading(self):
        """Profile memory usage during data loading."""
        print("\n=== Profiling Data Loading ===")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.take_memory_snapshot("before_data_loading")
        
        # Load data (this is the most memory-intensive part)
        (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test) = load_data(self.config, device)
        
        self.take_memory_snapshot("after_data_loading")
        
        # Create datasets
        train_dataset = SCA_Dataset(self.config, X_train, Y_train, P_train, K_train)
        val_dataset = SCA_Dataset(self.config, X_val, Y_val, P_val, K_val)
        test_dataset = SCA_Dataset(self.config, X_test, Y_test, P_test, K_test)
        
        self.take_memory_snapshot("after_dataset_creation")
        
        # Create dataloaders
        batch_size = self.config["batch_size"]
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        self.take_memory_snapshot("after_dataloader_creation")
        
        # Test one batch loading
        for batch in train_loader:
            self.take_memory_snapshot("after_first_batch")
            break
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_dataset': test_dataset
        }
    
    def _profile_training(self):
        """Profile memory usage during training (simplified version)."""
        print("\n=== Profiling Training Phase ===")
        
        self.take_memory_snapshot("before_training")
        
        # Run short experiment
        experiment(self.config)
        
        self.take_memory_snapshot("after_training")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.take_memory_snapshot("after_cleanup")
    
    def _generate_memory_report(self):
        """Generate comprehensive memory usage report."""
        # Create output directory
        profile_dir = f"./Memory_Profiling/expr_{self.config['expr_num']}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # Convert data for analysis
        phases = [snap['phase'] for snap in self.memory_snapshots]
        system_memory = [snap['system_memory'] / 1024 / 1024 for snap in self.memory_snapshots]  # MB
        gpu_memory = [snap['gpu_allocated'] / 1024 / 1024 for snap in self.memory_snapshots]  # MB
        python_memory = [snap['python_current'] / 1024 / 1024 for snap in self.memory_snapshots]  # MB
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        x_positions = range(len(phases))
        
        # System and GPU memory
        ax1.plot(x_positions, system_memory, 'b-o', label='System Memory', linewidth=2, markersize=6)
        if any(gpu_memory):
            ax1.plot(x_positions, gpu_memory, 'g-s', label='GPU Memory', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Execution Phase')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage by Execution Phase')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(phases, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Python memory tracking
        ax2.plot(x_positions, python_memory, 'r-^', label='Python Memory', linewidth=2, markersize=6)
        ax2.set_xlabel('Execution Phase')
        ax2.set_ylabel('Python Memory (MB)')
        ax2.set_title('Python Memory Allocation')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(phases, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(profile_dir, "memory_usage_by_phase.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_file = os.path.join(profile_dir, "memory_report.txt")
        with open(report_file, 'w') as f:
            f.write("Memory Profiling Report\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Memory Usage by Phase:\n")
            f.write("-" * 20 + "\n")
            for i, snap in enumerate(self.memory_snapshots):
                f.write(f"{snap['phase']:.<25} ")
                f.write(f"System: {snap['system_memory']/1024/1024:>8.1f} MB, ")
                f.write(f"GPU: {snap['gpu_allocated']/1024/1024:>8.1f} MB, ")
                f.write(f"Python: {snap['python_current']/1024/1024:>8.1f} MB\n")
            
            # Memory growth analysis
            f.write("\nMemory Growth Analysis:\n")
            f.write("-" * 23 + "\n")
            
            initial_system = self.memory_snapshots[0]['system_memory']
            peak_system = max(snap['system_memory'] for snap in self.memory_snapshots)
            f.write(f"Peak system memory: {peak_system/1024/1024:.1f} MB\n")
            f.write(f"Memory growth: {(peak_system - initial_system)/1024/1024:.1f} MB\n")
            
            if torch.cuda.is_available():
                initial_gpu = self.memory_snapshots[0]['gpu_allocated']
                peak_gpu = max(snap['gpu_allocated'] for snap in self.memory_snapshots)
                f.write(f"Peak GPU memory: {peak_gpu/1024/1024:.1f} MB\n")
                f.write(f"GPU memory growth: {(peak_gpu - initial_gpu)/1024/1024:.1f} MB\n")
            
            # Memory efficiency metrics
            f.write("\nMemory Efficiency Metrics:\n")
            f.write("-" * 26 + "\n")
            total_traces = self.config['train_size'] + self.config['val_size']
            memory_per_trace = peak_system / total_traces if total_traces > 0 else 0
            f.write(f"Memory per trace: {memory_per_trace:.2f} bytes\n")
            f.write(f"Total traces processed: {total_traces:,}\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            if peak_system/1024/1024/1024 > 16:  # > 16GB
                f.write("• Consider reducing batch size or data size for lower memory usage\n")
            if memory_per_trace > 1000:  # > 1KB per trace
                f.write("• Memory usage per trace is high - consider data optimization\n")
            if torch.cuda.is_available() and peak_gpu/1024/1024 > 8000:  # > 8GB GPU
                f.write("• High GPU memory usage - consider gradient checkpointing\n")
            else:
                f.write("• Memory usage appears reasonable for this dataset size\n")
        
        print(f"\nMemory profiling report saved to: {profile_dir}")
        print(f"Peak system memory: {peak_system/1024/1024:.1f} MB")
        if torch.cuda.is_available():
            print(f"Peak GPU memory: {max(snap['gpu_allocated'] for snap in self.memory_snapshots)/1024/1024:.1f} MB")


def main():
    """Main memory profiling function."""
    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())

    profiler = MemoryProfiler(config_template)
    profiler.profile_memory_usage()


if __name__ == "__main__":
    main()
