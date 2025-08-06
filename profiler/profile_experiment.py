#!/usr/bin/env python3
"""
Comprehensive profiling script for the CHES 2025 PyTorch experiment.
This script profiles CPU usage, memory consumption, GPU utilization, and execution time.
"""

import cProfile
import pstats
import psutil
import time
import tracemalloc
import torch
import threading
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import json
import os
from datetime import datetime
import sys

# Add the parent directory to Python path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the experiment modules
from src.experiment import experiment
from src.config import Config

class ExperimentProfiler:
    """Comprehensive profiler for the CHES experiment."""
    
    def __init__(self, config_template: Dict[str, Any]):
        self.config = Config(config_template)
        self.profiling_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'timestamps': [],
            'execution_phases': [],
            'total_time': 0,
            'peak_memory': 0,
            'peak_gpu_memory': 0
        }
        self.monitoring = False
        self.monitor_thread = None
        
    def _monitor_resources(self):
        """Monitor CPU, memory, and GPU usage in a separate thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and system memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                
                # GPU memory if available
                gpu_memory_mb = 0
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Record data
                self.profiling_data['cpu_usage'].append(cpu_percent)
                self.profiling_data['memory_usage'].append(memory_mb)
                self.profiling_data['gpu_memory'].append(gpu_memory_mb)
                self.profiling_data['timestamps'].append(time.time())
                
                # Update peaks
                self.profiling_data['peak_memory'] = max(self.profiling_data['peak_memory'], memory_mb)
                self.profiling_data['peak_gpu_memory'] = max(self.profiling_data['peak_gpu_memory'], gpu_memory_mb)
                
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                break
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def profile_experiment(self):
        """Run the experiment with comprehensive profiling."""
        print("Starting comprehensive profiling of CHES 2025 experiment...")
        print(f"Configuration: expr_num={self.config['expr_num']}, model={self.config['model']}")
        
        # Start memory tracking
        tracemalloc.start()
        
        # Start resource monitoring
        self.start_monitoring()
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        
        start_time = time.time()
        
        try:
            print("\n=== Starting experiment execution ===")
            profiler.enable()
            
            # Run the actual experiment
            experiment(self.config)
            
            profiler.disable()
            end_time = time.time()
            
            self.profiling_data['total_time'] = end_time - start_time
            
        except Exception as e:
            print(f"Error during experiment execution: {e}")
            raise
        finally:
            # Stop monitoring
            self.stop_monitoring()
            
            # Get memory peak
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"\n=== Profiling Results ===")
            print(f"Total execution time: {self.profiling_data['total_time']:.2f} seconds")
            print(f"Peak memory usage: {self.profiling_data['peak_memory']:.2f} MB")
            print(f"Peak GPU memory: {self.profiling_data['peak_gpu_memory']:.2f} MB")
            print(f"Peak traced memory: {peak / 1024 / 1024:.2f} MB")
            
            # Save profiling statistics
            self._save_profiling_stats(profiler)
            self._create_visualizations()
            self._save_summary_report()
    
    def _save_profiling_stats(self, profiler: cProfile.Profile):
        """Save detailed profiling statistics."""
        # Create profiling output directory
        profile_dir = f"./Profiling_Results/expr_{self.config['expr_num']}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # Save raw profile data
        profile_file = os.path.join(profile_dir, "profile_stats.prof")
        profiler.dump_stats(profile_file)
        
        # Save human-readable statistics
        stats_file = os.path.join(profile_dir, "profile_report.txt")
        with open(stats_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            f.write("=== Top 20 functions by cumulative time ===\n")
            stats.sort_stats('cumulative').print_stats(20)
            f.write("\n=== Top 20 functions by total time ===\n")
            stats.sort_stats('tottime').print_stats(20)
            f.write("\n=== Functions with most calls ===\n")
            stats.sort_stats('ncalls').print_stats(20)
        
        print(f"Detailed profiling stats saved to: {profile_dir}")
    
    def _create_visualizations(self):
        """Create visualization plots for profiling data."""
        if not self.profiling_data['timestamps']:
            print("No monitoring data available for visualization")
            return
            
        profile_dir = f"./Profiling_Results/expr_{self.config['expr_num']}"
        
        # Convert timestamps to relative time
        start_time = self.profiling_data['timestamps'][0]
        relative_times = [(t - start_time) for t in self.profiling_data['timestamps']]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # CPU usage plot
        axes[0].plot(relative_times, self.profiling_data['cpu_usage'], 'b-', linewidth=2)
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('Resource Usage During Experiment')
        axes[0].grid(True, alpha=0.3)
        
        # Memory usage plot
        axes[1].plot(relative_times, self.profiling_data['memory_usage'], 'r-', linewidth=2, label='System Memory')
        if any(self.profiling_data['gpu_memory']):
            axes[1].plot(relative_times, self.profiling_data['gpu_memory'], 'g-', linewidth=2, label='GPU Memory')
        axes[1].set_ylabel('Memory Usage (MB)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Combined overview
        ax2_twin = axes[2].twinx()
        line1 = axes[2].plot(relative_times, self.profiling_data['cpu_usage'], 'b-', label='CPU %')
        line2 = ax2_twin.plot(relative_times, self.profiling_data['memory_usage'], 'r-', label='Memory (MB)')
        
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('CPU Usage (%)', color='b')
        ax2_twin.set_ylabel('Memory Usage (MB)', color='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(profile_dir, "resource_usage.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Resource usage plots saved to: {plot_file}")
    
    def _save_summary_report(self):
        """Save a comprehensive summary report."""
        profile_dir = f"./Profiling_Results/expr_{self.config['expr_num']}"
        
        # Calculate statistics
        avg_cpu = np.mean(self.profiling_data['cpu_usage']) if self.profiling_data['cpu_usage'] else 0
        avg_memory = np.mean(self.profiling_data['memory_usage']) if self.profiling_data['memory_usage'] else 0
        avg_gpu_memory = np.mean(self.profiling_data['gpu_memory']) if self.profiling_data['gpu_memory'] else 0
        
        # System information
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'total_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        }
        
        # Experiment configuration
        config_summary = {
            'model_type': self.config['model'],
            'train_size': self.config['train_size'],
            'val_size': self.config['val_size'],
            'test_size': self.config['test_size'],
            'batch_size': self.config['batch_size'],
            'num_epochs': self.config['num_epochs'],
            'learning_rate': self.config['lr'],
            'model_args': self.config['model_args']
        }
        
        # Performance summary
        performance_summary = {
            'total_execution_time': self.profiling_data['total_time'],
            'peak_memory_usage_mb': self.profiling_data['peak_memory'],
            'peak_gpu_memory_mb': self.profiling_data['peak_gpu_memory'],
            'average_cpu_usage_percent': avg_cpu,
            'average_memory_usage_mb': avg_memory,
            'average_gpu_memory_mb': avg_gpu_memory,
            'traces_per_second': (self.config['train_size'] + self.config['val_size']) / self.profiling_data['total_time'] if self.profiling_data['total_time'] > 0 else 0,
            'memory_efficiency_mb_per_trace': self.profiling_data['peak_memory'] / self.config['train_size'] if self.config['train_size'] > 0 else 0
        }
        
        # Create comprehensive report
        report = {
            'profiling_timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'experiment_config': config_summary,
            'performance_metrics': performance_summary,
            'recommendations': self._generate_recommendations(performance_summary)
        }
        
        # Save as JSON
        report_file = os.path.join(profile_dir, "profiling_summary.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as human-readable text
        text_report_file = os.path.join(profile_dir, "profiling_summary.txt")
        with open(text_report_file, 'w') as f:
            f.write("CHES 2025 Experiment Profiling Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['profiling_timestamp']}\n\n")
            
            f.write("System Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in system_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nExperiment Configuration:\n")
            f.write("-" * 25 + "\n")
            for key, value in config_summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nPerformance Metrics:\n")
            f.write("-" * 20 + "\n")
            for key, value in performance_summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"Profiling summary saved to: {profile_dir}")
    
    def _generate_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        # Memory recommendations
        if performance['peak_memory_usage_mb'] > 8000:  # > 8GB
            recommendations.append("High memory usage detected. Consider reducing batch size or using gradient accumulation.")
        
        # CPU recommendations
        if performance['average_cpu_usage_percent'] < 50:
            recommendations.append("Low CPU utilization. Consider increasing num_workers in DataLoader or using data prefetching.")
        
        # GPU recommendations
        if torch.cuda.is_available() and performance['average_gpu_memory_mb'] < 2000:
            recommendations.append("GPU memory underutilized. Consider increasing batch size for better GPU efficiency.")
        
        # Training efficiency
        if performance['traces_per_second'] < 1000:
            recommendations.append("Low training throughput. Consider optimizing data loading or using mixed precision training.")
        
        # General recommendations
        if performance['total_execution_time'] > 3600:  # > 1 hour
            recommendations.append("Long execution time. Consider using learning rate scheduling or early stopping.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major optimizations needed.")
        
        return recommendations


def main():
    """Main profiling function."""
    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())
    
    # Create and run profiler
    profiler = ExperimentProfiler(config_template)
    profiler.profile_experiment()
    
    print("\n=== Profiling completed! ===")
    print("Check the './Profiling_Results/' directory for detailed results.")


if __name__ == "__main__":
    main()
