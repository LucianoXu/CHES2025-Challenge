#!/usr/bin/env python3
"""
Flame graph profiling using py-spy for the CHES 2025 experiment.
Generates interactive flame graphs that can be viewed in Chrome.
"""

import subprocess
import time
import os
import sys
import multiprocessing
from typing import Optional

# Add the parent directory to Python path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.experiment import experiment
from src.config import Config
import json


class FlameGraphProfiler:
    """Flame graph profiler using py-spy."""
    
    def __init__(self, config_template: dict):
        self.config = Config(config_template)
        self.output_dir = f"./Flame_Graphs/expr_{self.config['expr_num']}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_py_spy(self) -> bool:
        """Check if py-spy is available."""
        try:
            result = subprocess.run(['py-spy', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ py-spy found: {result.stdout.strip()}")
                return True
            else:
                return False
        except FileNotFoundError:
            return False
    
    def install_py_spy(self) -> bool:
        """Try to install py-spy."""
        print("Installing py-spy...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'py-spy'], 
                          check=True)
            return self.check_py_spy()
        except subprocess.CalledProcessError as e:
            print(f"Failed to install py-spy: {e}")
            return False
    
    def run_experiment_with_flamegraph(self):
        """Run experiment and generate flame graph simultaneously."""
        if not self.check_py_spy():
            print("py-spy not found. Attempting to install...")
            if not self.install_py_spy():
                print("Failed to install py-spy. Please install manually:")
                print("  pip install py-spy")
                return False
        
        print("Starting flame graph profiling...")
        print(f"Output directory: {self.output_dir}")
        
        # Start the experiment in a separate process
        def run_experiment():
            experiment(self.config)
        
        process = multiprocessing.Process(target=run_experiment)
        process.start()
        
        # Wait a moment for the process to start
        time.sleep(2)
        
        try:
            # Generate flame graph
            flame_graph_file = os.path.join(self.output_dir, "flame_graph.svg")
            py_spy_cmd = [
                'py-spy', 'record',
                '-o', flame_graph_file,
                '-d', '60',  # Duration in seconds
                '-p', str(process.pid)
            ]
            
            print(f"Running py-spy: {' '.join(py_spy_cmd)}")
            subprocess.run(py_spy_cmd, check=True)
            
            print(f"✓ Flame graph saved to: {flame_graph_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"py-spy failed: {e}")
        except KeyboardInterrupt:
            print("Profiling interrupted by user")
        finally:
            # Wait for experiment to finish or terminate it
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        
        return True
    
    def generate_cpu_profile(self, duration: int = 30):
        """Generate CPU profile for a specific duration."""
        if not self.check_py_spy():
            if not self.install_py_spy():
                return False
        
        print(f"Starting CPU profiling for {duration} seconds...")
        
        def run_experiment():
            experiment(self.config)
        
        process = multiprocessing.Process(target=run_experiment)
        process.start()
        time.sleep(1)  # Wait for process to start
        
        try:
            # Generate flame graph
            flame_graph_file = os.path.join(self.output_dir, f"cpu_profile_{duration}s.svg")
            py_spy_cmd = [
                'py-spy', 'record',
                '-o', flame_graph_file,
                '-d', str(duration),
                '-f', 'flamegraph',
                '-p', str(process.pid)
            ]
            
            subprocess.run(py_spy_cmd, check=True)
            print(f"✓ CPU flame graph saved to: {flame_graph_file}")
            
            # Also generate speedscope format for interactive viewing
            speedscope_file = os.path.join(self.output_dir, f"speedscope_profile_{duration}s.json")
            speedscope_cmd = [
                'py-spy', 'record',
                '-o', speedscope_file,
                '-d', str(duration),
                '-f', 'speedscope',
                '-p', str(process.pid)
            ]
            
            subprocess.run(speedscope_cmd, check=True)
            print(f"✓ Speedscope profile saved to: {speedscope_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"Profiling failed: {e}")
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        
        return True
    
    def create_viewing_instructions(self):
        """Create instructions for viewing the generated profiles."""
        instructions_file = os.path.join(self.output_dir, "viewing_instructions.txt")
        
        with open(instructions_file, 'w') as f:
            f.write("Flame Graph Viewing Instructions\n")
            f.write("=" * 35 + "\n\n")
            
            f.write("1. SVG Flame Graphs (*.svg files):\n")
            f.write("   - Open directly in Chrome browser\n")
            f.write("   - Drag and drop the .svg file into Chrome\n")
            f.write("   - Interactive: click to zoom, hover for details\n\n")
            
            f.write("2. Speedscope Profiles (*.json files):\n")
            f.write("   - Visit: https://www.speedscope.app/\n")
            f.write("   - Upload the .json file\n")
            f.write("   - Interactive flame graph with multiple views\n\n")
            
            f.write("3. Chrome Tracing (gpu_trace.json):\n")
            f.write("   - Open Chrome and go to: chrome://tracing\n")
            f.write("   - Load the gpu_trace.json file\n")
            f.write("   - View GPU operations timeline\n\n")
            
            f.write("Generated Files in this directory:\n")
            for file in os.listdir(self.output_dir):
                if file.endswith(('.svg', '.json')):
                    f.write(f"   - {file}\n")
        
        print(f"✓ Viewing instructions saved to: {instructions_file}")

def main():
    """Main function."""
    # Use configuration from the JSON file
    with open("expr_config.json", "r") as f:
        config_template = json.loads(f.read())
    
    profiler = FlameGraphProfiler(config_template)
    
    print("Generating flame graph for CHES 2025 experiment...")
    profiler.generate_cpu_profile(duration=30)
    profiler.create_viewing_instructions()

if __name__ == "__main__":
    main()
