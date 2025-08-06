# CHES 2025 Experiment Profiling Suite (AI Generated)

This profiling suite provides comprehensive performance analysis tools for the CHES 2025 PyTorch experiment. It includes multiple profiling approaches to help you understand performance bottlenecks, memory usage patterns, and optimization opportunities.

## Available Profiling Tools

### 1. Comprehensive Profiler (`profile_experiment.py`)
- **CPU usage monitoring** - Real-time CPU utilization tracking
- **Memory usage analysis** - System and GPU memory consumption
- **Execution time profiling** - Function-level timing analysis with cProfile
- **Resource visualization** - Interactive plots of resource usage over time
- **Performance recommendations** - Automated suggestions for optimization

### 2. Memory Profiler (`memory_profiler.py`)
- **Memory allocation tracking** - Detailed memory usage by execution phase
- **Memory growth analysis** - Identify memory leaks and inefficient patterns
- **Phase-based profiling** - Memory usage during data loading, training, etc.
- **Memory efficiency metrics** - Memory usage per trace and optimization suggestions

### 3. GPU Profiler (`gpu_profiler.py`)
- **CUDA operation profiling** - Detailed GPU kernel analysis
- **GPU memory tracking** - Allocated vs reserved memory patterns
- **Memory efficiency analysis** - GPU memory utilization optimization
- **CUDA trace export** - Chrome trace format for detailed analysis

### 4. Quick Profiler (`run_profiling.py --quick`)
- **Fast cProfile analysis** - Quick function-level timing
- **Automated analysis** - Top functions by time and call count
- **Minimal overhead** - Suitable for quick performance checks

## Quick Start

### Run All Profiling Types
```bash
python run_profiling.py --all
```

### Run Specific Profiling
```bash
# Comprehensive profiling (recommended for first-time analysis)
python run_profiling.py --comprehensive

# Memory-focused analysis
python run_profiling.py --memory

# GPU-specific profiling
python run_profiling.py --gpu

# Quick cProfile analysis
python run_profiling.py --quick

# Baseline experiment (no profiling overhead)
python run_profiling.py --baseline
```

### Manual Execution
```bash
# Run individual profilers directly
python profile_experiment.py       # Comprehensive profiling
python memory_profiler.py          # Memory profiling
python gpu_profiler.py            # GPU profiling
```

## Output Structure

The profiling tools create organized output directories:

```
├── Profiling_Results/expr_X/
│   ├── profiling_summary.json      # Comprehensive metrics
│   ├── profiling_summary.txt       # Human-readable report
│   ├── profile_stats.prof          # Raw cProfile data
│   ├── profile_report.txt          # Function timing analysis
│   └── resource_usage.png          # Resource usage plots
│
├── Memory_Profiling/expr_X/
│   ├── memory_report.txt           # Memory analysis report
│   └── memory_usage_by_phase.png   # Memory usage visualization
│
└── GPU_Profiling/expr_X/
    ├── gpu_report.txt              # GPU analysis report
    ├── gpu_memory_usage.png        # GPU memory plots
    ├── gpu_trace.json              # Chrome trace (open in chrome://tracing)
    └── gpu_summary.txt             # GPU operation summary
```

## Key Metrics Analyzed

### Performance Metrics
- **Total execution time** - Overall experiment duration
- **Traces per second** - Training throughput
- **CPU utilization** - Processor efficiency
- **Memory efficiency** - Memory usage per training sample

### Memory Metrics
- **Peak memory usage** - Maximum system memory consumption
- **Memory growth patterns** - How memory usage evolves during execution
- **GPU memory utilization** - CUDA memory allocation efficiency
- **Memory per trace** - Data efficiency analysis

### GPU Metrics (CUDA environments)
- **GPU memory utilization** - Percentage of GPU memory used
- **Memory allocation efficiency** - Allocated vs reserved memory ratio
- **CUDA kernel performance** - Detailed GPU operation timing
- **Memory transfer patterns** - Host-to-device transfer analysis

## Interpreting Results

### Comprehensive Profiling Results
The `profiling_summary.json` file contains key metrics:
- `total_execution_time`: Total experiment duration
- `peak_memory_usage_mb`: Maximum system memory used
- `traces_per_second`: Training throughput
- `recommendations`: Automated optimization suggestions

### Memory Profiling Results
Look for:
- **Memory growth during data loading**: Should be one-time spike
- **Training memory stability**: Memory usage should remain stable during training
- **Peak memory vs dataset size**: Indicates data loading efficiency

### GPU Profiling Results
Key indicators:
- **GPU memory utilization > 70%**: Good GPU memory usage
- **Memory efficiency > 80%**: Efficient allocation patterns
- **Chrome trace analysis**: Open `gpu_trace.json` in Chrome's `chrome://tracing`

## Optimization Recommendations

Based on profiling results, consider these optimizations:

### High Memory Usage
- Reduce `batch_size` in configuration
- Use gradient accumulation for effective larger batches
- Enable mixed precision training (`torch.cuda.amp`)

### Low CPU Utilization
- Increase `num_workers` in DataLoader
- Enable data prefetching with `pin_memory=True`
- Use faster data loading formats (e.g., preprocessed tensors)

### Low GPU Utilization
- Increase `batch_size` if memory allows
- Use multiple GPUs with `DataParallel` or `DistributedDataParallel`
- Optimize data transfer with `non_blocking=True`

### Slow Training
- Enable mixed precision training
- Use compiled models with `torch.compile()` (PyTorch 2.0+)
- Optimize learning rate schedule for faster convergence

## Troubleshooting

### Common Issues

**"CUDA not available" during GPU profiling**
- GPU profiling requires a CUDA-enabled environment
- Run other profiling types on CPU-only systems

**"Out of memory" errors during profiling**
- Reduce dataset size in profiling scripts
- Use memory profiler with smaller batch sizes
- Monitor system memory before running

**Missing dependencies**
```bash
pip install matplotlib psutil torch tensorboard tqdm
```

**Permission errors on Linux/Mac**
```bash
chmod +x *.py
```

## Advanced Usage

### Custom Configuration
Modify the `config_template` in each profiler to test different configurations:
```python
config_template = {
    "batch_size": 128,        # Reduce for memory profiling
    "num_steps": 10,          # Reduce for quick testing
    "train_size": 50000,      # Reduce dataset size
    # ... other parameters
}
```

### Analyzing Chrome Traces
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load the `gpu_trace.json` file
4. Analyze GPU kernel timing and memory transfers

### Continuous Profiling
For production monitoring, integrate profiling into your training loop:
```python
# Add to training script
import psutil
process = psutil.Process()

# Log metrics periodically
if step % 100 == 0:
    memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Step {step}: Memory={memory_mb:.1f}MB, GPU={gpu_memory_mb:.1f}MB")
```

## Contributing

To add new profiling capabilities:
1. Create a new profiler class following existing patterns
2. Add visualization and reporting functions
3. Integrate with `run_profiling.py`
4. Update this README with new features

## Requirements

- Python 3.8+
- PyTorch 1.12+
- matplotlib
- psutil
- numpy
- tqdm
- tensorboard (for experiment logging)

For GPU profiling:
- CUDA-enabled PyTorch
- NVIDIA GPU with CUDA support



# Chrome Flame Graph Viewing Guide

This guide will help you view PyTorch experiment performance analysis results in Chrome browser, including flame graphs and other visualization analysis.

## Method 1: Chrome Tracing (Recommended for GPU Analysis)

### 1.1 Generate Chrome Trace File

First run GPU profiling to generate trace file:

```bash
python run_profiling.py --gpu
```

Or run directly:
```bash
python gpu_profiler.py
```

This will generate `gpu_trace.json` file in the `./GPU_Profiling/expr_X/` directory.

### 1.2 Open Trace in Chrome

1. **Open Chrome browser**

2. **Visit Chrome Tracing page**:
   - Enter in address bar: `chrome://tracing`
   - Or visit: `about://tracing`

3. **Load trace file**:
   - Click "Load" button
   - Select the generated `gpu_trace.json` file
   - Or directly drag the file to the page

4. **View analysis results**:
   - Use WASD keys or mouse for navigation
   - Click events to view detailed information
   - Use search function to find specific operations

### 1.3 Chrome Tracing Interface Description

- **Timeline**: Shows the chronological order of operations
- **Process/Thread**: Different rows show different processing units
- **Event blocks**: Each colored block represents an operation
- **Flame Graph mode**: Shows by call stack hierarchy

## Method 2: Using Dedicated Flame Graph Tools

### 2.1 Install py-spy (Recommended)

```bash
# Install in virtual environment
pip install py-spy
```

### 2.2 Generate Flame Graph

```bash
# Install py-spy
pip install py-spy

# Run flame graph profiling
python flame_graph_profiler.py
```

Or use our created interactive runner:
```bash
python run_flamegraph.py
```

### 2.3 View Flame Graph

Generated files can be viewed in Chrome:

1. **SVG Flame Graph**:
   - File location: `./Flame_Graphs/expr_X/flame_graph.svg`
   - Directly drag to Chrome browser window
   - Can click to zoom, hover to view details

2. **Speedscope Profile**:
   - File location: `./Flame_Graphs/expr_X/speedscope_profile_XXs.json`
   - Visit: https://www.speedscope.app/
   - Upload JSON file to view interactive analysis

## Method 3: Using cProfile + snakeviz

### 3.1 Install snakeviz

```bash
pip install snakeviz
```

### 3.2 Generate and View Profile

```bash
# Run profiling
python run_profiling.py --quick

# View with snakeviz
snakeviz quick_profile.prof
```

This will open an interactive flame graph in the browser.

## Method 4: Integration with Existing Profiling Tools

### 4.1 Update Existing Profiler

I've added flame graph support to the profiling suite. Run:

```bash
# Comprehensive analysis including flame graph
python run_profiling.py --all

# Or dedicated flame graph analysis
python flame_graph_profiler.py
```

## Navigation and Analysis Tips in Chrome

### Chrome Tracing Interface Operations

1. **Navigation Controls**:
   - `W/S`: Zoom in/out timeline
   - `A/D`: Move timeline left/right
   - Mouse wheel: Zoom
   - Drag: Pan

2. **Search and Filter**:
   - Press `Ctrl+F` to search for specific operations
   - Click events to view detailed information
   - Use right panel to view statistics

3. **Analysis Key Points**:
   - **Long bars**: Time-consuming operations
   - **Dense areas**: High-frequency operations
   - **Gaps**: Wait time
   - **Colors**: Different types of operations

### Flame Graph Analysis Key Points

1. **Width**: Function execution time ratio
2. **Height**: Call stack depth
3. **Colors**: Usually represent different modules or hotness
4. **Click**: Focus on specific function
5. **Search**: Find specific function names

## Practical Usage Examples

### Quick Start

1. **Generate all types of analysis**:
```bash
python run_profiling.py --all
python flame_graph_profiler.py
```

2. **View in Chrome**:
```bash
# GPU timeline
# Open chrome://tracing
# Load ./GPU_Profiling/expr_X/gpu_trace.json

# Flame graph
# Drag ./Flame_Graphs/expr_X/flame_graph.svg to Chrome

# Interactive analysis
# Visit https://www.speedscope.app/
# Upload speedscope_profile_XXs.json
```

3. **Analyze performance bottlenecks**:
   - Find widest areas in flame graph
   - Find longest operations in Chrome tracing
   - Compare CPU and GPU utilization

## Configure Custom Analysis

### Modify Analysis Parameters

In `flame_graph_profiler.py` you can adjust:

```python
config_template = {
    "train_size": 50000,    # Reduce data size for quick analysis
    "num_steps": 10,        # Reduce training steps
    "batch_size": 128,      # Adjust batch size
    # ... other parameters
}
```

### Analyze Specific Phases

You can add profiling markers in code:

```python
# Add in training loop
with torch.profiler.profile() as prof:
    # Training code
    pass

prof.export_chrome_trace("training_phase.json")
```

## Common Issues Resolution

### py-spy Permission Issues

On Linux may need:
```bash
sudo sysctl kernel.yama.ptrace_scope=0
```

Or run with sudo:
```bash
sudo python flame_graph_profiler.py
```

### Chrome Tracing File Too Large

If trace file is too large for Chrome to open:
1. Reduce analysis time
2. Use `--trace-level` to limit detail level
3. Analyze different parts in phases

Now you can view flame graphs and performance analysis results in Chrome through multiple methods!
