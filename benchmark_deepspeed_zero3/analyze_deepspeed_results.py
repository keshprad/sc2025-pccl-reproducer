#!/usr/bin/env python3
"""
Analysis script for DeepSpeed ZeRO-3 benchmark results
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd

def load_results(results_dir: str) -> Dict:
    """Load all benchmark result files from directory"""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and 'deepspeed_zero3' in filename:
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                model_size = data['model_size']
                results[model_size] = data
    
    return results

def print_summary(results: Dict):
    """Print summary statistics"""
    print("=" * 80)
    print("DEEPSPEED ZERO-3 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    for model_size, data in results.items():
        print(f"\n{model_size} Model Results:")
        print("-" * 40)
        print(f"Global batch size: {data['global_batch_size']:,} samples ({data['global_batch_size_tokens']:,} tokens)")
        print(f"Sequence length: {data['seq_len']}")
        print(f"World size: {data['world_size']}")
        print(f"Local batch size: {data['local_batch_size']}")
        print(f"Number of trials: {data['num_trials']}")
        print(f"Batches per trial: {data['num_batches']} (warmup: {data['warmup_batches']})")
        print()
        print(f"Average throughput: {data['avg_samples_throughput']:.2f} ± {data['samples_throughput_std']:.2f} samples/sec")
        print(f"Average throughput: {data['avg_tokens_throughput']:.2f} ± {data['tokens_throughput_std']:.2f} tokens/sec")
        print(f"Average batch time: {data['avg_batch_time']:.3f} seconds")
        
        # Calculate tokens per second per parameter
        model_params = {'7B': 7e9, '13B': 13e9}
        if model_size in model_params:
            tokens_per_param_per_sec = data['avg_tokens_throughput'] / model_params[model_size]
            print(f"Tokens/param/sec: {tokens_per_param_per_sec:.2e}")

def create_plots(results: Dict, output_dir: str):
    """Create visualization plots"""
    
    # Extract data for plotting
    model_sizes = list(results.keys())
    samples_throughput = [results[size]['avg_samples_throughput'] for size in model_sizes]
    samples_std = [results[size]['samples_throughput_std'] for size in model_sizes] 
    tokens_throughput = [results[size]['avg_tokens_throughput'] for size in model_sizes]
    tokens_std = [results[size]['tokens_throughput_std'] for size in model_sizes]
    batch_times = [results[size]['avg_batch_time'] for size in model_sizes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Samples throughput
    x_pos = np.arange(len(model_sizes))
    ax1.bar(x_pos, samples_throughput, yerr=samples_std, capsize=5, 
            color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Samples/Second')
    ax1.set_title('Average Samples Throughput')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_sizes)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (val, std) in enumerate(zip(samples_throughput, samples_std)):
        ax1.text(i, val + std + max(samples_throughput) * 0.01, 
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # Tokens throughput
    ax2.bar(x_pos, tokens_throughput, yerr=tokens_std, capsize=5,
            color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Tokens/Second')
    ax2.set_title('Average Tokens Throughput')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_sizes)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (val, std) in enumerate(zip(tokens_throughput, tokens_std)):
        ax2.text(i, val + std + max(tokens_throughput) * 0.01,
                f'{val:.0f}±{std:.0f}', ha='center', va='bottom')
    
    # Batch times
    ax3.bar(x_pos, batch_times, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax3.set_xlabel('Model Size')
    ax3.set_ylabel('Seconds')
    ax3.set_title('Average Batch Time')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_sizes)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, val in enumerate(batch_times):
        ax3.text(i, val + max(batch_times) * 0.01, f'{val:.3f}s', 
                ha='center', va='bottom')
    
    # Efficiency comparison (tokens per parameter per second)
    model_params = {'7B': 7e9, '13B': 13e9}
    efficiency = []
    efficiency_labels = []
    
    for size in model_sizes:
        if size in model_params:
            eff = results[size]['avg_tokens_throughput'] / model_params[size]
            efficiency.append(eff)
            efficiency_labels.append(size)
    
    if efficiency:
        x_eff = np.arange(len(efficiency_labels))
        ax4.bar(x_eff, efficiency, color='gold', alpha=0.7, edgecolor='orange')
        ax4.set_xlabel('Model Size')  
        ax4.set_ylabel('Tokens/Parameter/Second')
        ax4.set_title('Model Efficiency (Tokens per Parameter per Second)')
        ax4.set_xticks(x_eff)
        ax4.set_xticklabels(efficiency_labels)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, val in enumerate(efficiency):
            ax4.text(i, val + max(efficiency) * 0.01, f'{val:.2e}',
                    ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No efficiency data available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'deepspeed_zero3_benchmark_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    # Also create a detailed trial comparison plot
    create_trial_plots(results, output_dir)

def create_trial_plots(results: Dict, output_dir: str):
    """Create detailed plots showing individual trial results"""
    
    for model_size, data in results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract trial data
        trials = data['all_trials']
        trial_numbers = [t['trial'] + 1 for t in trials]
        trial_samples = [t['avg_samples_throughput'] for t in trials]
        trial_tokens = [t['avg_tokens_throughput'] for t in trials]
        
        # Samples throughput by trial
        ax1.plot(trial_numbers, trial_samples, 'o-', linewidth=2, markersize=8,
                label='Individual Trials')
        ax1.axhline(y=np.mean(trial_samples), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(trial_samples):.2f}')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Samples/Second')
        ax1.set_title(f'{model_size} Model - Samples Throughput by Trial')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Tokens throughput by trial
        ax2.plot(trial_numbers, trial_tokens, 'o-', linewidth=2, markersize=8,
                color='orange', label='Individual Trials')
        ax2.axhline(y=np.mean(trial_tokens), color='red', linestyle='--',
                   label=f'Mean: {np.mean(trial_tokens):.0f}')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Tokens/Second')
        ax2.set_title(f'{model_size} Model - Tokens Throughput by Trial')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save trial plot
        trial_plot_file = os.path.join(output_dir, f'deepspeed_zero3_{model_size}_trials.png')
        plt.savefig(trial_plot_file, dpi=300, bbox_inches='tight')
        print(f"Trial plot for {model_size} saved to: {trial_plot_file}")
        plt.close()

def create_csv_summary(results: Dict, output_dir: str):
    """Create CSV summary of results"""
    
    summary_data = []
    for model_size, data in results.items():
        summary_data.append({
            'model_size': model_size,
            'global_batch_size': data['global_batch_size'],
            'global_batch_size_tokens': data['global_batch_size_tokens'],
            'seq_len': data['seq_len'],
            'world_size': data['world_size'],
            'local_batch_size': data['local_batch_size'],
            'num_trials': data['num_trials'],
            'avg_samples_throughput': data['avg_samples_throughput'],
            'samples_throughput_std': data['samples_throughput_std'],
            'avg_tokens_throughput': data['avg_tokens_throughput'],
            'tokens_throughput_std': data['tokens_throughput_std'],
            'avg_batch_time': data['avg_batch_time']
        })
    
    df = pd.DataFrame(summary_data)
    csv_file = os.path.join(output_dir, 'deepspeed_zero3_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f"CSV summary saved to: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze DeepSpeed ZeRO-3 benchmark results')
    parser.add_argument('results_dir', type=str, help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots (default: same as results_dir)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return
    
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print(f"Error: No benchmark result files found in {args.results_dir}")
        return
    
    # Print summary
    print_summary(results)
    
    # Create plots
    try:
        create_plots(results, output_dir)
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
    
    # Create CSV summary
    try:
        create_csv_summary(results, output_dir)
    except ImportError:
        print("Warning: pandas not available, skipping CSV summary")

if __name__ == "__main__":
    main()
