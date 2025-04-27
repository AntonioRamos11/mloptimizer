#!/usr/bin/env python3
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from app.common.hardware_performance_logger import HardwarePerformanceLogger

def find_log_files(log_dir, experiment_id=None):
    """Find all hardware performance log files"""
    log_files = []
    
    if experiment_id:
        search_dir = os.path.join(log_dir, experiment_id)
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.endswith('.json'):
                    log_files.append(os.path.join(search_dir, file))
    else:
        # Search all experiment directories
        for exp_dir in os.listdir(log_dir):
            exp_path = os.path.join(log_dir, exp_dir)
            if os.path.isdir(exp_path):
                for file in os.listdir(exp_path):
                    if file.endswith('.json'):
                        log_files.append(os.path.join(exp_path, file))
    
    return log_files

def display_comparison_table(comparison_data):
    """Display a formatted comparison table"""
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(comparison_data)
    
    # Format columns for better display
    df['build_time_ms'] = df['build_time_ms'].apply(lambda x: f"{x:.1f} ms")
    df['train_time_ms'] = df['train_time_ms'].apply(lambda x: f"{x:.1f} ms")
    df['time_per_epoch_ms'] = df['time_per_epoch_ms'].apply(lambda x: f"{x:.1f} ms")
    df['parameters'] = df['parameters'].apply(lambda x: f"{x:,}")
    if 'final_accuracy' in df.columns:
        df['final_accuracy'] = df['final_accuracy'].apply(lambda x: f"{x*100:.2f}%" if x is not None else "N/A")
    
    # Select and rename columns for display
    display_df = df[[
        'model_id', 'hardware', 'parameters', 
        'build_time_ms', 'train_time_ms', 'time_per_epoch_ms', 
        'epochs_completed', 'final_loss', 'final_accuracy'
    ]]
    
    display_df.columns = [
        'Model ID', 'Hardware', 'Parameters', 
        'Build Time', 'Total Train Time', 'Time/Epoch', 
        'Epochs', 'Final Loss', 'Final Accuracy'
    ]
    
    print(tabulate(display_df, headers='keys', tablefmt='pretty'))

def plot_comparison(comparison_data, metric='train_time_ms'):
    """Plot a comparison chart of the specified metric"""
    df = pd.DataFrame(comparison_data)
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    
    metric_name_map = {
        'train_time_ms': 'Training Time (ms)',
        'time_per_epoch_ms': 'Time per Epoch (ms)',
        'build_time_ms': 'Model Build Time (ms)'
    }
    
    y_label = metric_name_map.get(metric, metric)
    
    # Sort by the selected metric
    df = df.sort_values(by=metric)
    
    # Create labels combining model ID and hardware
    labels = [f"{row['model_id']}\n{row['hardware']}" for _, row in df.iterrows()]
    
    plt.bar(labels, df[metric])
    plt.ylabel(y_label)
    plt.title(f'Hardware Performance Comparison - {y_label}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(f'hardware_comparison_{metric}.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare hardware performance logs')
    parser.add_argument('--log-dir', default='hardware_performance_logs', 
                        help='Directory containing hardware performance logs')
    parser.add_argument('--experiment-id', help='Filter logs by experiment ID')
    parser.add_argument('--metric', default='train_time_ms',
                        choices=['train_time_ms', 'build_time_ms', 'time_per_epoch_ms'],
                        help='Metric to use for comparison')
    parser.add_argument('--no-plot', action='store_true', 
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    log_files = find_log_files(args.log_dir, args.experiment_id)
    
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return
    
    comparison_data = HardwarePerformanceLogger.compare_logs(log_files)
    display_comparison_table(comparison_data)
    
    if not args.no_plot:
        plot_comparison(comparison_data, args.metric)

if __name__ == "__main__":
    main()