#!/usr/bin/env python3
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from hardware_performance_logger import HardwarePerformanceLogger

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

def display_comparison_table(comparison_data, sort_by='train_time_ms', ascending=True):
    """Display a formatted comparison table with customized sorting"""
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(comparison_data)
    
    # Add efficiency column (ms per million parameters)
    if 'parameters' in df.columns and 'train_time_ms' in df.columns:
        df['efficiency'] = df['train_time_ms'] / (df['parameters'] / 1_000_000)
        df['params_per_ms'] = df['parameters'] / df['train_time_ms']
    
    # Sort the data according to the specified column
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Format columns for better display
    df['build_time_ms'] = df['build_time_ms'].apply(lambda x: f"{x:.1f} ms")
    df['train_time_ms'] = df['train_time_ms'].apply(lambda x: f"{x:.1f} ms")
    df['time_per_epoch_ms'] = df['time_per_epoch_ms'].apply(lambda x: f"{x:.1f} ms")
    df['parameters'] = df['parameters'].apply(lambda x: f"{x:,}")
    if 'efficiency' in df.columns:
        df['efficiency'] = df['efficiency'].apply(lambda x: f"{x:.2f} ms/M")
    if 'params_per_ms' in df.columns:
        df['params_per_ms'] = df['params_per_ms'].apply(lambda x: f"{x:.2f}K/ms")
    if 'final_accuracy' in df.columns:
        df['final_accuracy'] = df['final_accuracy'].apply(lambda x: f"{x*100:.2f}%" if x is not None else "N/A")
    
    # Select and rename columns for display
    display_df = df[[
        'model_id', 'hardware', 'parameters', 
        'build_time_ms', 'train_time_ms', 'time_per_epoch_ms', 
        'efficiency', 'params_per_ms', 'epochs_completed', 'final_loss', 'final_accuracy'
    ]]
    
    display_df.columns = [
        'Model ID', 'Hardware', 'Parameters', 
        'Build Time', 'Total Train Time', 'Time/Epoch', 
        'ms/Million Params', 'K Params/ms', 'Epochs', 'Final Loss', 'Final Accuracy'
    ]
    
    print(tabulate(display_df, headers='keys', tablefmt='pretty'))
    
    return df  # Return the dataframe for possible use in other functions

def plot_comparison(comparison_data, metric='train_time_ms', sort_by=None, ascending=True):
    """Plot a comparison chart of the specified metric"""
    df = pd.DataFrame(comparison_data)
    
    # Add efficiency column (ms per million parameters)
    if 'parameters' in df.columns and 'train_time_ms' in df.columns:
        df['efficiency'] = df['train_time_ms'] / (df['parameters'] / 1_000_000)
        df['params_per_ms'] = df['parameters'] / df['train_time_ms']
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    
    metric_name_map = {
        'train_time_ms': 'Training Time (ms)',
        'time_per_epoch_ms': 'Time per Epoch (ms)',
        'build_time_ms': 'Model Build Time (ms)',
        'efficiency': 'Training Time per Million Parameters (ms/M)',
        'params_per_ms': 'Parameters Processed per ms (K/ms)'
    }
    
    y_label = metric_name_map.get(metric, metric)
    
    # Sort by the specified column or the metric itself if not specified
    sort_column = sort_by if sort_by else metric
    if sort_column in df.columns:
        df = df.sort_values(by=sort_column, ascending=ascending)
    
    # Create labels combining model ID and hardware
    labels = [f"{row['model_id']}\n{row['hardware']}" for _, row in df.iterrows()]
    
    plt.bar(labels, df[metric])
    plt.ylabel(y_label)
    
    # Update title to include sorting information
    sort_text = f"(sorted by {sort_by})" if sort_by and sort_by != metric else ""
    plt.title(f'Hardware Performance Comparison - {y_label} {sort_text}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    sort_suffix = f"_sorted_by_{sort_by}" if sort_by and sort_by != metric else ""
    plt.savefig(f'hardware_comparison_{metric}{sort_suffix}.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare hardware performance logs')
    parser.add_argument('--log-dir', default='hardware_performance_logs', 
                        help='Directory containing hardware performance logs')
    parser.add_argument('--experiment-id', help='Filter logs by experiment ID')
    parser.add_argument('--metric', default='train_time_ms',
                        choices=['train_time_ms', 'build_time_ms', 'time_per_epoch_ms', 
                                 'efficiency', 'params_per_ms'],
                        help='Metric to use for comparison')
    parser.add_argument('--sort-by', 
                        choices=['train_time_ms', 'build_time_ms', 'time_per_epoch_ms', 
                                 'parameters', 'efficiency', 'params_per_ms'],
                        help='Sort results by this column (defaults to the chosen metric)')
    parser.add_argument('--descending', action='store_true',
                        help='Sort in descending order (largest values first)')
    parser.add_argument('--no-plot', action='store_true', 
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    log_files = find_log_files(args.log_dir, args.experiment_id)
    
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return
    
    comparison_data = HardwarePerformanceLogger.compare_logs(log_files)
    
    # Determine sorting order
    ascending = not args.descending
    
    # If sort_by is not specified, default to the metric being displayed
    sort_by = args.sort_by if args.sort_by else args.metric
    
    # Display table with sorting applied
    df = display_comparison_table(comparison_data, sort_by=sort_by, ascending=ascending)
    
    if not args.no_plot:
        plot_comparison(comparison_data, args.metric, sort_by=sort_by, ascending=ascending)

if __name__ == "__main__":
    main()