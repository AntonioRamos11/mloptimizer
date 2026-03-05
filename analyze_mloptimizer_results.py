#!/usr/bin/env python3
"""Analyze ML Optimizer results from result files."""

import os
import re
import statistics
from collections import defaultdict


def parse_file(path):
    """Parse a result file using regex to extract dataset and timing info."""
    text = open(path, "r", errors="ignore").read()

    dataset = None
    seconds = None

    m_dataset = re.search(r'"dataset_tag"\s*:\s*"([^"]+)"', text)
    if m_dataset:
        dataset = m_dataset.group(1)

    m_time = re.search(r'Optimization took:.*?([0-9]+\.[0-9]+)', text)
    if m_time:
        seconds = float(m_time.group(1))

    return dataset, seconds


def analyze_results(results_dir="results"):
    """Analyze all result files in the directory."""
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    files = [f for f in os.listdir(results_dir) if not f.startswith('.') and not f.endswith('.c3621')]
    files = [os.path.join(results_dir, f) for f in files]

    datasets = set()
    data_by_dataset = defaultdict(list)

    for path in files:
        dataset, seconds = parse_file(path)
        if dataset:
            datasets.add(dataset)
            data_by_dataset[dataset].append(seconds)

    return datasets, data_by_dataset


def print_summary(datasets, data_by_dataset):
    """Print summary table of results."""
    overhead = 1.15
    print(f"Datasets found: {sorted(datasets)}")
    print()

    for dataset in sorted(datasets):
        times = [t for t in data_by_dataset[dataset] if t is not None]
        if not times:
            continue

        base_time = statistics.mean(times)

        print(f"\nDATASET: {dataset}")
        print(f"Average optimization time: {base_time/3600:.1f} h")
        print("GPUs | Time(h) | Speedup | Efficiency")

        for num_gpus in [1, 2, 4]:
            time_h = (base_time / num_gpus) * overhead / 3600
            speedup = base_time / ((base_time / num_gpus) * overhead)
            efficiency = (speedup / num_gpus) * 100
            print(f"{num_gpus} | {time_h:.1f} | {speedup:.1f} | {efficiency:.0f} %")


def main():
    datasets, data_by_dataset = analyze_results()
    print_summary(datasets, data_by_dataset)


if __name__ == "__main__":
    main()
