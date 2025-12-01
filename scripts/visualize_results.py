#!/usr/bin/env python3
"""
Visualization tool for experimental results.

Generates comparison plots and tables for retrieval effectiveness.

Usage:
    python scripts/visualize_results.py --results results.json --output plots/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_markdown_table(results: Dict[str, Dict[str, float]], metrics: List[str]) -> str:
    """
    Generate a markdown comparison table.

    Args:
        results: Dictionary mapping run_name -> {metric: value}
        metrics: List of metrics to display

    Returns:
        Markdown table string
    """
    lines = []

    # Header
    header = f"| {'Run':<30s} |"
    separator = f"|{'-' * 32}|"
    for metric in metrics:
        header += f" {metric:<15s} |"
        separator += f"{'-' * 17}|"

    lines.append(header)
    lines.append(separator)

    # Rows
    for run_name, run_results in sorted(results.items()):
        row = f"| {run_name:<30s} |"
        for metric in metrics:
            value = run_results.get(metric, 0.0)
            row += f" {value:>15.4f} |"
        lines.append(row)

    return "\n".join(lines)


def generate_ascii_chart(results: Dict[str, Dict[str, float]], metric: str) -> str:
    """
    Generate ASCII bar chart for a metric.

    Args:
        results: Results dictionary
        metric: Metric to visualize

    Returns:
        ASCII chart string
    """
    lines = []
    lines.append(f"\n{metric} Comparison:")
    lines.append("=" * 60)

    # Extract values
    values = [(name, res.get(metric, 0.0)) for name, res in results.items()]
    values.sort(key=lambda x: x[1], reverse=True)

    if not values:
        return "\nNo data available\n"

    max_value = max(v[1] for v in values)

    # Generate bars
    for name, value in values:
        bar_length = int((value / max_value) * 40) if max_value > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{name[:25]:<25s} | {bar} {value:.4f}")

    return "\n".join(lines)


def visualize_results(results_files: List[str], output_path: str | None = None):
    """
    Visualize experimental results from JSON files.

    Args:
        results_files: List of paths to result JSON files
        output_path: Optional output path for markdown report
    """
    # Load all results
    all_results = {}

    for results_file in results_files:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            run_name = data.get('run_file', Path(results_file).stem)
            all_results[run_name] = data.get('metrics', {})

    if not all_results:
        logger.warning("No results loaded")
        return

    # Get all metrics
    all_metrics = set()
    for results in all_results.values():
        all_metrics.update(results.keys())
    all_metrics = sorted(all_metrics)

    # Key metrics for visualization
    key_metrics = [m for m in ['ndcg_cut.10', 'ndcg_cut.20', 'map', 'recip_rank'] if m in all_metrics]

    # Generate visualizations
    report_lines = []
    report_lines.append("# Experimental Results Report\n")

    # Summary table
    report_lines.append("## Summary Table\n")
    report_lines.append(generate_markdown_table(all_results, key_metrics))
    report_lines.append("")

    # ASCII charts for key metrics
    for metric in key_metrics:
        report_lines.append(generate_ascii_chart(all_results, metric))
        report_lines.append("")

    report = "\n".join(report_lines)

    # Output
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
    else:
        print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize experimental results"
    )
    parser.add_argument(
        '--results',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to result JSON files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for markdown report'
    )

    args = parser.parse_args()

    visualize_results(args.results, args.output)


if __name__ == '__main__':
    main()
