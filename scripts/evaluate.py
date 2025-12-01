#!/usr/bin/env python3
"""
Automatic evaluation script using trec_eval.

Evaluates TREC run files against qrels and outputs formatted results.

Usage:
    python scripts/evaluate.py --config config/neuclir.yaml --run runs/bm25/bm25_fas.run --lang fas
    python scripts/evaluate.py --config config/neuclir.yaml --run_dir runs/bm25/ --lang fas
    python scripts/evaluate.py --config config/neuclir.yaml --batch
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from utils_io import load_yaml, ensure_dir, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_trec_eval_output(output: str) -> Dict[str, float]:
    """
    Parse trec_eval output into dictionary.

    Args:
        output: Raw trec_eval output text

    Returns:
        Dictionary mapping metric names to values
    """
    results = {}

    for line in output.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 3:
            metric = parts[0]
            # Some metrics have "all" as second field, some don't
            if parts[1] == "all":
                value = float(parts[2])
            else:
                value = float(parts[1])
            results[metric] = value

    return results


def run_trec_eval(
    qrels_path: str,
    run_path: str,
    metrics: List[str] | None = None,
    trec_eval_path: str = "trec_eval"
) -> Dict[str, float]:
    """
    Run trec_eval on a run file.

    Args:
        qrels_path: Path to qrels file
        run_path: Path to run file
        metrics: List of metrics to evaluate (e.g., ['ndcg_cut.10', 'map'])
        trec_eval_path: Path to trec_eval binary

    Returns:
        Dictionary of metric names to values

    Raises:
        FileNotFoundError: If trec_eval binary or files not found
        RuntimeError: If trec_eval execution fails
    """
    # Validate files exist
    qrels_path = Path(qrels_path)
    run_path = Path(run_path)

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    # Build command
    cmd = [trec_eval_path]

    # Add metrics
    if metrics:
        for metric in metrics:
            cmd.extend(['-m', metric])
    else:
        # Default: use all metrics
        cmd.append('-m')
        cmd.append('all_trec')

    cmd.extend([str(qrels_path), str(run_path)])

    # Run trec_eval
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"trec_eval binary not found at: {trec_eval_path}\n"
            "Please install trec_eval or specify path with --trec_eval_path"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"trec_eval failed:\n{e.stderr}")

    # Parse output
    results = parse_trec_eval_output(result.stdout)

    return results


def evaluate_run(
    config: Dict[str, Any],
    run_path: str,
    lang: str,
    repo_root: Path,
    output_file: str | None = None
) -> Dict[str, float]:
    """
    Evaluate a single run file.

    Args:
        config: Configuration dictionary
        run_path: Path to run file
        lang: Language code
        repo_root: Repository root path
        output_file: Optional path to save results JSON

    Returns:
        Dictionary of evaluation results
    """
    qrels_dir = resolve_path(config['data']['qrels_dir'], repo_root)
    qrels_path = qrels_dir / f"{lang}.qrels.txt"

    # Get evaluation settings
    metrics = config['evaluation']['metrics']
    trec_eval_path = config['evaluation']['trec_eval_path']

    logger.info(f"Evaluating run: {run_path}")
    logger.info(f"Qrels: {qrels_path}")
    logger.info(f"Metrics: {', '.join(metrics)}")

    # Run evaluation
    results = run_trec_eval(
        str(qrels_path),
        run_path,
        metrics,
        trec_eval_path
    )

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"{'='*60}")
    for metric, value in sorted(results.items()):
        logger.info(f"  {metric:30s}: {value:.4f}")
    logger.info(f"{'='*60}\n")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        ensure_dir(output_path.parent)

        eval_data = {
            'run_file': str(run_path),
            'qrels_file': str(qrels_path),
            'language': lang,
            'metrics': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    return results


def evaluate_directory(
    config: Dict[str, Any],
    run_dir: str,
    lang: str,
    repo_root: Path,
    output_dir: str | None = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all run files in a directory.

    Args:
        config: Configuration dictionary
        run_dir: Directory containing run files
        lang: Language code
        repo_root: Repository root path
        output_dir: Optional directory to save results

    Returns:
        Dictionary mapping run names to their results
    """
    run_dir = Path(run_dir)
    all_results = {}

    # Find all .run files
    run_files = sorted(run_dir.glob("*.run"))

    if not run_files:
        logger.warning(f"No .run files found in: {run_dir}")
        return all_results

    logger.info(f"Found {len(run_files)} run files to evaluate")

    for run_file in run_files:
        run_name = run_file.stem

        # Determine output file path
        if output_dir:
            output_file = Path(output_dir) / f"{run_name}_eval.json"
        else:
            output_file = None

        try:
            results = evaluate_run(
                config,
                str(run_file),
                lang,
                repo_root,
                str(output_file) if output_file else None
            )
            all_results[run_name] = results
        except Exception as e:
            logger.error(f"Error evaluating {run_name}: {e}")
            continue

    # Print summary table
    print_comparison_table(all_results, config['evaluation']['metrics'])

    return all_results


def print_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str]
) -> None:
    """
    Print comparison table of multiple runs.

    Args:
        results: Dictionary mapping run names to results
        metrics: List of metrics to display
    """
    if not results:
        return

    logger.info("\nComparison Table:")
    logger.info(f"{'='*80}")

    # Header
    header = f"{'Run':<30s}"
    for metric in metrics:
        header += f" | {metric:<15s}"
    logger.info(header)
    logger.info(f"{'-'*80}")

    # Rows
    for run_name, run_results in sorted(results.items()):
        row = f"{run_name:<30s}"
        for metric in metrics:
            value = run_results.get(metric, 0.0)
            row += f" | {value:<15.4f}"
        logger.info(row)

    logger.info(f"{'='*80}\n")


def batch_evaluate(
    config: Dict[str, Any],
    repo_root: Path
) -> None:
    """
    Evaluate all runs for all languages.

    Args:
        config: Configuration dictionary
        repo_root: Repository root path
    """
    languages = config['languages']
    run_base = resolve_path(config['runs']['bm25_dir'], repo_root)

    logger.info(f"Batch evaluation for languages: {languages}")

    for lang in languages:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating language: {lang}")
        logger.info(f"{'='*80}\n")

        # Evaluate BM25 runs
        bm25_dir = resolve_path(config['runs']['bm25_dir'], repo_root)
        if bm25_dir.exists():
            logger.info(f"Evaluating BM25 runs...")
            evaluate_directory(config, str(bm25_dir), lang, repo_root)

        # Evaluate dense runs
        dense_dir = resolve_path(config['runs']['dense_dir'], repo_root)
        if dense_dir.exists():
            logger.info(f"Evaluating dense runs...")
            evaluate_directory(config, str(dense_dir), lang, repo_root)

        # Evaluate reranked runs
        reranked_dir = resolve_path(config['runs']['reranked_dir'], repo_root)
        if reranked_dir.exists():
            logger.info(f"Evaluating reranked runs...")
            evaluate_directory(config, str(reranked_dir), lang, repo_root)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate TREC run files using trec_eval"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Path to single run file to evaluate'
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='Directory containing run files to evaluate'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Language code (required for single run or directory evaluation)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for batch results'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Evaluate all runs for all languages'
    )
    parser.add_argument(
        '--trec_eval_path',
        type=str,
        default=None,
        help='Path to trec_eval binary (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    repo_root = get_repo_root()

    # Override trec_eval path if specified
    if args.trec_eval_path:
        config['evaluation']['trec_eval_path'] = args.trec_eval_path

    # Batch mode
    if args.batch:
        batch_evaluate(config, repo_root)
    # Single run
    elif args.run:
        if not args.lang:
            parser.error("--lang is required when evaluating a single run")
        evaluate_run(config, args.run, args.lang, repo_root, args.output)
    # Directory of runs
    elif args.run_dir:
        if not args.lang:
            parser.error("--lang is required when evaluating a directory")
        evaluate_directory(config, args.run_dir, args.lang, repo_root, args.output_dir)
    else:
        parser.error("Must specify --run, --run_dir, or --batch")

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
