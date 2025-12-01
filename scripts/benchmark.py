#!/usr/bin/env python3
"""
Performance benchmarking tool for IR systems.

Measures:
- Indexing time and memory usage
- Query latency (avg, p50, p95, p99)
- Throughput (queries per second)
- Memory consumption

Usage:
    python scripts/benchmark.py --config config/neuclir.yaml \
        --lang fas --systems bm25 dense

    python scripts/benchmark.py --config config/neuclir.yaml \
        --lang fas --systems bm25 --output benchmarks/results.json
"""

import argparse
import json
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

from pyserini.search.lucene import LuceneSearcher

from utils_io import load_yaml, ensure_dir, get_repo_root, resolve_path
from utils_topics import parse_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking utility."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = defaultdict(dict)
        self.process = psutil.Process(os.getpid())

    def measure_memory(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_search(
        self,
        searcher,
        queries: Dict[str, str],
        top_k: int = 1000,
        warmup: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark search performance.

        Args:
            searcher: Search engine instance
            queries: Dictionary of qid -> query_text
            top_k: Number of results to retrieve
            warmup: Number of warmup queries

        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking with {len(queries)} queries...")

        # Warmup
        warmup_queries = list(queries.values())[:warmup]
        for query in warmup_queries:
            _ = searcher.search(query, k=top_k)

        # Measure memory before
        mem_before = self.measure_memory()

        # Benchmark queries
        latencies = []
        query_list = list(queries.items())

        for qid, query_text in query_list:
            start_time = time.time()
            _ = searcher.search(query_text, k=top_k)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

        # Measure memory after
        mem_after = self.measure_memory()

        # Calculate statistics
        latencies.sort()
        n = len(latencies)

        results = {
            'num_queries': n,
            'total_time_sec': sum(latencies) / 1000,
            'throughput_qps': n / (sum(latencies) / 1000),
            'latency_ms': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': latencies[int(n * 0.95)] if n > 0 else 0,
                'p99': latencies[int(n * 0.99)] if n > 0 else 0,
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if n > 1 else 0
            },
            'memory_mb': {
                'before': mem_before,
                'after': mem_after,
                'delta': mem_after - mem_before
            }
        }

        return results

    def print_results(self, system_name: str, results: Dict[str, Any]) -> None:
        """Print benchmark results."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {system_name}")
        print(f"{'='*60}")
        print(f"Queries:          {results['num_queries']}")
        print(f"Total Time:       {results['total_time_sec']:.2f} sec")
        print(f"Throughput:       {results['throughput_qps']:.2f} queries/sec")
        print(f"\nLatency (ms):")
        print(f"  Mean:           {results['latency_ms']['mean']:.2f}")
        print(f"  Median:         {results['latency_ms']['median']:.2f}")
        print(f"  P95:            {results['latency_ms']['p95']:.2f}")
        print(f"  P99:            {results['latency_ms']['p99']:.2f}")
        print(f"  Min:            {results['latency_ms']['min']:.2f}")
        print(f"  Max:            {results['latency_ms']['max']:.2f}")
        print(f"  Std Dev:        {results['latency_ms']['stdev']:.2f}")
        print(f"\nMemory (MB):")
        print(f"  Before:         {results['memory_mb']['before']:.2f}")
        print(f"  After:          {results['memory_mb']['after']:.2f}")
        print(f"  Delta:          {results['memory_mb']['delta']:.2f}")
        print(f"{'='*60}\n")


def benchmark_bm25(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    queries: Dict[str, str]
) -> Dict[str, Any]:
    """
    Benchmark BM25 retrieval.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        queries: Dictionary of queries

    Returns:
        Benchmark results
    """
    bm25_config = config['bm25']
    index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)
    index_path = index_dir / lang

    logger.info(f"Loading BM25 index: {index_path}")
    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(bm25_config['k1'], bm25_config['b'])

    benchmark = PerformanceBenchmark()
    results = benchmark.benchmark_search(
        searcher,
        queries,
        top_k=bm25_config['top_k']
    )

    results['system'] = 'BM25'
    results['parameters'] = {
        'k1': bm25_config['k1'],
        'b': bm25_config['b'],
        'top_k': bm25_config['top_k']
    }

    return results


def benchmark_dense(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    queries: Dict[str, str]
) -> Dict[str, Any]:
    """
    Benchmark dense retrieval.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        queries: Dictionary of queries

    Returns:
        Benchmark results
    """
    from pyserini.search.faiss import FaissSearcher
    from pyserini.encode import AutoQueryEncoder

    mdpr_config = config['dense']['mdpr']
    index_dir = resolve_path(config['indexes']['dense_dir'], repo_root)
    index_name = mdpr_config['index_name']
    index_path = index_dir / f"{index_name}_{lang}"

    logger.info(f"Loading query encoder: {mdpr_config['query_encoder']}")
    encoder = AutoQueryEncoder(
        model_name=mdpr_config['query_encoder'],
        pooling='cls',
        l2_norm=True,
        device='cuda' if config['system']['use_gpu'] else 'cpu'
    )

    logger.info(f"Loading FAISS index: {index_path}")
    searcher = FaissSearcher(str(index_path), encoder)

    benchmark = PerformanceBenchmark()
    results = benchmark.benchmark_search(
        searcher,
        queries,
        top_k=mdpr_config['top_k']
    )

    results['system'] = 'Dense (mDPR)'
    results['parameters'] = {
        'model': mdpr_config['model_name'],
        'top_k': mdpr_config['top_k'],
        'device': 'cuda' if config['system']['use_gpu'] else 'cpu'
    }

    return results


def run_benchmarks(
    config: Dict[str, Any],
    lang: str,
    systems: List[str],
    output_path: str | None = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks for specified systems.

    Args:
        config: Configuration dictionary
        lang: Language code
        systems: List of systems to benchmark ('bm25', 'dense')
        output_path: Optional output path for results

    Returns:
        Dictionary mapping system names to results
    """
    repo_root = get_repo_root()

    # Load topics
    topics_dir = resolve_path(config['data']['topics_dir'], repo_root)
    topics_path = topics_dir / f"{lang}.topics.txt"
    logger.info(f"Loading topics from: {topics_path}")
    queries = parse_trec_topics(str(topics_path))
    logger.info(f"Loaded {len(queries)} queries for benchmarking")

    all_results = {}
    benchmark = PerformanceBenchmark()

    # Benchmark each system
    for system in systems:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Benchmarking: {system.upper()}")
        logger.info(f"{'#'*60}\n")

        if system == 'bm25':
            results = benchmark_bm25(config, lang, repo_root, queries)
        elif system == 'dense':
            results = benchmark_dense(config, lang, repo_root, queries)
        else:
            logger.warning(f"Unknown system: {system}, skipping")
            continue

        all_results[system] = results
        benchmark.print_results(system.upper(), results)

    # Save results
    if output_path:
        output_file = Path(output_path)
        ensure_dir(output_file.parent)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Benchmark results saved to: {output_path}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for IR systems"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help='Language code (e.g., fas, rus, zho)'
    )
    parser.add_argument(
        '--systems',
        type=str,
        nargs='+',
        choices=['bm25', 'dense'],
        required=True,
        help='Systems to benchmark'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)

    # Run benchmarks
    run_benchmarks(config, args.lang, args.systems, args.output)

    logger.info("Benchmarking complete!")


if __name__ == '__main__':
    main()
