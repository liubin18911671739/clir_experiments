#!/usr/bin/env python3
"""
Hybrid retrieval combining BM25 and dense retrieval.

Implements various fusion strategies:
- Reciprocal Rank Fusion (RRF)
- Linear combination of scores
- Weighted fusion
- CombSUM - Sum of normalized scores
- CombMNZ - CombSUM with non-zero count multiplier

Usage:
    python scripts/run_hybrid.py --config config/neuclir.yaml \\
        --bm25_run runs/bm25/bm25_fas.run \\
        --dense_run runs/dense/mdpr_fas.run \\
        --lang fas --method rrf

    python scripts/run_hybrid.py --config config/neuclir.yaml \\
        --bm25_run runs/bm25/bm25_fas.run \\
        --dense_run runs/dense/mdpr_fas.run \\
        --lang fas --method weighted --alpha 0.7
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict

from utils_io import load_yaml, read_trec_run, write_trec_run, ensure_dir, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    runs: List[Dict[str, List[Tuple[str, int, float]]]],
    k: int = 60
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine multiple runs using Reciprocal Rank Fusion.

    RRF score = sum over runs of 1 / (k + rank)

    Args:
        runs: List of run dictionaries (qid -> [(docid, rank, score)])
        k: RRF constant (default: 60)

    Returns:
        Dictionary mapping qid to list of (docid, rrf_score) tuples
    """
    fused_results = {}

    # Get all query IDs
    all_qids: Set[str] = set()
    for run in runs:
        all_qids.update(run.keys())

    for qid in all_qids:
        doc_scores = defaultdict(float)

        # Accumulate RRF scores from each run
        for run in runs:
            if qid not in run:
                continue

            for docid, rank, _ in run[qid]:
                rrf_score = 1.0 / (k + rank)
                doc_scores[docid] += rrf_score

        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results[qid] = sorted_docs

    return fused_results


def linear_combination(
    runs: List[Dict[str, List[Tuple[str, int, float]]]],
    weights: List[float] | None = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine multiple runs using linear combination of normalized scores.

    Args:
        runs: List of run dictionaries (qid -> [(docid, rank, score)])
        weights: Weights for each run (must sum to 1.0)

    Returns:
        Dictionary mapping qid to list of (docid, combined_score) tuples
    """
    if weights is None:
        weights = [1.0 / len(runs)] * len(runs)

    assert len(weights) == len(runs), "Number of weights must match number of runs"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

    fused_results = {}

    # Get all query IDs
    all_qids: Set[str] = set()
    for run in runs:
        all_qids.update(run.keys())

    for qid in all_qids:
        doc_scores = defaultdict(float)

        # Normalize scores for each run and combine
        for run, weight in zip(runs, weights):
            if qid not in run:
                continue

            # Extract scores
            scores = [score for _, _, score in run[qid]]
            if not scores:
                continue

            # Min-max normalization
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                # All scores are the same, use uniform normalization
                normalized_scores = {docid: 1.0 for docid, _, _ in run[qid]}
            else:
                normalized_scores = {
                    docid: (score - min_score) / score_range
                    for docid, _, score in run[qid]
                }

            # Add weighted normalized scores
            for docid, norm_score in normalized_scores.items():
                doc_scores[docid] += weight * norm_score

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results[qid] = sorted_docs

    return fused_results


def combsum(
    runs: List[Dict[str, List[Tuple[str, int, float]]]]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombSUM fusion: Sum of normalized scores.

    Normalizes scores using min-max normalization, then sums across runs.

    Args:
        runs: List of run dictionaries (qid -> [(docid, rank, score)])

    Returns:
        Dictionary mapping qid to list of (docid, combined_score) tuples
    """
    fused_results = {}

    # Get all query IDs
    all_qids: Set[str] = set()
    for run in runs:
        all_qids.update(run.keys())

    for qid in all_qids:
        doc_scores = defaultdict(float)

        # Sum normalized scores from each run
        for run in runs:
            if qid not in run:
                continue

            # Extract and normalize scores
            scores = [score for _, _, score in run[qid]]
            if not scores:
                continue

            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                normalized_scores = {docid: 1.0 for docid, _, _ in run[qid]}
            else:
                normalized_scores = {
                    docid: (score - min_score) / score_range
                    for docid, _, score in run[qid]
                }

            # Sum normalized scores
            for docid, norm_score in normalized_scores.items():
                doc_scores[docid] += norm_score

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results[qid] = sorted_docs

    return fused_results


def combmnz(
    runs: List[Dict[str, List[Tuple[str, int, float]]]]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombMNZ fusion: CombSUM multiplied by number of non-zero scores.

    CombMNZ = CombSUM × (number of runs where document appears)

    This gives preference to documents that appear in multiple runs.

    Args:
        runs: List of run dictionaries (qid -> [(docid, rank, score)])

    Returns:
        Dictionary mapping qid to list of (docid, combined_score) tuples
    """
    fused_results = {}

    # Get all query IDs
    all_qids: Set[str] = set()
    for run in runs:
        all_qids.update(run.keys())

    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)  # Count how many runs contain each doc

        # Sum normalized scores and count occurrences
        for run in runs:
            if qid not in run:
                continue

            # Extract and normalize scores
            scores = [score for _, _, score in run[qid]]
            if not scores:
                continue

            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                normalized_scores = {docid: 1.0 for docid, _, _ in run[qid]}
            else:
                normalized_scores = {
                    docid: (score - min_score) / score_range
                    for docid, _, score in run[qid]
                }

            # Sum normalized scores and count
            for docid, norm_score in normalized_scores.items():
                doc_scores[docid] += norm_score
                doc_counts[docid] += 1

        # Multiply by count (MNZ = Minimum Non-Zero)
        final_scores = {}
        for docid, sum_score in doc_scores.items():
            final_scores[docid] = sum_score * doc_counts[docid]

        # Sort by final score
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results[qid] = sorted_docs

    return fused_results


def run_hybrid_retrieval(
    config: Dict[str, Any],
    bm25_run_path: str,
    dense_run_path: str,
    lang: str,
    repo_root: Path,
    method: str = 'rrf',
    alpha: float = 0.5,
    rrf_k: int = 60,
    top_k: int = 1000
) -> None:
    """
    Run hybrid retrieval combining BM25 and dense results.

    Args:
        config: Configuration dictionary
        bm25_run_path: Path to BM25 run file
        dense_run_path: Path to dense run file
        lang: Language code
        repo_root: Repository root path
        method: Fusion method ('rrf', 'linear', 'weighted', 'combsum', 'combmnz')
        alpha: Weight for BM25 in weighted fusion (dense weight = 1 - alpha)
        rrf_k: RRF constant
        top_k: Number of top results to keep
    """
    runs_dir = resolve_path(config['runs']['dense_dir'], repo_root)

    logger.info(f"Running hybrid retrieval for {lang}")
    logger.info(f"BM25 run: {bm25_run_path}")
    logger.info(f"Dense run: {dense_run_path}")
    logger.info(f"Fusion method: {method}")

    # Load runs
    logger.info("Loading run files...")
    bm25_run = read_trec_run(bm25_run_path)
    dense_run = read_trec_run(dense_run_path)

    logger.info(f"BM25 queries: {len(bm25_run)}")
    logger.info(f"Dense queries: {len(dense_run)}")

    # Combine runs
    logger.info(f"Combining runs with method: {method}")

    if method == 'rrf':
        fused_results = reciprocal_rank_fusion([bm25_run, dense_run], k=rrf_k)
        run_id_suffix = f"_hybrid_rrf_k{rrf_k}"
    elif method == 'linear':
        fused_results = linear_combination([bm25_run, dense_run])
        run_id_suffix = "_hybrid_linear"
    elif method == 'weighted':
        weights = [alpha, 1.0 - alpha]
        fused_results = linear_combination([bm25_run, dense_run], weights=weights)
        run_id_suffix = f"_hybrid_w{alpha:.2f}"
    elif method == 'combsum':
        fused_results = combsum([bm25_run, dense_run])
        run_id_suffix = "_hybrid_combsum"
    elif method == 'combmnz':
        fused_results = combmnz([bm25_run, dense_run])
        run_id_suffix = "_hybrid_combmnz"
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # Convert to TREC format
    all_results: List[Tuple[str, str, float]] = []
    for qid, doc_scores in fused_results.items():
        for docid, score in doc_scores[:top_k]:
            all_results.append((qid, docid, score))

    logger.info(f"Hybrid results: {len(all_results)} total")

    # Generate run ID
    bm25_name = Path(bm25_run_path).stem
    dense_name = Path(dense_run_path).stem
    run_id = f"{bm25_name}_{dense_name}{run_id_suffix}"

    # Write output
    output_path = runs_dir / f"{run_id}.run"
    ensure_dir(runs_dir)

    logger.info(f"Writing results to: {output_path}")
    write_trec_run(all_results, str(output_path), run_id, max_rank=top_k)

    logger.info(f"Hybrid run saved: {output_path}")
    logger.info(f"Run ID: {run_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval combining BM25 and dense retrieval"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--bm25_run',
        type=str,
        required=True,
        help='Path to BM25 run file'
    )
    parser.add_argument(
        '--dense_run',
        type=str,
        required=True,
        help='Path to dense run file'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help='Language code (e.g., fas, rus, zho)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['rrf', 'linear', 'weighted', 'combsum', 'combmnz'],
        default='rrf',
        help='Fusion method (default: rrf)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weight for BM25 in weighted fusion (default: 0.5)'
    )
    parser.add_argument(
        '--rrf_k',
        type=int,
        default=60,
        help='RRF constant (default: 60)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=1000,
        help='Number of top results to keep (default: 1000)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    repo_root = get_repo_root()

    # Validate language
    if args.lang not in config['languages']:
        logger.warning(
            f"Language '{args.lang}' not in configured languages: {config['languages']}"
        )

    # Validate alpha for weighted fusion
    if args.method == 'weighted':
        if not 0 <= args.alpha <= 1:
            parser.error("Alpha must be between 0 and 1")

    # Run hybrid retrieval
    run_hybrid_retrieval(
        config,
        args.bm25_run,
        args.dense_run,
        args.lang,
        repo_root,
        method=args.method,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        top_k=args.top_k
    )

    logger.info("Hybrid retrieval complete!")


if __name__ == '__main__':
    main()
