#!/usr/bin/env python3
"""
Run BM25 retrieval using Pyserini.

Uses Pyserini's SimpleSearcher for BM25 ranking with configurable
parameters (k1, b).

Usage:
    python scripts/run_bm25.py --config config/neuclir.yaml --lang fas
    python scripts/run_bm25.py --config config/neuclir.yaml --lang rus --top_k 100
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from pyserini.search.lucene import LuceneSearcher

from utils_io import load_yaml, write_trec_run, ensure_dir, get_repo_root, resolve_path
from utils_topics import parse_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_bm25_search(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    top_k: int | None = None,
    k1: float | None = None,
    b: float | None = None
) -> None:
    """
    Run BM25 retrieval using Pyserini.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        top_k: Number of documents to retrieve (overrides config)
        k1: BM25 k1 parameter (overrides config)
        b: BM25 b parameter (overrides config)
    """
    bm25_config = config['bm25']
    topics_dir = resolve_path(config['data']['topics_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)
    runs_dir = resolve_path(config['runs']['bm25_dir'], repo_root)

    # Get parameters
    if top_k is None:
        top_k = bm25_config['top_k']
    if k1 is None:
        k1 = bm25_config['k1']
    if b is None:
        b = bm25_config['b']

    index_path = index_dir / lang

    # Validate index exists
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}\n"
            f"Run build_index_bm25.py first to create the index."
        )

    # Load topics
    topics_path = topics_dir / f"{lang}.topics.txt"
    logger.info(f"Loading topics from: {topics_path}")
    queries = parse_trec_topics(str(topics_path))
    logger.info(f"Loaded {len(queries)} queries")

    # Initialize BM25 searcher
    logger.info(f"Loading BM25 index: {index_path}")
    logger.info(f"BM25 parameters: k1={k1}, b={b}")

    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(k1, b)

    # Run search for all queries
    logger.info(f"Searching with top_k={top_k}...")
    all_results: List[Tuple[str, str, float]] = []

    for qid, query_text in queries.items():
        # Search
        hits = searcher.search(query_text, k=top_k)

        # Collect results
        for hit in hits:
            all_results.append((qid, hit.docid, hit.score))

        if len(all_results) % (top_k * 10) == 0:
            logger.info(f"Processed {len(all_results) // top_k} queries...")

    logger.info(f"Search complete. Total results: {len(all_results)}")

    # Write TREC run file
    run_id = bm25_config['run_id_template'].format(lang=lang)
    output_path = runs_dir / f"{run_id}.run"
    ensure_dir(runs_dir)

    logger.info(f"Writing results to: {output_path}")
    write_trec_run(all_results, str(output_path), run_id, max_rank=top_k)

    logger.info(f"Run file saved: {output_path}")
    logger.info(f"Run ID: {run_id}")

    # Print statistics
    logger.info(f"\nSearch Statistics:")
    logger.info(f"  Queries: {len(queries)}")
    logger.info(f"  Total results: {len(all_results)}")
    logger.info(f"  Avg results per query: {len(all_results) / len(queries):.2f}")


def batch_search(
    config: Dict[str, Any],
    languages: List[str],
    repo_root: Path
) -> None:
    """
    Run BM25 search for multiple languages.

    Args:
        config: Configuration dictionary
        languages: List of language codes
        repo_root: Repository root path
    """
    logger.info(f"Running batch BM25 search for languages: {languages}")

    for lang in languages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*60}\n")

        try:
            run_bm25_search(config, lang, repo_root)
        except Exception as e:
            logger.error(f"Error processing {lang}: {e}")
            continue

    logger.info("\nBatch search complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run BM25 retrieval using Pyserini"
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
        default=None,
        help='Language code (e.g., fas, rus, zho). If not specified, process all languages.'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Number of documents to retrieve (overrides config)'
    )
    parser.add_argument(
        '--k1',
        type=float,
        default=None,
        help='BM25 k1 parameter (overrides config)'
    )
    parser.add_argument(
        '--b',
        type=float,
        default=None,
        help='BM25 b parameter (overrides config)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all configured languages'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    repo_root = get_repo_root()

    # Batch mode or single language
    if args.batch or args.lang is None:
        languages = config['languages']
        batch_search(config, languages, repo_root)
    else:
        # Validate language
        if args.lang not in config['languages']:
            logger.warning(
                f"Language '{args.lang}' not in configured languages: {config['languages']}"
            )

        # Run search
        run_bm25_search(config, args.lang, repo_root, args.top_k, args.k1, args.b)

    logger.info("BM25 retrieval complete!")


if __name__ == '__main__':
    main()
