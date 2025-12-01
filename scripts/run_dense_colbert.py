#!/usr/bin/env python3
"""
Run dense retrieval using ColBERT-style late interaction models.

This script provides a simplified interface for ColBERT search.
For production use, consider using ColBERT's native search tools.

Usage:
    python scripts/run_dense_colbert.py --config config/neuclir.yaml --lang fas
    python scripts/run_dense_colbert.py --config config/neuclir.yaml --lang rus --top_k 100
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from utils_io import load_yaml, write_trec_run, ensure_dir, get_repo_root, resolve_path
from utils_topics import parse_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_colbert_search(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    top_k: int | None = None
) -> None:
    """
    Run dense retrieval using ColBERT model.

    Note: This is a placeholder implementation. For production use,
    integrate with ColBERT's native searcher or use Pyserini's ColBERT support.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        top_k: Number of documents to retrieve (overrides config)
    """
    colbert_config = config['dense']['colbert']
    topics_dir = resolve_path(config['data']['topics_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['dense_dir'], repo_root)
    runs_dir = resolve_path(config['runs']['dense_dir'], repo_root)

    # Get parameters
    if top_k is None:
        top_k = colbert_config['top_k']

    index_name = colbert_config['index_name']
    index_path = index_dir / f"{index_name}_{lang}"

    # Validate index exists
    if not index_path.exists():
        raise FileNotFoundError(
            f"ColBERT index not found: {index_path}\n"
            f"Run build_index_dense.py first to create the index."
        )

    # Load topics
    topics_path = topics_dir / f"{lang}.topics.txt"
    logger.info(f"Loading topics from: {topics_path}")
    queries = parse_trec_topics(str(topics_path))
    logger.info(f"Loaded {len(queries)} queries")

    # Try to use Pyserini's ColBERT searcher if available
    try:
        from pyserini.search.faiss import FaissSearcher
        from pyserini.encode import ColbertQueryEncoder

        logger.info(f"Loading ColBERT query encoder: {colbert_config['model_name']}")
        encoder = ColbertQueryEncoder(
            model_name=colbert_config['model_name'],
            device='cuda' if config['system']['use_gpu'] else 'cpu'
        )

        logger.info(f"Loading ColBERT index: {index_path}")
        searcher = FaissSearcher(
            str(index_path),
            encoder
        )

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
        run_id = colbert_config['run_id_template'].format(lang=lang)
        output_path = runs_dir / f"{run_id}.run"
        ensure_dir(runs_dir)

        logger.info(f"Writing results to: {output_path}")
        write_trec_run(all_results, str(output_path), run_id, max_rank=top_k)

        logger.info(f"Run file saved: {output_path}")
        logger.info(f"Run ID: {run_id}")

    except ImportError:
        logger.error(
            "ColBERT support not available in Pyserini.\n"
            "Install with: pip install pyserini[colbert]\n"
            "\n"
            "Alternatively, use ColBERT's native search tools:\n"
            "1. Clone ColBERT: https://github.com/stanford-futuredata/ColBERT\n"
            "2. Follow their search documentation\n"
            "3. Convert results to TREC format"
        )
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run dense retrieval using ColBERT"
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
        '--top_k',
        type=int,
        default=None,
        help='Number of documents to retrieve (overrides config)'
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

    # Run search
    run_colbert_search(config, args.lang, repo_root, args.top_k)

    logger.info("ColBERT retrieval complete!")


if __name__ == '__main__':
    main()
