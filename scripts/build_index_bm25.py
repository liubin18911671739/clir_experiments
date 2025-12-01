#!/usr/bin/env python3
"""
Build BM25 (sparse) indexes using Pyserini.

Uses Pyserini's IndexReader and Anserini indexer for building
inverted indexes suitable for BM25 retrieval.

Usage:
    python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas
    python scripts/build_index_bm25.py --config config/neuclir.yaml --lang rus --threads 16
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

from utils_io import load_yaml, load_corpus_from_dir, ensure_dir, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_corpus_for_indexing(
    corpus_dir: str,
    lang: str,
    output_dir: Path
) -> Path:
    """
    Prepare corpus in format required by Anserini/Pyserini indexer.

    Anserini expects JSONL files with 'id' and 'contents' fields.

    Args:
        corpus_dir: Base corpus directory
        lang: Language code
        output_dir: Output directory for prepared corpus

    Returns:
        Path to prepared corpus directory
    """
    prepared_dir = output_dir / "prepared_corpus" / lang
    ensure_dir(prepared_dir)

    logger.info(f"Preparing corpus for indexing: {lang}")
    logger.info(f"Reading from: {corpus_dir}/{lang}")
    logger.info(f"Writing to: {prepared_dir}")

    # Copy or symlink corpus files
    # If corpus is already in correct format, we can use it directly
    from utils_io import load_corpus_from_dir
    import json

    output_file = prepared_dir / "corpus.jsonl"
    doc_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in load_corpus_from_dir(corpus_dir, lang):
            # Ensure correct format
            prepared_doc = {
                'id': doc['id'],
                'contents': doc.get('contents', doc.get('text', ''))
            }
            f.write(json.dumps(prepared_doc, ensure_ascii=False) + '\n')
            doc_count += 1

            if doc_count % 10000 == 0:
                logger.info(f"Prepared {doc_count} documents...")

    logger.info(f"Corpus preparation complete: {doc_count} documents")
    return prepared_dir


def build_bm25_index_pyserini(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    threads: int | None = None
) -> None:
    """
    Build BM25 index using Pyserini's Python interface.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        threads: Number of threads (overrides config)
    """
    corpus_dir = resolve_path(config['data']['corpus_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)

    if threads is None:
        threads = config['system']['n_threads']

    # Prepare output paths
    index_path = index_dir / lang
    ensure_dir(index_path)

    logger.info(f"Building BM25 index for {lang}")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Index output: {index_path}")
    logger.info(f"Threads: {threads}")

    # Prepare corpus
    temp_dir = index_dir / "temp"
    prepared_corpus = prepare_corpus_for_indexing(str(corpus_dir), lang, temp_dir)

    # Build index using Pyserini's SimpleIndexer
    try:
        from pyserini.index.lucene import IndexReader, LuceneIndexer

        logger.info("Building Lucene index with Pyserini...")

        # Use Pyserini's command-line indexer (more stable)
        # This calls Anserini's indexer under the hood
        cmd = [
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', str(prepared_corpus),
            '--index', str(index_path),
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', str(threads),
            '--storePositions', '--storeDocvectors', '--storeRaw'
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Indexing failed with error:\n{result.stderr}")
            raise RuntimeError("BM25 indexing failed")

        logger.info("Indexing complete!")

        # Verify index
        index_reader = IndexReader(str(index_path))
        num_docs = index_reader.stats()['documents']
        logger.info(f"Index contains {num_docs} documents")

    except ImportError as e:
        logger.error(
            "Pyserini Lucene indexer not available.\n"
            "Make sure Pyserini is properly installed with Java support.\n"
            f"Error: {e}"
        )
        raise

    logger.info(f"BM25 index built successfully: {index_path}")


def build_bm25_index_anserini(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path,
    threads: int | None = None
) -> None:
    """
    Build BM25 index using Anserini directly (alternative method).

    This method calls Anserini's indexer if available.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
        threads: Number of threads (overrides config)
    """
    corpus_dir = resolve_path(config['data']['corpus_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)

    if threads is None:
        threads = config['system']['n_threads']

    # Prepare output paths
    index_path = index_dir / lang
    ensure_dir(index_path)

    logger.info(f"Building BM25 index for {lang} using Anserini")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Index output: {index_path}")

    # Prepare corpus
    temp_dir = index_dir / "temp"
    prepared_corpus = prepare_corpus_for_indexing(str(corpus_dir), lang, temp_dir)

    # Check if Anserini is available
    anserini_path = Path.home() / "anserini"
    if not anserini_path.exists():
        logger.warning(
            f"Anserini not found at {anserini_path}. "
            "Falling back to Pyserini method."
        )
        build_bm25_index_pyserini(config, lang, repo_root, threads)
        return

    # Build index with Anserini
    cmd = [
        str(anserini_path / "target" / "appassembler" / "bin" / "IndexCollection"),
        "-collection", "JsonCollection",
        "-input", str(prepared_corpus),
        "-index", str(index_path),
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", str(threads),
        "-storePositions", "-storeDocvectors", "-storeRaw"
    ]

    logger.info(f"Running Anserini indexer...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Anserini indexing failed:\n{result.stderr}")
        logger.info("Falling back to Pyserini method...")
        build_bm25_index_pyserini(config, lang, repo_root, threads)
        return

    logger.info(f"BM25 index built successfully: {index_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build BM25 indexes using Pyserini/Anserini"
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
        '--threads',
        type=int,
        default=None,
        help='Number of threads for indexing (overrides config)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['pyserini', 'anserini'],
        default='pyserini',
        help='Indexing method to use'
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

    # Build index
    if args.method == 'anserini':
        build_bm25_index_anserini(config, args.lang, repo_root, args.threads)
    else:
        build_bm25_index_pyserini(config, args.lang, repo_root, args.threads)

    logger.info("BM25 index building complete!")


if __name__ == '__main__':
    main()
