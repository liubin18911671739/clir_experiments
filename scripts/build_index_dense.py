#!/usr/bin/env python3
"""
Build dense retrieval indexes using Pyserini.

Supports:
- mDPR-style dual encoders with AutoDocumentEncoder
- ColBERT-style late interaction models

Usage:
    python scripts/build_index_dense.py --config config/neuclir.yaml --model mdpr --lang fas
    python scripts/build_index_dense.py --config config/neuclir.yaml --model colbert --lang rus
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Iterator

from pyserini.encode import AutoDocumentEncoder
from pyserini.index.faiss import FaissIndexWriter

from utils_io import load_yaml, load_corpus_from_dir, ensure_dir, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIterator:
    """Iterator adapter for corpus documents compatible with Pyserini encoders."""

    def __init__(self, corpus_dir: str, lang: str):
        """
        Initialize document iterator.

        Args:
            corpus_dir: Base corpus directory
            lang: Language code
        """
        self.corpus_dir = corpus_dir
        self.lang = lang
        self._docs = None

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """
        Iterate over documents.

        Yields:
            Dictionary with 'id' and 'contents' keys
        """
        for doc in load_corpus_from_dir(self.corpus_dir, self.lang):
            # Pyserini expects 'id' and 'contents' or 'text'
            if 'contents' in doc:
                yield {'id': doc['id'], 'contents': doc['contents']}
            elif 'text' in doc:
                yield {'id': doc['id'], 'contents': doc['text']}
            else:
                logger.warning(f"Document {doc.get('id', 'unknown')} missing 'contents' or 'text' field")


def build_mdpr_index(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path
) -> None:
    """
    Build mDPR-style dense index using Pyserini.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
    """
    mdpr_config = config['dense']['mdpr']
    corpus_dir = resolve_path(config['data']['corpus_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['dense_dir'], repo_root)

    # Prepare index output path
    index_name = mdpr_config['index_name']
    index_path = index_dir / f"{index_name}_{lang}"
    ensure_dir(index_path)

    logger.info(f"Building mDPR index for {lang}")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Index output: {index_path}")
    logger.info(f"Model: {mdpr_config['doc_encoder']}")

    # Initialize document encoder
    encoder = AutoDocumentEncoder(
        model_name=mdpr_config['doc_encoder'],
        pooling='cls',  # mDPR uses CLS pooling
        l2_norm=True,   # Normalize embeddings
        device='cuda' if config['system']['use_gpu'] else 'cpu'
    )

    # Initialize FAISS index writer
    index_writer = FaissIndexWriter(
        output_dir=str(index_path),
        dimension=mdpr_config['embedding_dim']
    )

    # Create document iterator
    doc_iterator = DocumentIterator(str(corpus_dir), lang)

    # Encode and index documents
    logger.info("Encoding and indexing documents...")
    batch_size = mdpr_config['batch_size']
    batch = []
    doc_count = 0

    for doc in doc_iterator:
        batch.append(doc)

        if len(batch) >= batch_size:
            # Encode batch
            doc_ids = [d['id'] for d in batch]
            doc_texts = [d['contents'] for d in batch]

            embeddings = encoder.encode(doc_texts)

            # Add to index
            for doc_id, embedding in zip(doc_ids, embeddings):
                index_writer.add(doc_id, embedding)

            doc_count += len(batch)
            logger.info(f"Indexed {doc_count} documents...")
            batch = []

    # Process remaining documents
    if batch:
        doc_ids = [d['id'] for d in batch]
        doc_texts = [d['contents'] for d in batch]

        embeddings = encoder.encode(doc_texts)

        for doc_id, embedding in zip(doc_ids, embeddings):
            index_writer.add(doc_id, embedding)

        doc_count += len(batch)

    # Finalize index
    logger.info(f"Finalizing index with {doc_count} documents...")
    index_writer.close()

    logger.info(f"mDPR index built successfully: {index_path}")


def build_colbert_index(
    config: Dict[str, Any],
    lang: str,
    repo_root: Path
) -> None:
    """
    Build ColBERT-style dense index using Pyserini.

    Args:
        config: Configuration dictionary
        lang: Language code
        repo_root: Repository root path
    """
    colbert_config = config['dense']['colbert']
    corpus_dir = resolve_path(config['data']['corpus_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['dense_dir'], repo_root)

    # Prepare index output path
    index_name = colbert_config['index_name']
    index_path = index_dir / f"{index_name}_{lang}"
    ensure_dir(index_path)

    logger.info(f"Building ColBERT index for {lang}")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Index output: {index_path}")
    logger.info(f"Model: {colbert_config['model_name']}")

    # For ColBERT, we need to use Pyserini's ColBERT encoder
    # Note: This requires the pyserini[colbert] extras
    try:
        from pyserini.encode import ColbertDocumentEncoder
    except ImportError:
        logger.error(
            "ColBERT encoder not available. "
            "Install with: pip install pyserini[colbert]"
        )
        raise

    # Initialize ColBERT document encoder
    encoder = ColbertDocumentEncoder(
        model_name=colbert_config['model_name'],
        device='cuda' if config['system']['use_gpu'] else 'cpu'
    )

    # For ColBERT, we typically use a specialized indexer
    # This is a simplified version; in practice, you might use ColBERT's native indexer
    logger.warning(
        "ColBERT indexing requires specialized setup. "
        "Consider using ColBERT's native indexing tools for production use."
    )

    # Create document iterator
    doc_iterator = DocumentIterator(str(corpus_dir), lang)

    # Save documents in ColBERT-compatible format
    docs_path = index_path / "collection.tsv"
    logger.info(f"Saving documents to {docs_path}")

    with open(docs_path, 'w', encoding='utf-8') as f:
        for idx, doc in enumerate(doc_iterator):
            # ColBERT format: id \t text
            f.write(f"{doc['id']}\t{doc['contents']}\n")

            if (idx + 1) % 10000 == 0:
                logger.info(f"Processed {idx + 1} documents...")

    logger.info(
        f"Documents saved to {docs_path}\n"
        f"To complete ColBERT indexing, run ColBERT's indexer on this file.\n"
        f"See: https://github.com/stanford-futuredata/ColBERT"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build dense retrieval indexes using Pyserini"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mdpr', 'colbert'],
        required=True,
        help='Dense model type to use'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help='Language code (e.g., fas, rus, zho)'
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

    # Build index based on model type
    if args.model == 'mdpr':
        build_mdpr_index(config, args.lang, repo_root)
    elif args.model == 'colbert':
        build_colbert_index(config, args.lang, repo_root)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    logger.info("Dense index building complete!")


if __name__ == '__main__':
    main()
