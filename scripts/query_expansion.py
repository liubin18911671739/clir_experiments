#!/usr/bin/env python3
"""
Query expansion using Pseudo-Relevance Feedback (PRF) and RM3.

Implements:
- RM3 (Relevance Model 3) query expansion
- Standard PRF with top documents
- Rocchio algorithm

Usage:
    python scripts/query_expansion.py --config config/neuclir.yaml \
        --base_run runs/bm25/bm25_fas.run --lang fas --method rm3

    python scripts/query_expansion.py --config config/neuclir.yaml \
        --base_run runs/bm25/bm25_fas.run --lang fas --method prf --fb_docs 10
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict, Counter
import math

from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer

from utils_io import (
    load_yaml, read_trec_run, write_trec_run,
    ensure_dir, get_repo_root, resolve_path
)
from utils_topics import parse_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryExpander:
    """Base class for query expansion methods."""

    def __init__(self, searcher: LuceneSearcher, analyzer: Analyzer):
        """
        Initialize query expander.

        Args:
            searcher: Pyserini LuceneSearcher for retrieving documents
            analyzer: Lucene analyzer for tokenization
        """
        self.searcher = searcher
        self.analyzer = analyzer

    def expand_query(
        self,
        original_query: str,
        feedback_docs: List[str],
        num_terms: int = 10
    ) -> str:
        """
        Expand query using feedback documents.

        Args:
            original_query: Original query string
            feedback_docs: List of feedback document IDs
            num_terms: Number of expansion terms to add

        Returns:
            Expanded query string
        """
        raise NotImplementedError


class RM3Expander(QueryExpander):
    """
    RM3 (Relevance Model 3) query expansion.

    RM3 interpolates the original query with the relevance model
    built from pseudo-relevant documents.
    """

    def __init__(
        self,
        searcher: LuceneSearcher,
        analyzer: Analyzer,
        original_query_weight: float = 0.5
    ):
        """
        Initialize RM3 expander.

        Args:
            searcher: Pyserini searcher
            analyzer: Lucene analyzer
            original_query_weight: Weight for original query (default: 0.5)
        """
        super().__init__(searcher, analyzer)
        self.original_query_weight = original_query_weight

    def expand_query(
        self,
        original_query: str,
        feedback_docs: List[str],
        num_terms: int = 10
    ) -> str:
        """
        Expand query using RM3.

        Args:
            original_query: Original query
            feedback_docs: Feedback document IDs
            num_terms: Number of expansion terms

        Returns:
            Expanded query with RM3
        """
        # Build relevance model from feedback docs
        term_scores = self._build_relevance_model(feedback_docs)

        # Get top expansion terms
        expansion_terms = sorted(
            term_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_terms]

        # Tokenize original query
        original_tokens = self.analyzer.analyze(original_query)

        # Build expanded query with interpolation
        # RM3: Q' = λQ + (1-λ)RM
        expanded_parts = []

        # Add original query terms with weight
        for token in original_tokens:
            expanded_parts.append(f"{token}^{self.original_query_weight}")

        # Add expansion terms with weight
        fb_weight = 1.0 - self.original_query_weight
        for term, score in expansion_terms:
            if term not in original_tokens:  # Avoid duplicates
                weighted_score = score * fb_weight
                expanded_parts.append(f"{term}^{weighted_score:.4f}")

        return " ".join(expanded_parts)

    def _build_relevance_model(self, doc_ids: List[str]) -> Dict[str, float]:
        """
        Build relevance model from feedback documents.

        Args:
            doc_ids: Feedback document IDs

        Returns:
            Dictionary mapping terms to relevance scores
        """
        term_scores = defaultdict(float)
        total_docs = len(doc_ids)

        for doc_id in doc_ids:
            # Get document
            try:
                doc = self.searcher.doc(doc_id)
                if doc is None:
                    continue

                doc_text = doc.raw()

                # Analyze document to get terms
                tokens = self.analyzer.analyze(doc_text)
                term_counts = Counter(tokens)

                # Calculate term scores (simplified relevance model)
                doc_length = len(tokens)
                for term, count in term_counts.items():
                    # P(w|D) * P(D|Q)
                    # Simplified: use term frequency in document
                    tf = count / doc_length if doc_length > 0 else 0
                    term_scores[term] += tf / total_docs

            except Exception as e:
                logger.warning(f"Error processing doc {doc_id}: {e}")
                continue

        return dict(term_scores)


class PRFExpander(QueryExpander):
    """
    Standard Pseudo-Relevance Feedback expander.

    Extracts top terms from feedback documents based on tf-idf.
    """

    def expand_query(
        self,
        original_query: str,
        feedback_docs: List[str],
        num_terms: int = 10
    ) -> str:
        """
        Expand query using PRF.

        Args:
            original_query: Original query
            feedback_docs: Feedback document IDs
            num_terms: Number of expansion terms

        Returns:
            Expanded query
        """
        # Collect terms from feedback documents
        term_doc_freq = defaultdict(int)
        term_total_freq = defaultdict(int)

        for doc_id in feedback_docs:
            try:
                doc = self.searcher.doc(doc_id)
                if doc is None:
                    continue

                doc_text = doc.raw()
                tokens = self.analyzer.analyze(doc_text)

                # Count term frequencies
                seen_terms = set()
                for token in tokens:
                    term_total_freq[token] += 1
                    if token not in seen_terms:
                        term_doc_freq[token] += 1
                        seen_terms.add(token)

            except Exception as e:
                logger.warning(f"Error processing doc {doc_id}: {e}")
                continue

        # Calculate tf-idf scores
        num_docs = len(feedback_docs)
        term_scores = {}

        for term in term_total_freq:
            tf = term_total_freq[term]
            df = term_doc_freq[term]
            idf = math.log(num_docs / df) if df > 0 else 0
            term_scores[term] = tf * idf

        # Get top terms
        expansion_terms = sorted(
            term_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_terms]

        # Build expanded query
        original_tokens = self.analyzer.analyze(original_query)
        expanded_parts = list(original_tokens)

        for term, score in expansion_terms:
            if term not in original_tokens:
                expanded_parts.append(term)

        return " ".join(expanded_parts)


def expand_and_rerank(
    config: Dict[str, Any],
    base_run_path: str,
    lang: str,
    repo_root: Path,
    method: str = 'rm3',
    fb_docs: int = 10,
    fb_terms: int = 10,
    original_query_weight: float = 0.5
) -> None:
    """
    Expand queries and re-retrieve with expanded queries.

    Args:
        config: Configuration dictionary
        base_run_path: Path to base run file
        lang: Language code
        repo_root: Repository root path
        method: Expansion method ('rm3' or 'prf')
        fb_docs: Number of feedback documents
        fb_terms: Number of expansion terms
        original_query_weight: Weight for original query (RM3 only)
    """
    bm25_config = config['bm25']
    topics_dir = resolve_path(config['data']['topics_dir'], repo_root)
    index_dir = resolve_path(config['indexes']['bm25_dir'], repo_root)
    runs_dir = resolve_path(config['runs']['bm25_dir'], repo_root)

    # Load base run
    logger.info(f"Loading base run: {base_run_path}")
    base_run = read_trec_run(base_run_path)

    # Load topics
    topics_path = topics_dir / f"{lang}.topics.txt"
    logger.info(f"Loading topics from: {topics_path}")
    queries = parse_trec_topics(str(topics_path))

    # Load index
    index_path = index_dir / lang
    logger.info(f"Loading BM25 index: {index_path}")
    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(bm25_config['k1'], bm25_config['b'])

    # Initialize analyzer
    analyzer = Analyzer(get_lucene_analyzer())

    # Initialize expander
    if method == 'rm3':
        expander = RM3Expander(
            searcher,
            analyzer,
            original_query_weight=original_query_weight
        )
    elif method == 'prf':
        expander = PRFExpander(searcher, analyzer)
    else:
        raise ValueError(f"Unknown expansion method: {method}")

    logger.info(f"Query expansion method: {method}")
    logger.info(f"Feedback docs: {fb_docs}, Expansion terms: {fb_terms}")

    # Expand queries and re-retrieve
    all_results: List[Tuple[str, str, float]] = []

    for qid, query_text in queries.items():
        # Get feedback documents from base run
        if qid not in base_run:
            logger.warning(f"Query {qid} not in base run, skipping")
            continue

        feedback_doc_ids = [
            docid for docid, rank, score in base_run[qid][:fb_docs]
        ]

        # Expand query
        expanded_query = expander.expand_query(
            query_text,
            feedback_doc_ids,
            num_terms=fb_terms
        )

        logger.debug(f"Query {qid}:")
        logger.debug(f"  Original: {query_text}")
        logger.debug(f"  Expanded: {expanded_query}")

        # Re-retrieve with expanded query
        hits = searcher.search(expanded_query, k=bm25_config['top_k'])

        for hit in hits:
            all_results.append((qid, hit.docid, hit.score))

    logger.info(f"Re-retrieval complete. Total results: {len(all_results)}")

    # Write results
    base_run_name = Path(base_run_path).stem
    run_id = f"{base_run_name}_{method}_fb{fb_docs}"
    output_path = runs_dir / f"{run_id}.run"
    ensure_dir(runs_dir)

    logger.info(f"Writing results to: {output_path}")
    write_trec_run(all_results, str(output_path), run_id, max_rank=bm25_config['top_k'])

    logger.info(f"Expanded run saved: {output_path}")
    logger.info(f"Run ID: {run_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query expansion using RM3 or PRF"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--base_run',
        type=str,
        required=True,
        help='Path to base run file for feedback'
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
        choices=['rm3', 'prf'],
        default='rm3',
        help='Expansion method (default: rm3)'
    )
    parser.add_argument(
        '--fb_docs',
        type=int,
        default=10,
        help='Number of feedback documents (default: 10)'
    )
    parser.add_argument(
        '--fb_terms',
        type=int,
        default=10,
        help='Number of expansion terms (default: 10)'
    )
    parser.add_argument(
        '--original_query_weight',
        type=float,
        default=0.5,
        help='Weight for original query in RM3 (default: 0.5)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    repo_root = get_repo_root()

    # Run query expansion
    expand_and_rerank(
        config,
        args.base_run,
        args.lang,
        repo_root,
        method=args.method,
        fb_docs=args.fb_docs,
        fb_terms=args.fb_terms,
        original_query_weight=args.original_query_weight
    )

    logger.info("Query expansion complete!")


if __name__ == '__main__':
    main()
