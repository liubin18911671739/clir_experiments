#!/usr/bin/env python3
"""
Rerank search results using monoT5 / mT5 models.

Implements pointwise reranking using seq2seq models with the standard
monoT5 format: "Query: <query> Document: <doc> Relevant:"

Usage:
    python scripts/rerank_mt5.py --config config/neuclir.yaml \\
        --base_run runs/bm25/bm25_fas.run --lang fas

    python scripts/rerank_mt5.py --config config/neuclir.yaml \\
        --base_run runs/dense/mdpr_rus.run --lang rus --model mt5_multilingual
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from utils_io import (
    load_yaml, read_trec_run, write_trec_run,
    load_corpus_from_dir, ensure_dir, get_repo_root, resolve_path
)
from utils_topics import parse_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonoT5Reranker:
    """monoT5 / mT5 reranker for cross-lingual IR."""

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize monoT5/mT5 reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
            use_fp16: Use mixed precision (FP16)
            batch_size: Batch size for inference
            max_length: Maximum input length
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(device)
            if use_fp16:
                self.model = self.model.half()
        elif device == 'cuda':
            logger.warning("CUDA not available, using CPU")
            self.device = 'cpu'

        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Score query-document pairs.

        Args:
            query_doc_pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        scores = []

        # Process in batches
        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch_pairs = query_doc_pairs[i:i + self.batch_size]
            batch_scores = self._score_batch(batch_pairs)
            scores.extend(batch_scores)

        return scores

    def _score_batch(self, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score a batch of query-document pairs.

        Args:
            batch_pairs: List of (query, document) tuples

        Returns:
            List of relevance scores for the batch
        """
        # Format inputs: "Query: <query> Document: <doc> Relevant:"
        inputs = [
            f"Query: {query} Document: {doc} Relevant:"
            for query, doc in batch_pairs
        ]

        # Tokenize
        encoded = self.tokenizer(
            inputs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        if self.device == 'cuda':
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate scores
        with torch.no_grad():
            if self.use_fp16 and self.device == 'cuda':
                with autocast():
                    outputs = self.model.generate(
                        **encoded,
                        max_length=2,  # Only need "true" or "false"
                        return_dict_in_generate=True,
                        output_scores=True
                    )
            else:
                outputs = self.model.generate(
                    **encoded,
                    max_length=2,
                    return_dict_in_generate=True,
                    output_scores=True
                )

        # Extract scores for "true" token
        # monoT5 models output logits for "true" vs "false"
        # We use the logit for "true" as the relevance score
        scores = []
        for i, score in enumerate(outputs.scores[0]):
            # Get token IDs for "true" and "false"
            # This may vary by model; adjust if needed
            true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
            false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]

            # Compute probability for "true"
            true_logit = score[true_token_id].item()
            false_logit = score[false_token_id].item()

            # Use logit difference as score
            relevance_score = true_logit - false_logit
            scores.append(relevance_score)

        return scores


def rerank_run(
    config: Dict[str, Any],
    base_run_path: str,
    lang: str,
    repo_root: Path,
    model_type: str = 'mt5',
    top_k: int | None = None
) -> None:
    """
    Rerank a base run using monoT5/mT5.

    Args:
        config: Configuration dictionary
        base_run_path: Path to base TREC run file
        lang: Language code
        repo_root: Repository root path
        model_type: Model configuration key ('mt5' or 'mt5_multilingual')
        top_k: Number of top documents to rerank (overrides config)
    """
    rerank_config = config['reranking'][model_type]
    corpus_dir = resolve_path(config['data']['corpus_dir'], repo_root)
    topics_dir = resolve_path(config['data']['topics_dir'], repo_root)
    runs_dir = resolve_path(config['runs']['reranked_dir'], repo_root)

    # Get parameters
    if top_k is None:
        top_k = rerank_config['top_k']

    # Load base run
    logger.info(f"Loading base run: {base_run_path}")
    base_run = read_trec_run(base_run_path)
    logger.info(f"Base run has {len(base_run)} queries")

    # Load topics
    topics_path = topics_dir / f"{lang}.topics.txt"
    logger.info(f"Loading topics from: {topics_path}")
    queries = parse_trec_topics(str(topics_path))

    # Load corpus into memory (for fast lookup)
    logger.info(f"Loading corpus from: {corpus_dir}/{lang}")
    corpus = {}
    for doc in load_corpus_from_dir(str(corpus_dir), lang):
        corpus[doc['id']] = doc.get('contents', doc.get('text', ''))
    logger.info(f"Loaded {len(corpus)} documents")

    # Initialize reranker
    logger.info(f"Initializing reranker: {rerank_config['model_name']}")
    reranker = MonoT5Reranker(
        model_name=rerank_config['model_name'],
        device=rerank_config['device'],
        use_fp16=rerank_config['use_fp16'],
        batch_size=rerank_config['batch_size'],
        max_length=rerank_config['max_length']
    )

    # Rerank each query
    logger.info(f"Reranking top-{top_k} documents per query...")
    all_reranked_results: List[Tuple[str, str, float]] = []

    for qid in tqdm(sorted(base_run.keys()), desc="Reranking queries"):
        if qid not in queries:
            logger.warning(f"Query {qid} not found in topics, skipping")
            continue

        query_text = queries[qid]

        # Get top-k documents from base run
        base_results = base_run[qid][:top_k]

        # Prepare query-document pairs
        pairs = []
        doc_ids = []

        for docid, rank, score in base_results:
            if docid not in corpus:
                logger.warning(f"Document {docid} not found in corpus, skipping")
                continue

            doc_text = corpus[docid]
            pairs.append((query_text, doc_text))
            doc_ids.append(docid)

        # Score pairs
        if pairs:
            scores = reranker.score_pairs(pairs)

            # Collect reranked results
            for docid, score in zip(doc_ids, scores):
                all_reranked_results.append((qid, docid, score))

    logger.info(f"Reranking complete. Total results: {len(all_reranked_results)}")

    # Write reranked run file
    base_run_name = Path(base_run_path).stem
    run_id_suffix = rerank_config['run_id_suffix']
    run_id = f"{base_run_name}{run_id_suffix}"
    output_path = runs_dir / f"{run_id}.run"
    ensure_dir(runs_dir)

    logger.info(f"Writing reranked results to: {output_path}")
    write_trec_run(all_reranked_results, str(output_path), run_id, max_rank=top_k)

    logger.info(f"Reranked run file saved: {output_path}")
    logger.info(f"Run ID: {run_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rerank search results using monoT5/mT5"
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
        help='Path to base TREC run file to rerank'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help='Language code (e.g., fas, rus, zho)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mt5', 'mt5_multilingual'],
        default='mt5',
        help='Reranking model configuration to use'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Number of top documents to rerank (overrides config)'
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

    # Run reranking
    rerank_run(config, args.base_run, args.lang, repo_root, args.model, args.top_k)

    logger.info("Reranking complete!")


if __name__ == '__main__':
    main()
