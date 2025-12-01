"""
I/O utility functions for cross-lingual IR experiments.

Provides helpers for:
- Loading/saving YAML configurations
- Directory management
- Reading/writing JSONL corpus files
- Reading/writing TREC run files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import yaml


def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_yaml(data: Dict[str, Any], output_path: str) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def ensure_dir(dir_path: Path | str) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        dir_path: Directory path to create

    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_jsonl(jsonl_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Load documents from JSONL file.

    Expects each line to be a JSON object with at least 'id' and 'contents' fields.

    Args:
        jsonl_path: Path to JSONL file

    Yields:
        Dictionary for each document

    Raises:
        FileNotFoundError: If JSONL file doesn't exist
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus_from_dir(corpus_dir: str, lang: str) -> Generator[Dict[str, Any], None, None]:
    """
    Load all JSONL files from a corpus directory for a given language.

    Args:
        corpus_dir: Base corpus directory
        lang: Language code (e.g., 'fas', 'rus', 'zho')

    Yields:
        Dictionary for each document across all JSONL files

    Raises:
        FileNotFoundError: If corpus directory doesn't exist
    """
    lang_dir = Path(corpus_dir) / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {lang_dir}")

    jsonl_files = sorted(lang_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {lang_dir}")

    for jsonl_file in jsonl_files:
        yield from load_jsonl(str(jsonl_file))


def write_trec_run(
    results: List[Tuple[str, str, float]],
    output_path: str,
    run_id: str,
    max_rank: int = 1000
) -> None:
    """
    Write search results to TREC-format run file.

    TREC format: qid Q0 docid rank score runid

    Args:
        results: List of (query_id, doc_id, score) tuples
        output_path: Path to output run file
        run_id: Run identifier for TREC format
        max_rank: Maximum rank to write (default: 1000)
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Group by query and sort by score
    query_results: Dict[str, List[Tuple[str, float]]] = {}
    for qid, docid, score in results:
        if qid not in query_results:
            query_results[qid] = []
        query_results[qid].append((docid, score))

    # Write TREC format
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in sorted(query_results.keys()):
            # Sort by score descending
            docs = sorted(query_results[qid], key=lambda x: x[1], reverse=True)

            # Write top-k results
            for rank, (docid, score) in enumerate(docs[:max_rank], start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_id}\n")


def read_trec_run(run_path: str) -> Dict[str, List[Tuple[str, int, float]]]:
    """
    Read TREC-format run file.

    Args:
        run_path: Path to TREC run file

    Returns:
        Dictionary mapping query_id to list of (doc_id, rank, score) tuples

    Raises:
        FileNotFoundError: If run file doesn't exist
    """
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    run_data: Dict[str, List[Tuple[str, int, float]]] = {}

    with open(run_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                docid = parts[2]
                rank = int(parts[3])
                score = float(parts[4])

                if qid not in run_data:
                    run_data[qid] = []
                run_data[qid].append((docid, rank, score))

    # Sort each query's results by rank
    for qid in run_data:
        run_data[qid].sort(key=lambda x: x[1])

    return run_data


def count_corpus_docs(corpus_dir: str, lang: str) -> int:
    """
    Count total number of documents in a corpus.

    Args:
        corpus_dir: Base corpus directory
        lang: Language code

    Returns:
        Total document count
    """
    count = 0
    try:
        for _ in load_corpus_from_dir(corpus_dir, lang):
            count += 1
    except FileNotFoundError:
        return 0
    return count


def get_repo_root() -> Path:
    """
    Get repository root directory.

    Returns:
        Path to repository root
    """
    # Assume scripts are in {repo_root}/scripts/
    return Path(__file__).parent.parent


def resolve_path(path: str, base_dir: Path | None = None) -> Path:
    """
    Resolve a path relative to base directory or repository root.

    Args:
        path: Path to resolve (can be absolute or relative)
        base_dir: Base directory for relative paths (default: repo root)

    Returns:
        Resolved absolute Path
    """
    path = Path(path)
    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = get_repo_root()

    return (base_dir / path).resolve()
