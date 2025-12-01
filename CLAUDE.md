# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **cross-lingual information retrieval (CLIR) experimentation toolkit** for NeuCLIR / CAKE-ILC style research. The codebase implements complete pipelines from corpus indexing through retrieval to evaluation, supporting multiple retrieval paradigms (sparse BM25, dense neural, hybrid fusion) and neural reranking.

**Key Design Principles:**
- Configuration-driven: All parameters in `config/neuclir.yaml`
- TREC-compatible: Standard formats for topics, runs, and qrels
- Pipeline-oriented: Each script performs one task, composable via `run_experiments.py`
- Research-focused: Clear, documented code over micro-optimization

## Core Architecture

### Data Flow

```
Corpus (JSONL) → Index (BM25/Dense) → Retrieval → Runs (TREC format) → Evaluation
                                          ↓
                                      Reranking (monoT5/mT5)
                                          ↓
                                    Reranked Runs
```

### Module Organization

**Utilities** (`scripts/utils_*.py`):
- `utils_io.py`: YAML config, JSONL corpus, TREC run I/O, path resolution
- `utils_topics.py`: TREC topic parsing (XML & simple formats), qrels loading

**Indexing** (`scripts/build_index_*.py`):
- `build_index_bm25.py`: Pyserini/Lucene sparse indexes
- `build_index_dense.py`: FAISS dense indexes (mDPR/ColBERT via Pyserini encoders)

**Retrieval** (`scripts/run_*.py`):
- `run_bm25.py`: BM25 search with configurable k1/b parameters
- `run_dense_mdpr.py`: Dense retrieval via Pyserini FaissSearcher
- `run_dense_colbert.py`: ColBERT late interaction retrieval
- `run_hybrid.py`: Fusion strategies (RRF, linear combination, weighted)

**Reranking**:
- `rerank_mt5.py`: monoT5/mT5 pointwise reranking with transformers

**Evaluation & Orchestration**:
- `evaluate.py`: trec_eval wrapper with batch processing and JSON output
- `run_experiments.py`: End-to-end pipeline orchestration (BM25/dense/full)

### Configuration Structure

`config/neuclir.yaml` drives all behavior:

```yaml
languages: [fas, rus, zho]
data: {corpus_dir, topics_dir, qrels_dir}
indexes: {bm25_dir, dense_dir}
runs: {bm25_dir, dense_dir, reranked_dir}
bm25: {k1, b, top_k, run_id_template}
dense:
  mdpr: {model_name, doc_encoder, query_encoder, batch_size, embedding_dim, ...}
  colbert: {model_name, batch_size, max_doc_length, ...}
reranking:
  mt5: {model_name, batch_size, top_k, device, use_fp16, ...}
evaluation: {trec_eval_path, metrics}
system: {n_threads, use_gpu, gpu_device}
```

## Development Commands

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_utils_io.py -v

# Single test function
pytest tests/test_hybrid.py::test_reciprocal_rank_fusion -v
```

### Building Indexes

```bash
# BM25 index for a language
python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas

# Dense (mDPR) index
python scripts/build_index_dense.py --config config/neuclir.yaml --model mdpr --lang fas
```

### Running Retrieval

```bash
# BM25 retrieval
python scripts/run_bm25.py --config config/neuclir.yaml --lang fas

# Dense retrieval
python scripts/run_dense_mdpr.py --config config/neuclir.yaml --lang fas

# Hybrid fusion (RRF)
python scripts/run_hybrid.py --config config/neuclir.yaml \
    --bm25_run runs/bm25/bm25_fas.run \
    --dense_run runs/dense/mdpr_fas.run \
    --lang fas --method rrf
```

### Reranking & Evaluation

```bash
# Rerank with monoT5
python scripts/rerank_mt5.py --config config/neuclir.yaml \
    --base_run runs/bm25/bm25_fas.run --lang fas

# Evaluate single run
python scripts/evaluate.py --config config/neuclir.yaml \
    --run runs/bm25/bm25_fas.run --lang fas

# Evaluate directory of runs
python scripts/evaluate.py --config config/neuclir.yaml \
    --run_dir runs/reranked --lang fas
```

### Batch Experiments (Recommended for Full Pipelines)

```bash
# Complete BM25 pipeline (index → retrieve → evaluate)
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline bm25

# Complete dense pipeline
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline dense_mdpr

# Reranking pipeline on existing runs
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline rerank

# Full end-to-end (BM25 + Dense + Reranking + Evaluation)
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline full
```

## Key Implementation Patterns

### Path Resolution

All scripts use `utils_io.resolve_path()` to handle both absolute and relative paths from config. Paths are resolved relative to repository root (`get_repo_root()` assumes scripts are in `{repo_root}/scripts/`).

### TREC Run Format

Run files follow strict TREC format: `qid Q0 docid rank score runid`
- Use `write_trec_run(results, output_path, run_id, max_rank)` for writing
- Use `read_trec_run(run_path)` → `Dict[qid, List[(docid, rank, score)]]` for reading

### Corpus Loading

Corpus files are JSONL with `{"id": "...", "contents": "..."}` structure:
```python
from utils_io import load_corpus_from_dir

for doc in load_corpus_from_dir(corpus_dir, lang):
    # doc is {"id": str, "contents": str}
```

### Topic Parsing

Supports both XML-style TREC topics and simple tab-separated:
```python
from utils_topics import parse_trec_topics

queries = parse_trec_topics(topics_path, use_desc=False, use_narr=False)
# Returns: Dict[qid, query_text]
```

### Hybrid Retrieval Fusion

Three fusion methods in `run_hybrid.py`:
- `reciprocal_rank_fusion(runs, k=60)`: RRF with constant k
- `linear_combination(runs, weights)`: Min-max normalized linear combo
- Use via `--method rrf|linear|weighted --alpha 0.7` (alpha is BM25 weight)

### Reranking Pattern

monoT5/mT5 reranking uses standard format: `"Query: <q> Document: <d> Relevant:"`
- Scores are logit differences: `logit(true) - logit(false)`
- Batch processing with optional FP16 via `--use_fp16`
- Only reranks top-k from base run (configurable)

## Adding New Features

### Adding a New Dense Model

1. Update `config/neuclir.yaml` under `dense:` section with model parameters
2. Add encoder initialization in `build_index_dense.py`:
   ```python
   def build_new_model_index(config, lang, repo_root):
       encoder = AutoDocumentEncoder(model_name=..., pooling=..., l2_norm=...)
       index_writer = FaissIndexWriter(output_dir=..., dimension=...)
       # Encode and index documents
   ```
3. Create `run_dense_newmodel.py` following `run_dense_mdpr.py` pattern
4. Add to `run_experiments.py` pipeline options if needed

### Adding a New Reranking Model

1. Add config under `reranking:` in `neuclir.yaml`
2. Extend `MonoT5Reranker` class in `rerank_mt5.py` or create new class:
   - Implement `score_pairs(query_doc_pairs) → List[float]`
   - Handle tokenization and model inference
3. Add model type to `rerank_mt5.py` argument parser

### Adding a New Fusion Strategy

1. Implement fusion function in `run_hybrid.py`:
   ```python
   def new_fusion_method(runs: List[Dict], **kwargs) -> Dict[qid, List[(docid, score)]]:
       # Combine run results and return sorted
   ```
2. Add to `run_hybrid_retrieval()` method selection
3. Add test case in `tests/test_hybrid.py`

## Important Constraints

### File Format Requirements

- **Corpus**: JSONL with `id` and `contents` (or `text`) keys
- **Topics**: TREC XML format or tab-separated `qid\tquery`
- **Qrels**: Standard TREC format `qid 0 docid relevance`
- **Runs**: TREC format `qid Q0 docid rank score runid`

### Pyserini Dependencies

- BM25 indexing requires Java (Anserini/Lucene backend)
- Dense indexing requires FAISS (CPU or GPU variant)
- ColBERT requires `pyserini[colbert]` extras

### Model Paths

- Models specified in config are HuggingFace model names by default
- Can use local paths: `doc_encoder: "/path/to/model"`
- GPU device selection via `system.gpu_device` in config

### Run ID Templates

Run IDs use template strings with `{lang}` placeholder:
- `run_id_template: "bm25_{lang}"` → `bm25_fas`, `bm25_rus`
- Reranking appends suffix: `run_id_suffix: "_mt5"` → `bm25_fas_mt5`

## Testing Philosophy

Tests focus on core utilities and algorithms, not end-to-end pipelines:
- `test_utils_io.py`: YAML, JSONL, TREC I/O correctness
- `test_utils_topics.py`: Topic parsing (XML and simple formats)
- `test_hybrid.py`: Fusion algorithm correctness (RRF, linear combination)

Add tests when implementing new utilities or fusion strategies. End-to-end integration is validated manually via `run_experiments.py`.
