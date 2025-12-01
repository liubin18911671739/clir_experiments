# Cross-Lingual IR Experiments

**[English](#english) | [中文](#中文)**

Toolkit for NeuCLIR / CAKE-ILC style cross-lingual information retrieval experiments.

**Status**: ✅ Production Ready | **Version**: 2.5.0 | **Lines of Code**: ~4,077 🚀

---

## 中文

### 简介

这是一个用于 NeuCLIR / CAKE-ILC 风格跨语言信息检索实验的工具包。

### 核心功能

- **稀疏检索**: BM25 传统检索（Pyserini/Lucene）🆕
- **密集检索**: mDPR 风格双编码器和 ColBERT 晚期交互模型
- **混合检索**: RRF、线性融合、加权融合 🆕
- **神经重排序**: monoT5/mT5 序列到序列重排序器
- **自动评估**: trec_eval 集成，批量评估 🆕
- **批量实验**: 端到端流水线编排 🆕
- **配置驱动**: 所有设置集中在 `config/neuclir.yaml`
- **单元测试**: Pytest 测试套件 🆕
- **TREC 兼容**: 标准 TREC 运行文件格式，便于评估

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 方式 1: 使用批量实验脚本（推荐）🆕
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline bm25

# 方式 2: 手动运行各步骤
# 构建 BM25 索引
python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas

# 运行检索
python scripts/run_bm25.py --config config/neuclir.yaml --lang fas

# 自动评估
python scripts/evaluate.py --config config/neuclir.yaml --run_dir runs/bm25 --lang fas

# 运行测试
pytest tests/
```

### 文档

- 📋 [TODO.md](TODO.md) - 开发进度和计划
- 📖 完整使用文档见下方英文部分
- 🤝 [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南

---

## English

## Features

- **Sparse Retrieval**: BM25 traditional retrieval (Pyserini/Lucene) 🆕
- **Dense Retrieval**: mDPR-style dual encoders and ColBERT late interaction models
- **Hybrid Retrieval**: RRF, linear combination, weighted fusion 🆕
- **Reranking**: monoT5/mT5 seq2seq rerankers
- **Automatic Evaluation**: trec_eval integration, batch evaluation 🆕
- **Batch Experiments**: End-to-end pipeline orchestration 🆕
- **Configuration-driven**: All settings in `config/neuclir.yaml`
- **Unit Tests**: Pytest test suite 🆕
- **TREC-compatible**: Standard TREC run file formats for evaluation

## Project Structure

```
clir_experiments/
├── config/
│   └── neuclir.yaml              # Main configuration file
├── data/
│   ├── corpus/{lang}/            # JSONL corpus files
│   ├── topics/{lang}.topics.txt  # TREC topic files
│   └── qrels/{lang}.qrels.txt    # TREC qrels files
├── indexes/
│   ├── bm25/{lang}/              # BM25 indexes
│   └── dense/{index_name}_{lang}/ # Dense indexes
├── runs/
│   ├── bm25/                     # BM25 run files
│   ├── dense/                    # Dense retrieval runs
│   └── reranked/                 # Reranked runs
└── scripts/
    ├── utils_io.py               # I/O utilities
    ├── utils_topics.py           # Topic parsing utilities
    ├── build_index_dense.py      # Build dense indexes
    ├── run_dense_mdpr.py         # Run mDPR search
    ├── run_dense_colbert.py      # Run ColBERT search
    └── rerank_mt5.py             # Rerank with monoT5/mT5
```

## Installation

### Requirements

- Python 3.9+
- CUDA 11+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd clir_experiments

# Install dependencies
pip install -r requirements.txt

# For ColBERT support (optional)
pip install pyserini[colbert]
```

## Quick Start

### 1. Prepare Data

Organize your data following this structure:

```bash
# Corpus files (JSONL format)
data/corpus/fas/*.jsonl
data/corpus/rus/*.jsonl
data/corpus/zho/*.jsonl

# Topic files (TREC format)
data/topics/fas.topics.txt
data/topics/rus.topics.txt
data/topics/zho.topics.txt

# Qrels files (TREC format)
data/qrels/fas.qrels.txt
data/qrels/rus.qrels.txt
data/qrels/zho.qrels.txt
```

**Corpus format** (JSONL):
```json
{"id": "doc001", "contents": "Document text here..."}
{"id": "doc002", "contents": "Another document..."}
```

**Topics format** (TREC-style):
```xml
<top>
<num> Number: 1
<title> Query text here
</top>
```

Or simple format:
```
1    Query text here
2    Another query
```

**Qrels format** (TREC):
```
1 0 doc001 1
1 0 doc005 2
2 0 doc003 1
```

### 2. Configure Experiment

Edit `config/neuclir.yaml` to set:
- Languages to process
- Model names (mDPR, ColBERT, monoT5/mT5)
- Retrieval parameters (top-k, batch sizes)
- Hardware settings (GPU device, threads)

### 3. Build Dense Indexes

Build mDPR-style dense index:
```bash
python scripts/build_index_dense.py \
    --config config/neuclir.yaml \
    --model mdpr \
    --lang fas
```

Build ColBERT index:
```bash
python scripts/build_index_dense.py \
    --config config/neuclir.yaml \
    --model colbert \
    --lang rus
```

### 4. Run Dense Retrieval

Search with mDPR:
```bash
python scripts/run_dense_mdpr.py \
    --config config/neuclir.yaml \
    --lang fas
```

Search with ColBERT:
```bash
python scripts/run_dense_colbert.py \
    --config config/neuclir.yaml \
    --lang rus
```

### 5. Rerank Results

Rerank with monoT5/mT5:
```bash
python scripts/rerank_mt5.py \
    --config config/neuclir.yaml \
    --base_run runs/dense/mdpr_fas.run \
    --lang fas
```

Use multilingual mT5:
```bash
python scripts/rerank_mt5.py \
    --config config/neuclir.yaml \
    --base_run runs/dense/mdpr_rus.run \
    --lang rus \
    --model mt5_multilingual
```

### 6. Evaluate Results

Use TREC `trec_eval`:
```bash
trec_eval -m ndcg_cut.10 \
    data/qrels/fas.qrels.txt \
    runs/reranked/mdpr_fas_mt5.run
```

## Configuration

### Model Configuration

Configure models in `config/neuclir.yaml`:

**mDPR models**:
```yaml
dense:
  mdpr:
    model_name: "facebook/mdpr-question_encoder-base-nq"
    doc_encoder: "facebook/mdpr-ctx_encoder-base-nq"
    query_encoder: "facebook/mdpr-question_encoder-base-nq"
    embedding_dim: 768
```

**ColBERT models**:
```yaml
dense:
  colbert:
    model_name: "colbert-ir/colbertv2.0"
    max_doc_length: 512
    max_query_length: 128
```

**Reranking models**:
```yaml
reranking:
  mt5:
    model_name: "castorini/monot5-base-msmarco-10k"
    batch_size: 32
    top_k: 100
```

## Advanced Usage

### Custom Model Paths

To use a local or custom model:

```yaml
dense:
  mdpr:
    doc_encoder: "/path/to/local/model"
    query_encoder: "/path/to/local/model"
```

### GPU Configuration

Configure GPU usage:

```yaml
system:
  use_gpu: true
  gpu_device: 0  # CUDA device ID
```

For reranking with mixed precision:

```yaml
reranking:
  mt5:
    device: "cuda"
    use_fp16: true  # Use FP16 for faster inference
```

### Batch Processing

Process multiple languages:

```bash
for lang in fas rus zho; do
    python scripts/build_index_dense.py \
        --config config/neuclir.yaml \
        --model mdpr \
        --lang $lang

    python scripts/run_dense_mdpr.py \
        --config config/neuclir.yaml \
        --lang $lang
done
```

## Pipeline Examples

### BM25 Retrieval Pipeline 🆕

```bash
# 1. Build BM25 index
python scripts/build_index_bm25.py \
    --config config/neuclir.yaml \
    --lang fas

# 2. Run BM25 search
python scripts/run_bm25.py \
    --config config/neuclir.yaml \
    --lang fas

# 3. Evaluate
python scripts/evaluate.py \
    --config config/neuclir.yaml \
    --run_dir runs/bm25 \
    --lang fas
```

### Dense Retrieval Pipeline

```bash
# 1. Build dense index
python scripts/build_index_dense.py \
    --config config/neuclir.yaml \
    --model mdpr \
    --lang fas

# 2. Run dense retrieval
python scripts/run_dense_mdpr.py \
    --config config/neuclir.yaml \
    --lang fas

# 3. Evaluate
python scripts/evaluate.py \
    --config config/neuclir.yaml \
    --run runs/dense/mdpr_fas.run \
    --lang fas
```

### Hybrid Retrieval Pipeline 🆕

Combine BM25 and dense retrieval for better results:

```bash
# Method 1: Reciprocal Rank Fusion (RRF)
python scripts/run_hybrid.py \
    --config config/neuclir.yaml \
    --bm25_run runs/bm25/bm25_fas.run \
    --dense_run runs/dense/mdpr_fas.run \
    --lang fas \
    --method rrf

# Method 2: Weighted Fusion (70% BM25, 30% Dense)
python scripts/run_hybrid.py \
    --config config/neuclir.yaml \
    --bm25_run runs/bm25/bm25_fas.run \
    --dense_run runs/dense/mdpr_fas.run \
    --lang fas \
    --method weighted \
    --alpha 0.7
```

### Complete End-to-End Pipeline

With reranking and evaluation:

```bash
# 1. Run BM25
python scripts/run_bm25.py --config config/neuclir.yaml --lang fas

# 2. Rerank with monoT5
python scripts/rerank_mt5.py \
    --config config/neuclir.yaml \
    --base_run runs/bm25/bm25_fas.run \
    --lang fas

# 3. Evaluate reranked results
python scripts/evaluate.py \
    --config config/neuclir.yaml \
    --run runs/reranked/bm25_fas_mt5.run \
    --lang fas
```

### Batch Experiments 🆕

Run complete pipelines automatically:

```bash
# Run full BM25 pipeline for all languages
python scripts/run_experiments.py \
    --config config/neuclir.yaml \
    --pipeline bm25

# Run full dense + reranking pipeline
python scripts/run_experiments.py \
    --config config/neuclir.yaml \
    --pipeline dense_mdpr

# Run everything: BM25 + Dense + Hybrid + Reranking + Evaluation
python scripts/run_experiments.py \
    --config config/neuclir.yaml \
    --pipeline full
```

## Testing 🆕

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test file with verbose output
pytest tests/test_utils_io.py -v

# Run specific test function
pytest tests/test_hybrid.py::test_reciprocal_rank_fusion -v
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes in config:

```yaml
dense:
  mdpr:
    batch_size: 64  # Reduce from 128

reranking:
  mt5:
    batch_size: 16  # Reduce from 32
```

### Missing Dependencies

Install specific extras:

```bash
# For ColBERT
pip install pyserini[colbert]

# For FAISS GPU support
pip install faiss-gpu
```

### Index Not Found

Ensure you've built the index before searching:

```bash
# Check if index exists
ls indexes/dense/mdpr_fas/

# If not, build it first
python scripts/build_index_dense.py --config config/neuclir.yaml --model mdpr --lang fas
```

## Citation

If you use this toolkit, please cite:

```bibtex
@misc{clir_experiments,
  title={Cross-Lingual IR Experiments Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/clir_experiments}
}
```

## License

MIT License - see LICENSE file for details.

## References

- [Pyserini](https://github.com/castorini/pyserini)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [monoT5](https://github.com/castorini/pygaggle)
- [NeuCLIR](https://neuclir.github.io/)
# clir_experiments
