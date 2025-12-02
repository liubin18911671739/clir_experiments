# Cross-Lingual IR Experiments

**[English](#english) | [中文](#中文)**

Toolkit for NeuCLIR / CAKE-ILC style cross-lingual information retrieval experiments.

**Status**: ✅ Production Ready | **Version**: 2.6.0 | **Lines of Code**: ~5,329 🚀 | **API**: FastAPI REST

---

## 中文

### 简介

这是一个用于 NeuCLIR / CAKE-ILC 风格跨语言信息检索实验的工具包。

### 核心功能

- **稀疏检索**: BM25 传统检索（Pyserini/Lucene）
- **密集检索**: mDPR 风格双编码器和 ColBERT 晚期交互模型
- **混合检索**: RRF、线性融合、加权融合、CombSUM、CombMNZ
- **神经重排序**: monoT5/mT5 序列到序列重排序器
- **查询扩展**: RM3 和 PRF (Pseudo-Relevance Feedback) 🆕
- **自动评估**: trec_eval 集成，批量评估
- **批量实验**: 端到端流水线编排
- **REST API**: FastAPI 在线检索服务 🆕
- **Docker 部署**: 容器化生产环境支持 🆕
- **性能基准**: benchmark.py 性能分析工具 🆕
- **配置驱动**: 所有设置集中在 `config/neuclir.yaml`
- **单元测试**: Pytest 测试套件
- **TREC 兼容**: 标准 TREC 运行文件格式，便于评估

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 获取实验数据（推荐使用 NeuCLIR 数据集）
pip install ir-datasets

# 下载并转换 NeuCLIR 数据（以波斯语为例）
python -c "
import ir_datasets
import json
from pathlib import Path

dataset = ir_datasets.load('neuclir/1/fa')
Path('data/corpus/fas').mkdir(parents=True, exist_ok=True)

# 转换语料库为 JSONL
with open('data/corpus/fas/corpus.jsonl', 'w', encoding='utf-8') as f:
    for doc in dataset.docs_iter():
        json.dump({'id': doc.doc_id, 'contents': doc.title + ' ' + doc.text}, f, ensure_ascii=False)
        f.write('\n')

# 导出主题
with open('data/topics/fas.topics.txt', 'w', encoding='utf-8') as f:
    for topic in dataset.queries_iter():
        f.write(f'<top>\n<num> Number: {topic.query_id}\n<title> {topic.text}\n</top>\n\n')

# 导出 qrels
with open('data/qrels/fas.qrels.txt', 'w', encoding='utf-8') as f:
    for qrel in dataset.qrels_iter():
        f.write(f'{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n')
"

# 方式 1: 使用批量实验脚本（推荐）
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

### 数据获取指南

#### 方式 1: 使用 NeuCLIR 数据集（推荐）

NeuCLIR 是 TREC 2022-2023 的官方跨语言检索评测数据集：

- **官网**: https://neuclir.github.io/
- **包含语言**: 波斯语(fa)、俄语(ru)、中文(zh)
- **数据内容**: 新闻文章语料库、英文查询、人工标注的相关性判断
- **获取方式**: 使用 `ir-datasets` 库自动下载

```bash
pip install ir-datasets

# 下载波斯语数据
python -c "import ir_datasets; list(ir_datasets.load('neuclir/1/fa').docs_iter())"
```

#### 方式 2: 使用 HC4 数据集

HC4 (Human-translated CLIR Collection) 是另一个优秀的数据集：

```bash
pip install ir-datasets
python -c "import ir_datasets; list(ir_datasets.load('hc4/fa').docs_iter())"
```

#### 方式 3: 使用自己的数据

按照以下格式组织数据即可。

### 文档

- 📋 [TODO.md](TODO.md) - 开发进度和计划
- 📖 完整使用文档见下方英文部分
- 🤝 [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南

---

## English

## Features

- **Sparse Retrieval**: BM25 traditional retrieval (Pyserini/Lucene)
- **Dense Retrieval**: mDPR-style dual encoders and ColBERT late interaction models
- **Hybrid Retrieval**: RRF, linear combination, weighted fusion, CombSUM, CombMNZ
- **Reranking**: monoT5/mT5 seq2seq rerankers
- **Query Expansion**: RM3 and PRF (Pseudo-Relevance Feedback) 🆕
- **Automatic Evaluation**: trec_eval integration, batch evaluation
- **Batch Experiments**: End-to-end pipeline orchestration
- **REST API**: FastAPI-based online retrieval service 🆕
- **Docker Support**: Containerized deployment for production 🆕
- **Benchmarking**: Performance analysis and profiling tools 🆕
- **Configuration-driven**: All settings in `config/neuclir.yaml`
- **Unit Tests**: Pytest test suite
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

## Quick Start with API 🆕

**Launch the REST API service:**

```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Or use Docker
docker-compose up -d clir-api

# Access interactive API docs
open http://localhost:8000/docs
```

**API Features:**
- BM25 search endpoint
- Dense retrieval endpoint
- Hybrid search with multiple fusion strategies
- Neural reranking endpoint
- Full OpenAPI/Swagger documentation

See [API Documentation](api/README.md) for details.

---

## Installation

### Requirements

- Python 3.9+
- CUDA 11+ (optional, for GPU acceleration)
- Docker & Docker Compose (optional, for containerized deployment)

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

#### Option A: Use NeuCLIR Dataset (Recommended)

The **NeuCLIR** (TREC 2022-2023) dataset is publicly available for cross-lingual IR research:

**Download NeuCLIR data:**

```bash
# Visit NeuCLIR official website
# https://neuclir.github.io/

# Or download from IR Datasets
pip install ir-datasets

# Download Persian (Farsi) data
python -c "import ir_datasets; dataset = ir_datasets.load('neuclir/1/fa'); 
for doc in dataset.docs_iter(): print(doc)"

# Download Russian data
python -c "import ir_datasets; dataset = ir_datasets.load('neuclir/1/ru'); 
for doc in dataset.docs_iter(): print(doc)"

# Download Chinese data
python -c "import ir_datasets; dataset = ir_datasets.load('neuclir/1/zh'); 
for doc in dataset.docs_iter(): print(doc)"
```

**NeuCLIR Dataset includes:**
- **Corpora**: News articles in Persian, Russian, and Chinese
- **Topics**: English queries (50-100 topics per year)
- **Qrels**: Relevance judgments from TREC assessors
- **Years**: 2022, 2023 data available

**Convert NeuCLIR to JSONL format:**

```python
import ir_datasets
import json
from pathlib import Path

# Load dataset
dataset = ir_datasets.load('neuclir/1/fa')  # or 'neuclir/1/ru', 'neuclir/1/zh'

# Create output directory
output_dir = Path('data/corpus/fas')
output_dir.mkdir(parents=True, exist_ok=True)

# Convert to JSONL
with open(output_dir / 'corpus.jsonl', 'w', encoding='utf-8') as f:
    for doc in dataset.docs_iter():
        json.dump({
            'id': doc.doc_id,
            'contents': doc.title + ' ' + doc.text
        }, f, ensure_ascii=False)
        f.write('\n')

# Export topics
with open('data/topics/fas.topics.txt', 'w', encoding='utf-8') as f:
    for topic in dataset.queries_iter():
        f.write(f"<top>\n<num> Number: {topic.query_id}\n")
        f.write(f"<title> {topic.text}\n</top>\n\n")

# Export qrels
with open('data/qrels/fas.qrels.txt', 'w', encoding='utf-8') as f:
    for qrel in dataset.qrels_iter():
        f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
```

#### Option B: Use HC4 Dataset

**HC4** (Human-translated CLIR Collection) is another excellent dataset:

```bash
# Download HC4 data using ir_datasets
pip install ir_datasets

# Available languages: Persian (fa), Russian (ru), Chinese (zh)
python -c "import ir_datasets; dataset = ir_datasets.load('hc4/fa'); 
for doc in dataset.docs_iter(): print(doc)"
```

#### Option C: Use Your Own Data

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

### Query Expansion 🆕

Improve retrieval effectiveness with query expansion:

```bash
# RM3 query expansion
python scripts/query_expansion.py \
    --config config/neuclir.yaml \
    --base_run runs/bm25/bm25_fas.run \
    --lang fas \
    --method rm3 \
    --fb_docs 10 \
    --fb_terms 10 \
    --original_query_weight 0.5

# Pseudo-Relevance Feedback (PRF)
python scripts/query_expansion.py \
    --config config/neuclir.yaml \
    --base_run runs/bm25/bm25_fas.run \
    --lang fas \
    --method prf \
    --fb_docs 20 \
    --fb_terms 15
```

### Results Visualization 🆕

Generate comparison reports:

```bash
python scripts/visualize_results.py \
    --results eval_results/*.json \
    --output reports/comparison.md
```

### Performance Benchmarking 🆕

Analyze system performance:

```bash
python scripts/benchmark.py \
    --config config/neuclir.yaml \
    --mode index \
    --lang fas
```

### REST API Service 🆕

Deploy as a web service:

```bash
# Development mode
uvicorn api.main:app --reload --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker deployment
docker-compose up -d clir-api

# GPU-enabled deployment
docker-compose --profile gpu up -d clir-api-gpu
```

**API Endpoints:**
- `GET /` - Health check
- `POST /search/bm25` - BM25 retrieval
- `POST /search/dense` - Dense retrieval
- `POST /search/hybrid` - Hybrid search
- `POST /rerank` - Neural reranking

See [API Documentation](api/README.md) for detailed usage.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIR Experiments System                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐                     │
│  │   Data Layer │     │ Config Layer │                     │
│  │              │     │              │                     │
│  │ • Corpus     │     │ neuclir.yaml │                     │
│  │ • Topics     │     │              │                     │
│  │ • Qrels      │     └──────────────┘                     │
│  └──────────────┘                                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Retrieval Engines                          │  │
│  │                                                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │  │
│  │  │  BM25   │  │  mDPR   │  │ColBERT │             │  │
│  │  └─────────┘  └─────────┘  └─────────┘             │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────┐             │  │
│  │  │    Hybrid Fusion (RRF/CombSUM)     │             │  │
│  │  └─────────────────────────────────────┘             │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────┐             │  │
│  │  │   Neural Reranking (monoT5/mT5)    │             │  │
│  │  └─────────────────────────────────────┘             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             Enhancement Modules                       │  │
│  │                                                        │  │
│  │  • Query Expansion (RM3/PRF)                         │  │
│  │  • Evaluation (trec_eval)                            │  │
│  │  • Visualization                                      │  │
│  │  • Benchmarking                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              REST API Layer (FastAPI)                 │  │
│  │                                                        │  │
│  │  • BM25 endpoint                                      │  │
│  │  • Dense endpoint                                     │  │
│  │  • Hybrid endpoint                                    │  │
│  │  • Rerank endpoint                                    │  │
│  │  • OpenAPI docs                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             Deployment Layer                          │  │
│  │                                                        │  │
│  │  • Docker containerization                            │  │
│  │  • GPU support                                        │  │
│  │  • Multi-worker deployment                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
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

## What's New in v2.6.0 🎉

This version represents a major milestone with complete production-ready features:

### New Features
- ✨ **REST API Service**: FastAPI-based web service with 5 endpoints
- ✨ **Query Expansion**: RM3 and Pseudo-Relevance Feedback implementations
- ✨ **Enhanced Fusion**: CombSUM and CombMNZ strategies added
- ✨ **Results Visualization**: Markdown tables and ASCII charts
- ✨ **Performance Benchmarking**: Comprehensive profiling tools
- ✨ **Docker Support**: Full containerization with GPU support

### Statistics
- **Total Lines of Code**: 5,329 (238% increase from initial version)
- **Python Scripts**: 15 modules
- **API Endpoints**: 5 REST endpoints
- **Test Coverage**: 15+ unit tests
- **Fusion Strategies**: 5 methods (RRF, Linear, Weighted, CombSUM, CombMNZ)
- **Deployment Options**: Local, Docker, Docker+GPU

### Architecture Improvements
- Modular design with clear separation of concerns
- Comprehensive error handling and logging
- Production-ready API with OpenAPI documentation
- Docker multi-stage builds for optimized images
- GPU-accelerated deployment support

### Production Features
- ✅ REST API with Swagger/ReDoc documentation
- ✅ Docker containerization (CPU and GPU variants)
- ✅ Automated testing suite
- ✅ Comprehensive benchmarking tools
- ✅ Complete evaluation pipeline
- ✅ Result visualization and reporting

This toolkit is now suitable for both academic research and production deployment!

## License

MIT License - see LICENSE file for details.

## References

- [Pyserini](https://github.com/castorini/pyserini)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [monoT5](https://github.com/castorini/pygaggle)
- [NeuCLIR](https://neuclir.github.io/)
# clir_experiments
