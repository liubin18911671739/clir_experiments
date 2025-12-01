# 开发进度 / Development Progress

## ✅ 已完成 / Completed

### 1. 项目结构搭建 (Project Structure)
- [x] 创建完整的目录结构
  - `config/` - 配置文件目录
  - `data/corpus/`, `data/topics/`, `data/qrels/` - 数据目录
  - `indexes/bm25/`, `indexes/dense/` - 索引目录
  - `runs/bm25/`, `runs/dense/`, `runs/reranked/` - 运行结果目录
  - `scripts/` - 脚本目录

### 2. 配置系统 (Configuration System)
- [x] `config/neuclir.yaml` - 主配置文件
  - 支持多语言配置（波斯语、俄语、中文）
  - mDPR 密集检索模型配置
  - ColBERT 模型配置
  - monoT5/mT5 重排序模型配置
  - 系统参数（GPU、线程数等）

### 3. 工具模块 (Utility Modules)
- [x] `scripts/utils_io.py` (329 行)
  - YAML 配置加载
  - JSONL 语料库读取
  - TREC 格式运行文件读写
  - 目录管理工具

- [x] `scripts/utils_topics.py` (227 行)
  - TREC 主题文件解析（支持 XML 和简单格式）
  - Qrels 文件加载
  - 主题格式转换

### 4. 密集检索管道 (Dense Retrieval Pipeline)
- [x] `scripts/build_index_dense.py` (234 行)
  - mDPR 风格双编码器索引构建
  - ColBERT 索引构建支持
  - 使用 Pyserini 的 AutoDocumentEncoder 和 FaissIndexWriter
  - 批处理文档编码

- [x] `scripts/run_dense_mdpr.py` (146 行)
  - mDPR 密集检索搜索
  - 使用 Pyserini 的 FaissSearcher 和 AutoQueryEncoder
  - TREC 格式结果输出

- [x] `scripts/run_dense_colbert.py` (153 行)
  - ColBERT 晚期交互检索
  - Pyserini ColBERT 支持集成

### 5. 重排序管道 (Reranking Pipeline)
- [x] `scripts/rerank_mt5.py` (329 行)
  - monoT5/mT5 序列到序列重排序
  - 标准 monoT5 输入格式实现
  - 批处理 GPU 推理
  - FP16 混合精度支持
  - 从基础运行文件读取并重排序 top-k 文档

### 6. 文档与示例 (Documentation & Examples)
- [x] `README.md` - 完整使用文档
  - 快速开始指南
  - 详细配置说明
  - 使用示例
  - 故障排除指南

- [x] `CONTRIBUTING.md` - 开发贡献指南
  - 开发环境设置
  - 代码风格指南
  - 添加新功能的流程
  - 测试指南

- [x] `requirements.txt` - 依赖项清单
  - Pyserini（IR 工具包）
  - Transformers（重排序模型）
  - FAISS（向量检索）
  - 其他必需依赖

- [x] `.gitignore` - Git 忽略规则

- [x] 示例数据文件
  - `data/corpus/example.jsonl`
  - `data/topics/example.topics.txt`
  - `data/qrels/example.qrels.txt`

### 7. BM25 检索管道 (BM25 Retrieval Pipeline) **🆕 NEW**
- [x] `scripts/build_index_bm25.py` (270 行)
  - 使用 Pyserini/Anserini 构建 Lucene 索引
  - 支持 JSONL 语料库格式
  - 多线程索引构建

- [x] `scripts/run_bm25.py` (180 行)
  - BM25 检索实现（可配置 k1, b 参数）
  - 批量处理多语言
  - TREC 格式输出

### 8. 自动评估系统 (Automatic Evaluation) **🆕 NEW**
- [x] `scripts/evaluate.py` (290 行)
  - 调用 trec_eval 进行自动评估
  - 支持单个运行文件或批量目录评估
  - JSON 格式结果输出
  - 比较表格生成

### 9. 混合检索 (Hybrid Retrieval) **🆕 NEW**
- [x] `scripts/run_hybrid.py` (240 行)
  - Reciprocal Rank Fusion (RRF)
  - 线性组合融合
  - 加权融合（可配置权重）
  - BM25 + Dense 结果合并

### 10. 批量实验管理 (Batch Experiments) **🆕 NEW**
- [x] `scripts/run_experiments.py` (280 行)
  - 端到端流水线编排
  - 支持 BM25、Dense、Reranking、完整流水线
  - 多语言批量处理
  - 自动错误处理和日志记录

### 11. 单元测试套件 (Unit Tests) **🆕 NEW**
- [x] `tests/test_utils_io.py` - I/O 工具测试
- [x] `tests/test_utils_topics.py` - 主题解析测试
- [x] `tests/test_hybrid.py` - 混合检索测试
- [x] `pytest.ini` - Pytest 配置

## 📊 代码统计 / Code Statistics

- **总代码行数**: ~3,800+ 行 ⬆️ (从 1,574 行)
- **Python 脚本**: 11 个 ⬆️ (从 6 个)
- **单元测试**: 3 个测试文件，15+ 测试用例
- **工具函数**: 50+ 个 ⬆️
- **配置选项**: 50+ 个参数

## 🎯 功能特性 / Features

### 已实现功能 (Implemented)
1. ✅ mDPR 风格密集检索（文档编码 + FAISS 索引 + 查询搜索）
2. ✅ ColBERT 晚期交互检索支持
3. ✅ monoT5/mT5 神经重排序
4. ✅ 完整的配置驱动架构
5. ✅ TREC 格式兼容（主题、运行文件、qrels）
6. ✅ 批处理和 GPU 加速
7. ✅ 多语言支持（波斯语、俄语、中文）
8. ✅ **BM25 索引构建和检索** 🆕
9. ✅ **自动评估（trec_eval 集成）** 🆕
10. ✅ **混合检索（RRF、线性融合、加权融合）** 🆕
11. ✅ **批量实验运行脚本** 🆕
12. ✅ **单元测试套件** 🆕

### 🎉 新增功能 v2.5.0 (New in v2.5.0)
- [x] **查询扩展支持** - RM3 和 PRF 实现 🆕
- [x] **更多融合策略** - CombSUM 和 CombMNZ 🆕
- [x] **实验结果可视化** - Markdown表格和ASCII图表 🆕

### 待扩展功能 (Future Enhancements)
- [ ] 跨语言查询翻译（机器翻译集成）
- [ ] 性能基准测试和分析
- [ ] 在线检索 API 服务（FastAPI）
- [ ] 交互式检索界面（Streamlit/Gradio）
- [ ] 高级可视化（matplotlib/plotly图表）

## 🔄 当前状态 / Current Status

**状态**: ✅ **核心功能开发完成，可投入使用**

该工具包现在可以用于：
- NeuCLIR / CAKE-ILC 风格的跨语言 IR 实验
- 密集检索实验（mDPR、ColBERT）
- 神经重排序实验（monoT5/mT5）
- 端到端检索管道：语料库 → 索引 → 检索 → 重排序 → 评估

## 📝 使用流程 / Workflow

### 方式 1: 使用批量实验脚本（推荐）

```bash
# 运行完整的 BM25 流水线
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline bm25

# 运行完整的密集检索流水线
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline dense_mdpr

# 运行重排序流水线
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline rerank

# 运行完整的端到端流水线（BM25 + Dense + Reranking + Evaluation）
python scripts/run_experiments.py --config config/neuclir.yaml --pipeline full
```

### 方式 2: 手动运行各个步骤

```bash
# 1. 准备数据
# 将 JSONL 语料库放入 data/corpus/{lang}/
# 将 TREC 主题放入 data/topics/{lang}.topics.txt
# 将 qrels 放入 data/qrels/{lang}.qrels.txt

# 2a. 构建 BM25 索引
python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas

# 2b. 或者构建密集索引
python scripts/build_index_dense.py --config config/neuclir.yaml --model mdpr --lang fas

# 3a. 运行 BM25 检索
python scripts/run_bm25.py --config config/neuclir.yaml --lang fas

# 3b. 或者运行密集检索
python scripts/run_dense_mdpr.py --config config/neuclir.yaml --lang fas

# 4. （可选）运行混合检索
python scripts/run_hybrid.py --config config/neuclir.yaml \
    --bm25_run runs/bm25/bm25_fas.run \
    --dense_run runs/dense/mdpr_fas.run \
    --lang fas --method rrf

# 5. 重排序
python scripts/rerank_mt5.py --config config/neuclir.yaml \
    --base_run runs/bm25/bm25_fas.run --lang fas

# 6. 自动评估
python scripts/evaluate.py --config config/neuclir.yaml \
    --run_dir runs/reranked --lang fas

# 或者使用 trec_eval
trec_eval -m ndcg_cut.10 data/qrels/fas.qrels.txt runs/reranked/bm25_fas_mt5.run
```

### 方式 3: 运行测试

```bash
# 运行所有单元测试
pytest tests/

# 运行特定测试文件
pytest tests/test_utils_io.py -v

# 运行特定测试函数
pytest tests/test_hybrid.py::test_reciprocal_rank_fusion -v
```

## 🚀 下一步计划 / Next Steps

### ✅ 已完成优先级 1-2 (Completed Priority 1-2)
1. ✅ BM25 索引和检索脚本
2. ✅ 自动评估脚本
3. ✅ 单元测试套件
4. ✅ 混合检索支持
5. ✅ 批量实验运行脚本

### 优先级 3 (Priority 3) - 高级功能
1. 查询扩展（RM3、Pseudo-Relevance Feedback）
2. 跨语言查询翻译（集成 MT 服务）
3. 结果可视化工具（图表生成、对比分析）
4. 性能基准测试工具

### 优先级 4 (Priority 4) - 生产化
1. 在线检索 API 服务（FastAPI/Flask）
2. 交互式 Web 界面
3. Docker 容器化部署
4. 分布式检索支持

## 📌 注意事项 / Notes

- 所有脚本都包含完整的类型提示和文档字符串
- 配置文件使用 YAML 格式，易于修改
- 支持 CPU 和 GPU 运行模式
- 遵循 TREC 标准格式，便于评估
- 代码结构清晰，易于扩展

---

**最后更新**: 2025-12-01
**版本**: v2.5.0 🚀
**状态**: 功能丰富，生产就绪 (Feature-Rich & Production Ready)
**新增**: 查询扩展(RM3/PRF)、CombSUM/CombMNZ融合、结果可视化
**代码量**: ~4,077 行 (+22.4%)
