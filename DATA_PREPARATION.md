# 数据准备指南 / Data Preparation Guide

本文档详细说明如何获取和准备跨语言信息检索实验所需的数据。

---

## 快速开始

### 自动下载脚本

我们提供了自动下载和转换脚本（推荐使用）：

```bash
# 下载并转换 NeuCLIR 数据集
python scripts/download_neuclir.py --lang fas --year 2022

# 批量下载所有语言
python scripts/download_neuclir.py --all
```

---

## 方式 1: 使用 NeuCLIR 数据集（推荐）

### 关于 NeuCLIR

**NeuCLIR** (Neural Cross-Lingual Information Retrieval) 是 TREC 2022-2023 的官方评测数据集。

- **官网**: https://neuclir.github.io/
- **论文**: [NeuCLIR 2022 Overview](https://trec.nist.gov/pubs/trec31/papers/Overview_neuclir.pdf)
- **支持语言**: 
  - Persian (波斯语, fa)
  - Russian (俄语, ru)
  - Chinese (中文, zh)
- **数据规模**:
  - 语料库: 每个语言 2-3M 文档
  - 主题: 每年 50 个英文查询
  - Qrels: NIST 人工标注

### 安装依赖

```bash
pip install ir-datasets
```

### 手动下载和转换

#### 1. 下载波斯语数据

```python
import ir_datasets
import json
from pathlib import Path

# 加载 NeuCLIR 2022 波斯语数据集
dataset = ir_datasets.load('neuclir/1/fa')

# 创建输出目录
Path('data/corpus/fas').mkdir(parents=True, exist_ok=True)
Path('data/topics').mkdir(parents=True, exist_ok=True)
Path('data/qrels').mkdir(parents=True, exist_ok=True)

# 1. 转换语料库为 JSONL 格式
print("Converting corpus to JSONL...")
with open('data/corpus/fas/corpus.jsonl', 'w', encoding='utf-8') as f:
    for idx, doc in enumerate(dataset.docs_iter()):
        if idx % 10000 == 0:
            print(f"Processed {idx} documents...")
        
        json.dump({
            'id': doc.doc_id,
            'contents': f"{doc.title} {doc.text}".strip()
        }, f, ensure_ascii=False)
        f.write('\n')

print("Corpus conversion complete!")

# 2. 导出 TREC 主题文件
print("Exporting topics...")
with open('data/topics/fas.topics.txt', 'w', encoding='utf-8') as f:
    for topic in dataset.queries_iter():
        f.write(f"<top>\n")
        f.write(f"<num> Number: {topic.query_id}\n")
        f.write(f"<title> {topic.text}\n")
        f.write(f"</top>\n\n")

print("Topics export complete!")

# 3. 导出 Qrels 文件
print("Exporting qrels...")
with open('data/qrels/fas.qrels.txt', 'w', encoding='utf-8') as f:
    for qrel in dataset.qrels_iter():
        # TREC qrels format: query_id 0 doc_id relevance
        f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")

print("Qrels export complete!")
print("\nData preparation finished! You can now run experiments.")
```

#### 2. 下载俄语数据

```python
import ir_datasets
import json
from pathlib import Path

# 加载俄语数据
dataset = ir_datasets.load('neuclir/1/ru')

# 创建目录
Path('data/corpus/rus').mkdir(parents=True, exist_ok=True)

# 转换数据（同上）
# ... (代码同波斯语，将 fas 改为 rus)
```

#### 3. 下载中文数据

```python
import ir_datasets
import json
from pathlib import Path

# 加载中文数据
dataset = ir_datasets.load('neuclir/1/zh')

# 创建目录
Path('data/corpus/zho').mkdir(parents=True, exist_ok=True)

# 转换数据（同上）
# ... (代码同波斯语，将 fas 改为 zho)
```

### 使用完整脚本

创建 `scripts/download_neuclir.py`:

```python
#!/usr/bin/env python3
"""
Download and convert NeuCLIR dataset to local format.

Usage:
    python scripts/download_neuclir.py --lang fas --year 2022
    python scripts/download_neuclir.py --all
"""

import argparse
import ir_datasets
import json
from pathlib import Path
from tqdm import tqdm

LANG_MAP = {
    'fas': 'fa',  # Persian
    'rus': 'ru',  # Russian
    'zho': 'zh',  # Chinese
}

def download_neuclir(lang: str, year: int = 1):
    """Download NeuCLIR data for specified language."""
    
    # Map internal language code to ir_datasets code
    ir_lang = LANG_MAP.get(lang, lang)
    dataset_name = f'neuclir/{year}/{ir_lang}'
    
    print(f"Loading NeuCLIR dataset: {dataset_name}")
    dataset = ir_datasets.load(dataset_name)
    
    # Create directories
    corpus_dir = Path(f'data/corpus/{lang}')
    corpus_dir.mkdir(parents=True, exist_ok=True)
    Path('data/topics').mkdir(parents=True, exist_ok=True)
    Path('data/qrels').mkdir(parents=True, exist_ok=True)
    
    # 1. Convert corpus
    print(f"\n[1/3] Converting corpus for {lang}...")
    corpus_file = corpus_dir / 'corpus.jsonl'
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(dataset.docs_iter(), desc="Documents"):
            json.dump({
                'id': doc.doc_id,
                'contents': f"{doc.title} {doc.text}".strip()
            }, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ Corpus saved to {corpus_file}")
    
    # 2. Export topics
    print(f"\n[2/3] Exporting topics for {lang}...")
    topics_file = Path(f'data/topics/{lang}.topics.txt')
    
    with open(topics_file, 'w', encoding='utf-8') as f:
        for topic in dataset.queries_iter():
            f.write(f"<top>\n")
            f.write(f"<num> Number: {topic.query_id}\n")
            f.write(f"<title> {topic.text}\n")
            f.write(f"</top>\n\n")
    
    print(f"✓ Topics saved to {topics_file}")
    
    # 3. Export qrels
    print(f"\n[3/3] Exporting qrels for {lang}...")
    qrels_file = Path(f'data/qrels/{lang}.qrels.txt')
    
    with open(qrels_file, 'w', encoding='utf-8') as f:
        for qrel in tqdm(dataset.qrels_iter(), desc="Qrels"):
            f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
    
    print(f"✓ Qrels saved to {qrels_file}")
    print(f"\n✅ Data preparation complete for {lang}!")

def main():
    parser = argparse.ArgumentParser(description='Download NeuCLIR dataset')
    parser.add_argument('--lang', type=str, choices=['fas', 'rus', 'zho', 'all'],
                       help='Language to download (or "all" for all languages)')
    parser.add_argument('--year', type=int, default=1, choices=[1, 2],
                       help='NeuCLIR year (1=2022, 2=2023)')
    
    args = parser.parse_args()
    
    if args.lang == 'all':
        for lang in ['fas', 'rus', 'zho']:
            print(f"\n{'='*60}")
            print(f"Processing {lang.upper()}")
            print(f"{'='*60}")
            download_neuclir(lang, args.year)
    else:
        download_neuclir(args.lang, args.year)

if __name__ == '__main__':
    main()
```

运行脚本：

```bash
# 安装依赖
pip install ir-datasets tqdm

# 下载单个语言
python scripts/download_neuclir.py --lang fas --year 1

# 下载所有语言
python scripts/download_neuclir.py --all --year 1
```

---

## 方式 2: 使用 HC4 数据集

### 关于 HC4

**HC4** (Human-translated CLIR Collection) 包含人工翻译的查询。

- **论文**: [HC4 Paper](https://arxiv.org/abs/2201.09992)
- **支持语言**: Persian, Russian, Chinese + 更多
- **特点**: 人工翻译的查询，质量更高

### 下载 HC4 数据

```python
import ir_datasets

# 加载 HC4 波斯语数据
dataset = ir_datasets.load('hc4/fa')

# 转换方式同 NeuCLIR
# ...
```

---

## 方式 3: 使用自己的数据

### 数据格式要求

#### 1. 语料库格式 (JSONL)

每行一个JSON对象：

```json
{"id": "doc001", "contents": "This is the document text..."}
{"id": "doc002", "contents": "Another document..."}
```

**字段说明**:
- `id`: 文档唯一标识符
- `contents`: 文档内容（标题+正文）

#### 2. 主题格式 (TREC XML)

```xml
<top>
<num> Number: 1
<title> machine learning applications
</top>

<top>
<num> Number: 2
<title> neural networks
</top>
```

或简化格式：

```
1    machine learning applications
2    neural networks
```

#### 3. Qrels 格式 (TREC)

```
query_id 0 doc_id relevance
1 0 doc001 2
1 0 doc002 1
2 0 doc005 2
```

**字段说明**:
- `query_id`: 查询ID
- `0`: 占位符（TREC标准格式）
- `doc_id`: 文档ID
- `relevance`: 相关性等级（0=不相关，1=相关，2=高度相关）

### 目录结构

```
data/
├── corpus/
│   ├── fas/
│   │   └── corpus.jsonl
│   ├── rus/
│   │   └── corpus.jsonl
│   └── zho/
│       └── corpus.jsonl
├── topics/
│   ├── fas.topics.txt
│   ├── rus.topics.txt
│   └── zho.topics.txt
└── qrels/
    ├── fas.qrels.txt
    ├── rus.qrels.txt
    └── zho.qrels.txt
```

---

## 数据验证

### 验证数据格式

```bash
# 检查语料库
head -n 5 data/corpus/fas/corpus.jsonl

# 检查主题
head -n 20 data/topics/fas.topics.txt

# 检查 qrels
head -n 10 data/qrels/fas.qrels.txt
```

### 统计数据

```python
import json
from pathlib import Path

# 统计文档数量
corpus_file = Path('data/corpus/fas/corpus.jsonl')
doc_count = sum(1 for _ in open(corpus_file, encoding='utf-8'))
print(f"Total documents: {doc_count:,}")

# 统计主题数量
topics_file = Path('data/topics/fas.topics.txt')
topic_count = len([l for l in open(topics_file) if l.startswith('<num>')])
print(f"Total topics: {topic_count}")

# 统计 qrels 数量
qrels_file = Path('data/qrels/fas.qrels.txt')
qrel_count = sum(1 for _ in open(qrels_file))
print(f"Total qrels: {qrel_count:,}")
```

---

## 其他数据集推荐

### 学术数据集

1. **CLEF eHealth** - 医疗领域多语言检索
   - https://clefehealth.imag.fr/

2. **NTCIR** - 亚洲语言信息检索
   - http://research.nii.ac.jp/ntcir/

3. **FIRE** - 印度语言信息检索
   - http://fire.irsi.res.in/

4. **MATERIAL** - 低资源语言检索
   - https://www.iarpa.gov/research-programs/material

### 商业数据集

1. **Microsoft MARCO** - 大规模检索数据集
   - https://microsoft.github.io/msmarco/

2. **Common Crawl** - 多语言网页数据
   - https://commoncrawl.org/

---

## 常见问题

### Q: 下载速度慢怎么办？

A: `ir-datasets` 会自动缓存数据，首次下载较慢，后续会很快。可以设置代理：

```bash
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Q: 磁盘空间不足？

A: NeuCLIR 每个语言约需 5-10GB 空间。可以只下载需要的语言。

### Q: 如何使用自定义语料库？

A: 按照 JSONL 格式组织数据，放入 `data/corpus/{lang}/` 目录即可。

### Q: 是否支持其他格式？

A: 目前支持 JSONL 和 TREC 格式。如需其他格式，可以编写转换脚本。

---

## 下一步

数据准备完成后，可以开始构建索引和运行实验：

```bash
# 构建 BM25 索引
python scripts/build_index_bm25.py --config config/neuclir.yaml --lang fas

# 运行检索实验
python scripts/run_bm25.py --config config/neuclir.yaml --lang fas

# 评估结果
python scripts/evaluate.py --config config/neuclir.yaml --run_dir runs/bm25 --lang fas
```

详见 [README.md](README.md) 获取完整使用指南。
