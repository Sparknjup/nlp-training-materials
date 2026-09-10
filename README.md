# NLP 部员培训材料

面向初学者设计的自然语言处理培训材料，覆盖从文本分词到循环神经网络的基础路径。仓库包含 5 讲讲义和可直接运行的 TensorFlow/Keras 示例。

## 内容结构

1. NLP 概述、应用与发展历史
2. Tokenizer、词表和词频限制
3. 文本序列化、OOV、Padding 与 Truncation
4. 基于新闻标题的讽刺文本分类
5. RNN、长距离依赖与 LSTM 入门

## 示例

- `examples/tokenization_demo.py`：分词、序列化和补齐
- `examples/dataset_preprocessing.py`：读取 JSON Lines 数据并构建张量
- `examples/sarcasm_classifier.py`：Embedding、GlobalAveragePooling 和二分类模型

## 数据准备

训练示例使用 `Sarcasm_Headlines_Dataset.json`，共 26,709 条新闻标题。出于数据许可与仓库体积考虑，本仓库不重新分发数据集。下载后将文件放入 `data/` 目录即可运行。

```bash
python -m pip install -r requirements.txt
python examples/sarcasm_classifier.py
```

## 项目职责

朱磊负责课程材料设计、代码示例整理与部员培训。
