# 多模态情感分析项目

本项目基于 CMU-MOSEI 数据集，实现了一个多模态情感分析系统，结合文本和音频特征进行情感分类。

## 项目进展

- [x] 数据集预处理
  - 实现了数据对齐和清洗
  - 添加了五分类情感标签（非常消极、消极、中性、积极、非常积极）
  - 支持数据子集采样
- [x] 模型实现
  - 文本模态：基于 BERT 的文本特征提取
  - 音频模态：基于 COVAREP 的音频特征处理
  - 决策融合：实现了基于权重的多模态融合
- [x] 训练系统
  - 实现了混合精度训练（AMP）
  - 添加了早停机制
  - 支持模型检查点保存
- [x] 可视化工具
  - 训练过程可视化（损失、准确率、融合权重）
  - 情感分布分析

## 环境配置

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备

1. 下载 CMU-MOSEI 数据集
2. 将数据集放在 `data` 目录下
3. 运行预处理脚本：
```bash
python preprocessing/preprocess_mosei.py
```

## 模型训练

1. 单模态训练（文本）：
```bash
python train_cuda.py
```

2. 决策融合模型训练：
```bash
python train_decision_fusion_cuda.py
```

## 模型测试

1. 单模态模型测试：
```bash
python test_cuda.py
```

2. 决策融合模型测试：
```bash
python test_decision_fusion_cuda.py
```

## 可视化

1. 训练过程可视化：
```bash
python visualize_training.py
```

2. 情感分布分析：
```bash
python preprocessing/analyze_labels.py
```

## 项目结构

```
.
├── data/                    # 数据集目录
├── models/                  # 模型定义
│   ├── text_model.py       # 文本模型
│   ├── audio_model.py      # 音频模型
│   └── fusion_model.py     # 融合模型
├── preprocessing/          # 数据预处理
│   ├── preprocess_mosei.py # 主预处理脚本
│   └── analyze_labels.py   # 标签分析工具
├── train_cuda.py          # 单模态训练
├── train_decision_fusion_cuda.py  # 融合模型训练
├── test_cuda.py           # 单模态测试
├── test_decision_fusion_cuda.py   # 融合模型测试
├── visualize_training.py   # 训练可视化
└── requirements.txt        # 项目依赖
```

## 注意事项

1. 确保有足够的 GPU 内存（推荐至少 8GB）
2. 训练过程中会自动保存最佳模型
3. 可以通过修改配置文件调整模型参数
4. 支持数据子集采样，便于快速实验

## 后续计划

- [ ] 添加更多评估指标
- [ ] 实现交叉验证
- [ ] 添加模型解释性分析
- [ ] 优化训练效率
- [ ] 添加更多可视化功能
