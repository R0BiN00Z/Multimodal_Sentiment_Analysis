# 多模态情感分析项目

本项目实现了一个基于CMU-MOSEI数据集的多模态情感分析系统，使用文本、语音和视觉特征进行情感分类。

## 项目结构

```
.
├── config/               # 配置管理
│   ├── __init__.py
│   ├── model_config.py  # 模型配置
│   ├── data_config.py   # 数据配置
│   ├── train_config.py  # 训练配置
│   └── config_manager.py # 配置管理器
├── models/              # 模型定义
│   ├── __init__.py
│   └── multimodal_model.py
├── preprocessing/       # 数据预处理
│   ├── __init__.py
│   └── preprocess_mosei.py
├── experiment/          # 实验管理
│   ├── __init__.py
│   ├── experiment_config.py
│   ├── experiment_manager.py
│   └── experiments/    # 具体实验
│       └── baseline_experiment.py
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── utils.py            # 工具函数
└── README.md           # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- NumPy
- scikit-learn
- tqdm
- pandas
- matplotlib
- seaborn
- mmsdk

## 安装

1. 克隆项目：
```bash
git clone [repository-url]
cd multimodal_sentiment_analysis
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

### 1. 模型配置 (config/model_config.py)

```python
@dataclass
class ModelConfig:
    # 文本编码器
    text_model_name: str = "bert-base-uncased"
    text_hidden_dim: int = 768
    
    # 语音编码器
    acoustic_input_dim: int = 74
    acoustic_hidden_dim: int = 256
    
    # 视觉编码器
    visual_input_dim: int = 35
    visual_hidden_dim: int = 256
    
    # 特征融合
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.1
    
    # 分类器
    num_labels: int = 7
    classifier_dropout: float = 0.1
```

### 2. 数据配置 (config/data_config.py)

```python
@dataclass
class DataConfig:
    # 数据集路径
    data_dir: str = "data/CMU_MOSEI"
    aligned_dir: str = "data/CMU_MOSEI/aligned"
    
    # 特征文件
    text_feature_file: str = "CMU_MOSEI_TimestampedWordVectors.csd"
    acoustic_feature_file: str = "CMU_MOSEI_COVAREP.csd"
    visual_feature_file: str = "CMU_MOSEI_VisualOpenFace2.csd"
    label_file: str = "CMU_MOSEI_Labels.csd"
    
    # 数据集划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
```

### 3. 训练配置 (config/train_config.py)

```python
@dataclass
class TrainConfig:
    # 训练参数
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # 优化器参数
    optimizer: str = "AdamW"
    scheduler: str = "linear_warmup"
    
    # 设备设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
```

## 使用指南

### 1. 数据准备

1. 下载CMU-MOSEI数据集：
```bash
python preprocessing/preprocess_mosei.py
```

2. 数据集将被下载到 `data/CMU_MOSEI` 目录，包含：
   - 文本特征 (GloVe词向量)
   - 语音特征 (COVAREP)
   - 视觉特征 (OpenFace)
   - 情感标签

### 2. 运行实验

1. 基线实验：
```bash
python -m experiment.experiments.baseline_experiment
```

2. 自定义实验：
```python
from experiment import ExperimentConfig, ExperimentManager
from experiment.experiments import baseline_experiment

# 创建实验配置
experiment_config = ExperimentConfig(
    experiment_name="my_experiment",
    experiment_id="exp_002",
    description="My custom experiment"
)

# 创建实验管理器
experiment_manager = ExperimentManager(experiment_config)

# 运行实验
results = experiment_manager.run_experiment(baseline_experiment.run_baseline_experiment)
```

### 3. 训练和评估

1. 训练模型：
```bash
python train.py
```

2. 评估模型：
```bash
python evaluate.py
```

### 4. 实验结果

实验结果将保存在 `experiments/[experiment_id]/` 目录下：
- `results/`: 包含实验结果和指标
- `logs/`: 包含实验日志
- `models/`: 包含保存的模型

## 实验管理

### 1. 创建新实验

1. 在 `experiment/experiments/` 目录下创建新的实验文件
2. 实现实验函数
3. 使用 `ExperimentManager` 运行实验

### 2. 超参数搜索

在 `experiment_config.py` 中配置超参数网格：
```python
hyperparameter_grid = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "batch_size": [16, 32, 64]
}
```

### 3. 实验记录

- 所有实验配置自动保存
- 训练指标实时记录
- 模型检查点定期保存
- 实验结果自动保存

## 常见问题

1. **数据集下载问题**
   - 确保网络连接稳定
   - 检查磁盘空间
   - 如果下载失败，可以手动下载并放置到对应目录

2. **内存不足**
   - 减小批次大小
   - 使用数据加载器的 `num_workers` 参数
   - 考虑使用梯度累积

3. **训练不稳定**
   - 调整学习率
   - 使用学习率预热
   - 检查数据预处理

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

[添加许可证信息]

## 参考文献

1. Zadeh, A., et al. (2018). Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. ACL 2018.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
