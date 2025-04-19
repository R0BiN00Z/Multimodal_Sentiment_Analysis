from dataclasses import dataclass
import torch

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
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 输出设置
    output_dir: str = "outputs"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # 早停设置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01 