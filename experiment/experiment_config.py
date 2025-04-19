from dataclasses import dataclass
from typing import Dict, List, Any
import os

@dataclass
class ExperimentConfig:
    # 实验基本信息
    experiment_name: str = "baseline"
    experiment_id: str = "exp_001"
    description: str = "Baseline experiment with default settings"
    
    # 实验设置
    model_variants: List[str] = None  # 模型变体列表
    data_variants: List[str] = None   # 数据变体列表
    hyperparameter_grid: Dict[str, List[Any]] = None  # 超参数网格搜索
    
    # 实验路径
    experiment_dir: str = "experiments"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.model_variants is None:
            self.model_variants = ["baseline"]
        if self.data_variants is None:
            self.data_variants = ["default"]
        if self.hyperparameter_grid is None:
            self.hyperparameter_grid = {
                "learning_rate": [1e-5, 2e-5, 5e-5],
                "batch_size": [16, 32, 64]
            }
        
        # 创建必要的目录
        self.experiment_path = os.path.join(self.experiment_dir, self.experiment_id)
        self.results_path = os.path.join(self.experiment_path, self.results_dir)
        self.logs_path = os.path.join(self.experiment_path, self.logs_dir)
        
        os.makedirs(self.experiment_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True) 