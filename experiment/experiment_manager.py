import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import torch
from ..config import ConfigManager
from .experiment_config import ExperimentConfig

class ExperimentManager:
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.config_manager = ConfigManager()
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(
            self.config.logs_path,
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.config.experiment_name)
    
    def save_experiment_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.config.experiment_path, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=4)
    
    def save_results(self, results: Dict[str, Any], variant: str = "default"):
        """保存实验结果"""
        results_path = os.path.join(
            self.config.results_path,
            f"results_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    
    def save_metrics(self, metrics: Dict[str, List[float]], variant: str = "default"):
        """保存训练指标"""
        metrics_path = os.path.join(
            self.config.results_path,
            f"metrics_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    
    def save_model(self, model: torch.nn.Module, variant: str = "default"):
        """保存模型"""
        model_path = os.path.join(
            self.config.experiment_path,
            f"model_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save(model.state_dict(), model_path)
    
    def log_metric(self, name: str, value: float, step: int = None):
        """记录指标"""
        if step is not None:
            self.logger.info(f"{name} at step {step}: {value}")
        else:
            self.logger.info(f"{name}: {value}")
    
    def log_config(self, config_type: str, config: Dict[str, Any]):
        """记录配置"""
        self.logger.info(f"{config_type} configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def run_experiment(self, experiment_func):
        """运行实验"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Experiment ID: {self.config.experiment_id}")
        self.logger.info(f"Description: {self.config.description}")
        
        # 保存实验配置
        self.save_experiment_config()
        
        # 运行实验
        try:
            results = experiment_func(self)
            self.logger.info("Experiment completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise 