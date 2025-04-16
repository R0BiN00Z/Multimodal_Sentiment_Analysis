import os
import json
from typing import Dict, Any
from .model_config import ModelConfig
from .data_config import DataConfig
from .train_config import TrainConfig

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.train_config = TrainConfig()
        
    def save_configs(self, output_dir: str):
        """保存所有配置到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        configs = {
            "model": self.model_config.__dict__,
            "data": self.data_config.__dict__,
            "train": self.train_config.__dict__
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(configs, f, indent=4)
    
    def load_configs(self, config_file: str):
        """从文件加载配置"""
        with open(config_file, "r") as f:
            configs = json.load(f)
            
        self.model_config = ModelConfig(**configs["model"])
        self.data_config = DataConfig(**configs["data"])
        self.train_config = TrainConfig(**configs["train"])
    
    def update_config(self, config_type: str, **kwargs):
        """更新特定类型的配置"""
        if config_type == "model":
            for key, value in kwargs.items():
                setattr(self.model_config, key, value)
        elif config_type == "data":
            for key, value in kwargs.items():
                setattr(self.data_config, key, value)
        elif config_type == "train":
            for key, value in kwargs.items():
                setattr(self.train_config, key, value)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def get_config(self, config_type: str) -> Any:
        """获取特定类型的配置"""
        if config_type == "model":
            return self.model_config
        elif config_type == "data":
            return self.data_config
        elif config_type == "train":
            return self.train_config
        else:
            raise ValueError(f"Unknown config type: {config_type}") 