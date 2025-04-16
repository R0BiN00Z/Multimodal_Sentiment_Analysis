from dataclasses import dataclass
import os

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
    
    # 数据加载
    batch_size: int = 32
    num_workers: int = 4
    
    def get_feature_paths(self):
        """获取特征文件的完整路径"""
        return {
            'text': os.path.join(self.data_dir, self.text_feature_file),
            'acoustic': os.path.join(self.data_dir, self.acoustic_feature_file),
            'visual': os.path.join(self.data_dir, self.visual_feature_file),
            'labels': os.path.join(self.data_dir, self.label_file)
        } 