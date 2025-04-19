from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Text encoder
    text_model_name: str = "bert-base-uncased"
    text_hidden_dim: int = 768
    
    # Acoustic encoder
    acoustic_input_dim: int = 74  # COVAREP特征维度
    acoustic_hidden_dim: int = 256
    
    # Visual encoder
    visual_input_dim: int = 35   # OpenFace特征维度
    visual_hidden_dim: int = 256
    
    # Fusion
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.1
    
    # Classifier
    num_labels: int = 7  # 情感类别数量
    classifier_dropout: float = 0.1 