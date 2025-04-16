from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model architecture
    text_model_name: str = "bert-base-uncased"
    hidden_dim: int = 768
    dropout: float = 0.1
    num_labels: int = 7  # 情感类别数量
    
    # Input dimensions
    acoustic_input_dim: int = 74  # COVAREP特征维度
    visual_input_dim: int = 35   # OpenFace特征维度
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 100
    
    # Data paths
    data_dir: str = "data/CMU_MOSEI"
    aligned_dir: str = "data/CMU_MOSEI/aligned"
    output_dir: str = "outputs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu" 