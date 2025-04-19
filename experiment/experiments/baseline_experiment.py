from typing import Dict, Any
import torch
from ..experiment_manager import ExperimentManager
from ..experiment_config import ExperimentConfig
from ...models import MultimodalModel
from ...preprocessing.preprocess_mosei import MOSEIPreprocessor

def run_baseline_experiment(experiment_manager: ExperimentManager) -> Dict[str, Any]:
    """运行基线实验"""
    # 获取配置
    model_config = experiment_manager.config_manager.get_config("model")
    data_config = experiment_manager.config_manager.get_config("data")
    train_config = experiment_manager.config_manager.get_config("train")
    
    # 记录配置
    experiment_manager.log_config("Model", model_config.__dict__)
    experiment_manager.log_config("Data", data_config.__dict__)
    experiment_manager.log_config("Train", train_config.__dict__)
    
    # 初始化模型和数据
    model = MultimodalModel(model_config)
    model.to(train_config.device)
    
    preprocessor = MOSEIPreprocessor(data_config.data_dir)
    train_dataset = preprocessor.get_dataset(split='train')
    val_dataset = preprocessor.get_dataset(split='val')
    
    # 训练循环
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate
    )
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(train_config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_dataset:
            # 训练步骤
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataset)
        metrics['train_loss'].append(avg_train_loss)
        experiment_manager.log_metric('train_loss', avg_train_loss, epoch)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataset:
                loss, acc = model.validation_step(batch)
                val_loss += loss
                correct += acc
                total += len(batch['label'])
        
        avg_val_loss = val_loss / len(val_dataset)
        val_accuracy = correct / total
        
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        experiment_manager.log_metric('val_loss', avg_val_loss, epoch)
        experiment_manager.log_metric('val_accuracy', val_accuracy, epoch)
        
        # 保存模型和指标
        experiment_manager.save_model(model, f"epoch_{epoch}")
        experiment_manager.save_metrics(metrics)
    
    # 返回最终结果
    return {
        'final_train_loss': metrics['train_loss'][-1],
        'final_val_loss': metrics['val_loss'][-1],
        'final_val_accuracy': metrics['val_accuracy'][-1],
        'best_val_accuracy': max(metrics['val_accuracy'])
    }

if __name__ == "__main__":
    # 创建实验配置
    experiment_config = ExperimentConfig(
        experiment_name="baseline",
        experiment_id="exp_001",
        description="Baseline experiment with default settings"
    )
    
    # 创建实验管理器
    experiment_manager = ExperimentManager(experiment_config)
    
    # 运行实验
    results = experiment_manager.run_experiment(run_baseline_experiment)
    
    # 保存最终结果
    experiment_manager.save_results(results) 