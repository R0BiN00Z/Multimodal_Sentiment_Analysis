#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
if lspci | grep -i nvidia > /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    # 启用混合精度训练
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
elif [ -d "/dev/accel0" ]; then
    export XRT_TPU_CONFIG='localservice;0;localhost:51011'
fi

# 开始训练
echo "Starting training..."
if lspci | grep -i nvidia > /dev/null; then
    # GPU 训练，使用更大的批处理大小
    python train_decision_fusion.py --batch-size 256 --mixed-precision
else
    # TPU 训练
    python train_decision_fusion.py --batch-size 128
fi

# 训练完成后关闭实例（可选）
# sudo shutdown -h now 