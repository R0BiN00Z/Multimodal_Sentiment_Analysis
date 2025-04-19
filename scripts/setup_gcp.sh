#!/bin/bash

# 更新系统
sudo apt-get update
sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y git python3-pip python3-venv

# 检测设备类型并安装相应驱动
if lspci | grep -i nvidia > /dev/null; then
    echo "Installing NVIDIA drivers for A100..."
    # 安装 NVIDIA 驱动
    sudo apt-get install -y nvidia-driver-525
    # 安装 CUDA
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    sudo sh cuda_12.1.0_530.30.02_linux.run --silent
elif [ -d "/dev/accel0" ]; then
    echo "Setting up TPU environment..."
    # 安装 TPU 驱动
    pip install cloud-tpu-client
fi

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch 和其他依赖
if [ -d "/dev/accel0" ]; then
    # TPU 配置
    pip install torch torchvision torchaudio
    pip install cloud-tpu-client
else
    # GPU 配置
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements.txt

# 设置环境变量
if lspci | grep -i nvidia > /dev/null; then
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda-12.1/bin:$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
elif [ -d "/dev/accel0" ]; then
    echo "export XRT_TPU_CONFIG='localservice;0;localhost:51011'" >> ~/.bashrc
fi

source ~/.bashrc

# 克隆代码仓库（如果需要）
# git clone <your-repo-url>
# cd <repo-name>

echo "Setup completed! Please activate virtual environment: source venv/bin/activate" 