#!/bin/bash
# GPU ECS环境安装脚本
# 适用于阿里云 gn7i-c8g1.2xlarge (NVIDIA A10 GPU)

set -e

echo "=========================================="
echo "GPU ECS环境安装"
echo "=========================================="

# 1. 检查GPU
echo "步骤1: 检查GPU状态"
nvidia-smi

# 2. 安装基础依赖
echo "步骤2: 安装基础依赖"
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian系统
    apt-get update -y
    apt-get install -y build-essential wget git python3 python3-pip python3-venv
elif command -v yum &> /dev/null; then
    # CentOS/RHEL系统
    yum update -y
    yum install -y gcc gcc-c++ make wget git python3 python3-pip python3-devel
else
    echo "不支持的包管理器"
    exit 1
fi

# 3. 安装CUDA (如果未安装)
echo "步骤3: 检查CUDA安装"
if ! command -v nvcc &> /dev/null; then
    echo "CUDA未安装，开始安装CUDA 12.1..."
    
    # 下载CUDA
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent --override
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    echo "CUDA安装完成"
else
    echo "CUDA已安装: $(nvcc --version)"
fi

# 4. 安装Python虚拟环境
echo "步骤4: 设置Python环境"
python3 -m pip install --upgrade pip
pip3 install virtualenv

# 创建虚拟环境
if [ ! -d "venv" ]; then
    virtualenv venv
fi
source venv/bin/activate

# 5. 安装PyTorch (使用pip默认源)
echo "步骤5: 安装PyTorch"
pip install torch torchvision torchaudio

# 验证PyTorch GPU支持
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 6. 安装transformers和其他依赖
echo "步骤6: 安装transformers和依赖"
pip install transformers accelerate datasets sentence-transformers huggingface-hub modelscope

# 7. 安装Ray
echo "步骤7: 安装Ray"
pip install ray[default] ray[serve]

# 8. 安装vLLM
echo "步骤8: 安装vLLM"
pip install vllm

# 9. 安装其他工具
echo "步骤9: 安装其他工具"
pip install psutil numpy matplotlib tensorboard

# 10. 验证安装
echo "步骤10: 验证安装"
echo "=== Python包版本 ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import ray; print(f'Ray: {ray.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM版本检查跳过"

echo ""
echo "=========================================="
echo "✅ GPU ECS环境安装完成！"
echo "=========================================="
echo "激活虚拟环境: source venv/bin/activate"
echo "测试GPU: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "=========================================="
