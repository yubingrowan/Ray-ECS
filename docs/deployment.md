# 部署指南

## 前置要求

### 硬件要求

- **Head节点**: 1台服务器
  - CPU: 8核+
  - 内存: 16GB+
  - GPU: 可选
  - 网络: 公网IP + 私网IP

- **Worker节点**: 1-10台服务器
  - CPU: 8核+
  - 内存: 16GB+
  - GPU: NVIDIA A10或类似
  - 网络: 私网IP

### 软件要求

- **操作系统**: Ubuntu 20.04+ 或 CentOS 7+
- **Python**: 3.10+
- **CUDA**: 11.8+
- **SSH**: 允许密钥登录

## 环境配置

### 1. 基础环境安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y python3.10 python3.10-venv git curl wget

# 创建虚拟环境
python3.10 -m venv ~/venv
source ~/venv/bin/activate
```

### 2. CUDA安装

```bash
# 下载CUDA
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装CUDA
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent

# 配置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证CUDA
nvidia-smi
```

### 3. 安装Python依赖

```bash
# 激活虚拟环境
source ~/venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install ray[all] vllm modelscope torch transformers
```

## Ray集群部署

### 1. 配置网络

确保节点间网络互通：

```bash
# 测试网络连通性
ping <worker-private-ip>

# 配置SSH免密登录
ssh-keygen -t rsa
ssh-copy-id root@<worker-ip>
```

### 2. 启动Head节点

```bash
# 在Head节点执行
source ~/venv/bin/activate

# 启动Ray Head
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=0

# 记录Head节点地址
# 格式: <head-private-ip>:6379
```

### 3. 启动Worker节点

```bash
# 在Worker节点执行
source venv/bin/activate

# 启动Ray Worker
ray start --address=<head-ip>:6379 --num-gpus=1

# 验证Worker连接
ray status
```

### 4. 验证集群

```bash
# 在Head节点执行
source ~/venv/bin/activate

# 检查集群状态
ray status

# 访问Dashboard
# http://<head-public-ip>:8265
```

## vLLM服务部署

### 1. 下载模型

```bash
# 使用ModelScope下载模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
```

### 2. 启动vLLM服务

```bash
# 启动vLLM推理服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

## 配置文件

### config.yaml

```yaml
# Ray集群配置
ray:
  head:
    ip: "172.31.84.221"
    port: 6379
    dashboard_port: 8265
  workers:
    - ip: "172.31.84.222"
      num_gpus: 1
    # 添加更多worker节点

# vLLM配置
vllm:
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  trust_remote_code: true
  gpu_memory_utilization: 0.8
  max_model_len: 2048

# 测试配置
test:
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  num_requests: 10
  max_tokens: 50
```

## 部署脚本

### 自动部署脚本

使用项目提供的自动部署脚本：

```bash
# 在Head节点执行
bash scripts/setup_gpu_ecs.sh

# 在Worker节点执行（修改IP配置）
bash scripts/setup_gpu_ecs.sh
```

### 手动部署

如果需要手动部署，参考上述步骤。

## 验证部署

### 1. 集群状态检查

```bash
# 检查Ray集群
ray status

# 检查GPU
nvidia-smi

# 检查进程
ps aux | grep ray
```

### 2. 运行测试

```bash
# 单节点测试
python tests/test_single_node.py

# 双节点测试
bash scripts/run_two_node_test.sh
```

### 3. 性能测试

```bash
# vLLM性能测试
python tests/test_vllm_performance.py
```

## 常见问题

### 1. Worker节点无法连接

**问题**: Worker节点启动后无法连接到Head节点

**解决方案**:
- 检查网络连通性: `ping <head-ip>`
- 检查防火墙规则
- 确认Head节点已启动
- 查看Worker节点日志: `tail -f /tmp/ray/session_latest/logs/raylet.out`

### 2. GPU不可用

**问题**: Ray无法检测到GPU

**解决方案**:
- 检查CUDA安装: `nvidia-smi`
- 检查PyTorch GPU支持: `python -c "import torch; print(torch.cuda.is_available())"`
- 重新安装CUDA和PyTorch
- 重启Ray集群

### 3. 内存不足

**问题**: OOM错误

**解决方案**:
- 减少`gpu_memory_utilization`
- 使用更小的模型
- 增加系统内存
- 使用vLLM的PagedAttention

### 4. Dashboard无法访问

**问题**: 无法访问Ray Dashboard

**解决方案**:
- 检查Dashboard端口: `netstat -tlnp | grep 8265`
- 检查防火墙规则
- 确认Head节点IP正确
- 使用`--dashboard-host=0.0.0.0`启动

## 监控和维护

### 监控指标

- **集群状态**: Ray Dashboard
- **GPU使用**: `nvidia-smi`
- **系统资源**: `htop`
- **日志**: `/tmp/ray/session_latest/logs/`

### 日常维护

```bash
# 清理Ray日志
rm -rf /tmp/ray/session_*

# 重启Ray集群
ray stop
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 清理GPU内存
nvidia-smi --gpu-reset -i 0
```

## 扩展集群

### 添加Worker节点

```bash
# 在新Worker节点执行
source venv/bin/activate
ray start --address=<head-ip>:6379 --num-gpus=1

# 验证新节点
ray status
```

### 移除Worker节点

```bash
# 在Worker节点执行
ray stop

# Head节点会自动检测节点离线
```

## 备份和恢复

### 备份配置

```bash
# 备份配置文件
tar -czf ray-config-backup.tar.gz config/

# 备份日志
tar -czf ray-logs-backup.tar.gz /tmp/ray/
```

### 恢复配置

```bash
# 恢复配置文件
tar -xzf ray-config-backup.tar.gz

# 重新启动集群
ray stop
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
```

## 安全建议

1. **网络安全**
   - 使用私网通信
   - 配置防火墙规则
   - 限制Dashboard访问

2. **认证**
   - 使用SSH密钥认证
   - 禁用密码登录
   - 定期更新密钥

3. **监控**
   - 监控异常访问
   - 记录操作日志
   - 设置告警规则

## 升级指南

### 升级Ray

```bash
# 停止集群
ray stop

# 升级Ray
pip install --upgrade ray[all]

# 重启集群
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
```

### 升级vLLM

```bash
# 升级vLLM
pip install --upgrade vllm

# 重启vLLM服务
# (根据部署方式调整)
```

## 成本优化

### 资源优化

- 使用自动扩缩容
- 选择合适的GPU型号
- 优化模型大小
- 使用Spot实例

### 监控成本

- 定期检查实例使用情况
- 设置成本告警
- 及时释放闲置资源
- 使用成本分析工具
