# Ray GPU Cluster - 分布式LLM推理集群

基于Ray的分布式GPU集群，用于LLM推理服务的负载均衡、故障恢复和自动扩缩容。

## 项目简介

本项目展示了一个完整的Ray分布式GPU集群架构，包含：
- **Ray负载均衡**: 多节点GPU资源调度
- **Worker节点故障恢复**: 分布式系统容错机制
- **Ray自动扩缩容**: 动态副本调整优化资源使用
- **vLLM推理优化**: PagedAttention和连续批处理

## 技术栈

- **Ray**: 分布式计算框架
- **vLLM**: 高性能LLM推理引擎
- **ModelScope**: 模型下载和管理
- **CUDA**: GPU计算加速
- **Python**: 3.10+

## 功能特性

### 1. Ray负载均衡
- 自动将LLM服务实例调度到不同节点
- 请求级负载均衡
- GPU资源感知调度

### 2. Worker节点故障恢复
- Worker节点故障自动检测
- 任务自动重新调度
- 集群状态监控

### 3. Ray自动扩缩容
- 基于负载动态调整副本数
- 低负载时减少副本，节省资源
- 高负载时自动扩容，提升性能

### 4. vLLM推理优化
- PagedAttention显存优化
- 连续批处理提升吞吐
- GPU内存管理

## 快速开始

### 环境要求

- 2台GPU服务器（NVIDIA A10或类似）
- Python 3.10+
- CUDA 11.8+
- SSH访问权限

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/ray-gpu-cluster.git
cd ray-gpu-cluster
```

2. **安装依赖**
```bash
pip install -r config/requirements.txt
```

3. **配置环境**
```bash
# 编辑 config/config.yaml 设置节点IP地址
vim config/config.yaml
```

4. **部署Ray集群**
```bash
# 在head节点执行
bash scripts/setup_gpu_ecs.sh

# 在worker节点执行（修改IP地址）
bash scripts/setup_gpu_ecs.sh
```

5. **运行测试**
```bash
# 单节点测试
python tests/test_single_node.py

# 双节点测试
bash scripts/run_two_node_test.sh
```

## 项目结构

```
ray-gpu-cluster/
├── README.md              # 项目说明
├── docs/
│   ├── architecture.md     # 架构说明
│   ├── deployment.md      # 部署指南
│   └── testing.md         # 测试说明
├── scripts/
│   ├── setup_gpu_ecs.sh   # 环境安装
│   ├── two_node_test.py   # 双节点测试
│   └── run_test.sh        # 运行脚本
├── config/
│   ├── config.yaml        # 配置文件
│   └── requirements.txt   # 依赖列表
└── tests/
    ├── test_single_node.py
    └── test_distributed.py
```

## 测试结果

### 单节点测试
- vLLM推理性能: 1.5 QPS
- 显存使用: 8GB
- 平均延迟: 0.686s

### 双节点测试
- Ray负载均衡: 2个服务实例
- 故障恢复时间: <30s
- 自动扩缩容: 动态副本调整

## 架构图

```
┌─────────────────────────────────────────────────┐
│              Ray Head Node                      │
│  - Ray Dashboard (8265)                         │
│  - GCS Server (6379)                            │
│  - Resource Scheduler                          │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼───────┐ ┌────▼────────┐
│ Worker Node 1 │ │ Worker Node 2│
│ - GPU (A10)   │ │ - GPU (A10)  │
│ - vLLM Service│ │ - vLLM Service│
└──────────────┘ └─────────────┘
```

## 部署指南

详细的部署指南请参考 [docs/deployment.md](docs/deployment.md)

## 测试说明

详细的测试说明请参考 [docs/testing.md](docs/testing.md)

## 性能优化

- 使用vLLM的PagedAttention优化显存
- Ray Serve自动扩缩容节省资源
- 连续批处理提升推理吞吐

## 故障排查

常见问题和解决方案请参考 [docs/troubleshooting.md](docs/troubleshooting.md)

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

- 作者: Your Name
- Email: your.email@example.com
- GitHub: https://github.com/yourusername

## 致谢

感谢Ray、vLLM和ModelScope开源项目的支持！
