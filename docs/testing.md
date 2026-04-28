# 测试说明

## 测试概述

本项目包含多个测试用例，用于验证Ray集群的各项功能：

1. **单节点测试**: 验证基本vLLM推理功能
2. **双节点测试**: 验证Ray分布式功能
3. **负载均衡测试**: 验证Ray Serve负载均衡
4. **故障恢复测试**: 验证Worker节点故障恢复
5. **自动扩缩容测试**: 验证Ray Serve自动扩缩容

## 测试环境

### 硬件配置

- **Head节点**: 8 CPU, 16GB RAM, 无GPU
- **Worker节点1**: 8 CPU, 16GB RAM, 1 GPU (A10)
- **Worker节点2**: 8 CPU, 16GB RAM, 1 GPU (A10)

### 软件版本

- Ray: 2.x
- vLLM: 0.x
- Python: 3.10
- CUDA: 11.8
- PyTorch: 2.x

## 测试用例

### 测试1: 单节点vLLM推理

**目的**: 验证vLLM基本推理功能

**测试脚本**: `tests/test_single_node.py`

**测试步骤**:
```bash
source ~/venv/bin/activate
python tests/test_single_node.py
```

**预期结果**:
- 模型加载成功
- 推理响应正常
- 显存使用合理

**性能指标**:
- QPS: ~1.5
- 平均延迟: ~0.686s
- 显存使用: ~8GB

### 测试2: Ray负载均衡

**目的**: 验证Ray Serve负载均衡功能

**测试脚本**: `scripts/two_node_test.py` (test_ray_load_balancing)

**测试步骤**:
```bash
source ~/venv/bin/activate
python scripts/two_node_test.py
```

**预期结果**:
- 2个服务实例启动成功
- 请求均匀分配到2个实例
- 每个实例处理5个请求

**性能指标**:
- 总时间: ~6.863s
- 平均时间: 0.686s/请求
- QPS: 1.5 请求/秒

### 测试3: Worker节点故障恢复

**目的**: 验证Worker节点故障恢复机制

**测试脚本**: `scripts/two_node_test.py` (test_worker_node_failure_recovery)

**测试步骤**:
```bash
# 1. 启动测试
source ~/venv/bin/activate
python scripts/two_node_test.py

# 2. 停止Worker节点（在另一个终端）
ssh root@<worker-ip> "source venv/bin/activate && ray stop"

# 3. 重启Worker节点
ssh root@<worker-ip> "cd /root && source venv/bin/activate && ray start --address=<head-ip>:6379 --num-gpus=1"

# 4. 观察测试恢复
```

**预期结果**:
- Worker节点停止后，任务排队
- Worker节点重启后，任务重新调度
- 测试正常完成

**性能指标**:
- 故障检测时间: <10s
- 恢复时间: <30s
- 任务重试成功

### 测试4: Ray自动扩缩容

**目的**: 验证Ray Serve自动扩缩容功能

**测试脚本**: `scripts/two_node_test.py` (test_ray_autoscaling)

**测试步骤**:
```bash
source ~/venv/bin/activate
python scripts/two_node_test.py
```

**预期结果**:
- 低负载时副本数减少
- 高负载时副本数增加
- 负载降低后副本数减少

**性能指标**:
- 低负载时间: ~0.826s
- 高负载时间: ~1.871s
- 降负载时间: ~0.376s

## 运行测试

### 运行所有测试

```bash
# 使用运行脚本
bash scripts/run_test.sh

# 或手动运行
source ~/venv/bin/activate
python scripts/two_node_test.py
```

### 运行单个测试

```bash
# 单节点测试
python tests/test_single_node.py

# 双节点测试
python scripts/two_node_test.py
```

## 测试结果分析

### 性能指标解读

**QPS (Queries Per Second)**:
- 每秒处理的请求数
- 越高越好
- 受GPU性能和模型大小影响

**延迟 (Latency)**:
- P50: 50%请求的延迟
- P95: 95%请求的延迟
- P99: 99%请求的延迟
- 越低越好

**显存使用 (GPU Memory)**:
- 模型加载占用的显存
- KV Cache占用的显存
- 需要控制在GPU显存范围内

### 资源利用率

**CPU利用率**:
- 正常范围: 50-80%
- 过高: 可能需要扩容
- 过低: 资源浪费

**GPU利用率**:
- 正常范围: 70-90%
- 过高: 可能需要优化
- 过低: 资源浪费

**内存利用率**:
- 正常范围: 60-80%
- 过高: 可能OOM
- 过低: 资源浪费

## 故障排查

### 测试失败常见原因

1. **集群未启动**
   - 检查: `ray status`
   - 解决: 启动Ray集群

2. **GPU不可用**
   - 检查: `nvidia-smi`
   - 解决: 安装CUDA或重启GPU

3. **模型下载失败**
   - 检查: 网络连接
   - 解决: 手动下载模型或使用代理

4. **内存不足**
   - 检查: `free -h`
   - 解决: 增加内存或减少任务数

5. **依赖缺失**
   - 检查: `pip list`
   - 解决: 安装缺失的依赖

### 日志分析

**Ray日志**:
```bash
# Raylet日志
tail -f /tmp/ray/session_latest/logs/raylet.out

# Dashboard日志
tail -f /tmp/ray/session_latest/logs/dashboard.log
```

**vLLM日志**:
```bash
# vLLM服务日志
tail -f /tmp/vllm.log
```

**应用日志**:
```bash
# 测试脚本日志
tail -f /tmp/test.log
```

## 性能优化

### 提升吞吐量

1. **使用更大的batch size**
   - 增加并发请求
   - 利用连续批处理

2. **优化模型**
   - 使用量化模型
   - 减少模型层数

3. **增加GPU**
   - 使用张量并行
   - 增加Worker节点

### 降低延迟

1. **减少max_tokens**
   - 限制输出长度
   - 提升响应速度

2. **使用更小的模型**
   - 权衡性能和精度
   - 选择合适的模型大小

3. **优化网络**
   - 减少网络延迟
   - 使用更快的网络

### 降低显存使用

1. **使用PagedAttention**
   - vLLM默认启用
   - 显存利用率提升2-4倍

2. **调整gpu_memory_utilization**
   - 设置为0.6-0.8
   - 避免OOM

3. **使用量化模型**
   - FP16/INT8量化
   - 显存占用减半

## 压力测试

### 高并发测试

```bash
# 使用100个并发请求
python tests/test_high_concurrency.py --num_requests 100 --concurrency 10
```

### 长时间稳定性测试

```bash
# 运行1小时测试
python tests/test_stability.py --duration 3600
```

### 极限性能测试

```bash
# 测试最大吞吐量
python tests/test_max_throughput.py
```

## 测试报告

### 生成测试报告

```bash
# 生成HTML报告
python tests/generate_report.py --format html

# 生成JSON报告
python tests/generate_report.py --format json

# 生成PDF报告
python tests/generate_report.py --format pdf
```

### 测试报告内容

- 测试环境信息
- 测试用例执行结果
- 性能指标统计
- 资源使用情况
- 故障和异常记录
- 优化建议

## 持续集成

### GitHub Actions

配置CI/CD自动运行测试：

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r config/requirements.txt
      - name: Run tests
        run: python tests/test_single_node.py
```

### 本地CI

使用Makefile管理测试：

```makefile
test:
    python tests/test_single_node.py

test-all:
    python scripts/two_node_test.py

clean:
    rm -rf /tmp/ray/
```

## 最佳实践

### 测试前准备

1. 确保集群状态正常
2. 检查GPU可用性
3. 清理之前的测试数据
4. 备份重要配置

### 测试执行

1. 按顺序执行测试
2. 记录测试结果
3. 监控资源使用
4. 及时处理异常

### 测试后清理

1. 停止测试进程
2. 清理临时文件
3. 释放GPU资源
4. 生成测试报告

## 参考资源

- [Ray文档](https://docs.ray.io/)
- [vLLM文档](https://docs.vllm.ai/)
- [ModelScope文档](https://modelscope.cn/docs)
