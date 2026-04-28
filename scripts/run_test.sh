#!/bin/bash

# Ray GPU Cluster Test Runner

set -e

# 激活虚拟环境
source ~/venv/bin/activate

echo "=========================================="
echo "Ray GPU Cluster Test Suite"
echo "=========================================="

# 检查集群状态
echo "检查Ray集群状态..."
ray status

if [ $? -ne 0 ]; then
    echo "❌ Ray集群未启动，请先启动集群"
    exit 1
fi

# 运行单节点测试
echo ""
echo "=========================================="
echo "测试1: 单节点vLLM推理"
echo "=========================================="
python tests/test_single_node.py

# 运行双节点测试
echo ""
echo "=========================================="
echo "测试2: 双节点分布式测试"
echo "=========================================="
python scripts/two_node_test.py

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
