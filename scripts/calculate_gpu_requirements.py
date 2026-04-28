#!/usr/bin/env python3
"""
GPU需求计算脚本
计算不同模型规模、QPS、上下文长度下的GPU需求
"""

def calculate_kv_cache(hidden_dim, num_layers, seq_len, dtype_bytes=2):
    """
    计算KV Cache大小
    
    Args:
        hidden_dim: 隐藏层维度
        num_layers: 层数
        seq_len: 序列长度
        dtype_bytes: 数据类型字节数 (FP16=2, FP32=4, INT8=1)
    
    Returns:
        KV Cache大小 (GB)
    """
    kv_per_token = 2 * hidden_dim * num_layers * dtype_bytes  # bytes
    kv_total = kv_per_token * seq_len  # bytes
    return kv_total / (1024**3)  # GB

def calculate_gpu_requirements(
    model_size_b,  # 模型大小（B参数）
    qps,  # QPS
    context_length,  # 上下文长度
    avg_context_length=None,  # 平均上下文长度
    long_request_ratio=0.2,  # 长请求比例
    kv_cache_reuse=0.5,  # KV Cache复用率
    paged_attention_efficiency=3,  # PagedAttention效率
    quantization_efficiency=2,  # 量化效率
    gqa_compression=1,  # GQA压缩比 (1=不使用, 8=压缩到1/8)
    gpu_memory_gb=80,  # 单GPU显存
    gpus_per_server=8  # 每台服务器GPU数
):
    """
    计算GPU需求
    
    Args:
        model_size_b: 模型大小（B参数）
        qps: QPS
        context_length: 最大上下文长度
        avg_context_length: 平均上下文长度
        long_request_ratio: 长请求比例
        kv_cache_reuse: KV Cache复用率
        paged_attention_efficiency: PagedAttention效率
        quantization_efficiency: 量化效率
        gpu_memory_gb: 单GPU显存
        gpus_per_server: 每台服务器GPU数
    
    Returns:
        GPU需求信息
    """
    # 模型参数估算
    if model_size_b == 7:
        hidden_dim = 4096
        num_layers = 28
    elif model_size_b == 13:
        hidden_dim = 5120
        num_layers = 36
    elif model_size_b == 70:
        hidden_dim = 8192
        num_layers = 80
    else:
        hidden_dim = 4096
        num_layers = 28
    
    # 模型本身显存占用 (FP16)
    model_memory_gb = model_size_b * 2  # 1B参数 ≈ 2GB (FP16)
    
    # KV Cache计算
    if avg_context_length is None:
        avg_context_length = context_length // 32  # 默认平均上下文长度
    
    # 长请求KV Cache
    long_kv = calculate_kv_cache(hidden_dim, num_layers, context_length)
    
    # 短请求KV Cache
    short_kv = calculate_kv_cache(hidden_dim, num_layers, avg_context_length)
    
    # 总KV Cache (考虑请求分布和复用)
    long_requests = int(qps * long_request_ratio)
    short_requests = qps - long_requests
    
    total_kv = (long_requests * long_kv + short_requests * short_kv) * kv_cache_reuse
    
    # 优化后KV Cache
    optimized_kv = total_kv / (paged_attention_efficiency * quantization_efficiency * gqa_compression)
    
    # 总显存需求
    total_memory = optimized_kv + model_memory_gb
    
    # GPU数量
    num_gpus = int(total_memory / gpu_memory_gb) + 1
    
    # 服务器数量
    num_servers = int(num_gpus / gpus_per_server) + 1
    
    return {
        "model_size_b": model_size_b,
        "qps": qps,
        "context_length": context_length,
        "avg_context_length": avg_context_length,
        "model_memory_gb": model_memory_gb,
        "long_kv_gb": long_kv,
        "short_kv_gb": short_kv,
        "total_kv_gb": total_kv,
        "optimized_kv_gb": optimized_kv,
        "total_memory_gb": total_memory,
        "num_gpus": num_gpus,
        "servers": num_servers,
        "gpu_memory_gb": gpu_memory_gb,
        "gpus_per_server": gpus_per_server
    }

def print_requirements(req):
    """打印GPU需求信息"""
    print("=" * 60)
    print(f"GPU需求计算 - {req['model_size_b']}B模型")
    print("=" * 60)
    print(f"QPS: {req['qps']}")
    print(f"上下文长度: {req['context_length']}")
    print(f"平均上下文长度: {req['avg_context_length']}")
    print()
    print("=== 显存需求 ===")
    print(f"模型显存: {req['model_memory_gb']:.2f} GB")
    print(f"长请求KV Cache: {req['long_kv_gb']:.2f} GB")
    print(f"短请求KV Cache: {req['short_kv_gb']:.2f} GB")
    print(f"总KV Cache: {req['total_kv_gb']:.2f} GB")
    print(f"优化后KV Cache: {req['optimized_kv_gb']:.2f} GB")
    print(f"总显存需求: {req['total_memory_gb']:.2f} GB")
    print()
    print("=== 硬件需求 ===")
    print(f"GPU数量: {req['num_gpus']}")
    print(f"服务器数量: {req['servers']}")
    print(f"每台服务器GPU: {req['gpus_per_server']}")
    print(f"单GPU显存: {req['gpu_memory_gb']} GB")
    print()

if __name__ == "__main__":
    # 70B模型，2000 QPS，128K上下文 - 不使用GQA
    print("=== 不使用GQA ===")
    req_no_gqa = calculate_gpu_requirements(
        model_size_b=70,
        qps=2000,
        context_length=128000,
        avg_context_length=4000,
        long_request_ratio=0.2,
        kv_cache_reuse=0.5,
        paged_attention_efficiency=3,
        quantization_efficiency=2,
        gqa_compression=1,  # 不使用GQA
        gpu_memory_gb=80,
        gpus_per_server=8
    )
    
    print_requirements(req_no_gqa)
    
    # 70B模型，2000 QPS，128K上下文 - 使用GQA
    print("\n=== 使用GQA (压缩到1/8) ===")
    req_with_gqa = calculate_gpu_requirements(
        model_size_b=70,
        qps=2000,
        context_length=128000,
        avg_context_length=4000,
        long_request_ratio=0.2,
        kv_cache_reuse=0.5,
        paged_attention_efficiency=3,
        quantization_efficiency=2,
        gqa_compression=8,  # 使用GQA，压缩到1/8
        gpu_memory_gb=80,
        gpus_per_server=8
    )
    
    print_requirements(req_with_gqa)
    
    # 对比
    print("\n=== 对比 ===")
    print(f"GPU数量减少: {req_no_gqa['num_gpus']} → {req_with_gqa['num_gpus']} (节省 {(req_no_gqa['num_gpus'] - req_with_gqa['num_gpus']) / req_no_gqa['num_gpus'] * 100:.1f}%)")
    print(f"服务器数量减少: {req_no_gqa['servers']} → {req_with_gqa['servers']} (节省 {(req_no_gqa['servers'] - req_with_gqa['servers']) / req_no_gqa['servers'] * 100:.1f}%)")
    print(f"显存需求减少: {req_no_gqa['total_memory_gb']:.2f} GB → {req_with_gqa['total_memory_gb']:.2f} GB (节省 {(req_no_gqa['total_memory_gb'] - req_with_gqa['total_memory_gb']) / req_no_gqa['total_memory_gb'] * 100:.1f}%)")
