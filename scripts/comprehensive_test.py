#!/usr/bin/env python3
"""
明天GPU集群综合测试脚本
2小时高强度测试计划：
1. OOM挑战测试（0-30分钟）
2. vLLM PagedAttention测试（30-60分钟）
3. 高并发压测（60-90分钟）
4. 性能优化对比（90-120分钟）
"""

import ray
import torch
import time
import psutil
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer as ModelScopeTokenizer, AutoModelForCausalLM as ModelScopeModel, snapshot_download
import numpy as np
import logging
import sys
from datetime import datetime

# 配置日志
log_file = f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("=" * 60)
print(f"日志文件: {log_file}")
print("=" * 60)

# ==========================================
# 挑战1: 故意制造OOM学习内存管理
# ==========================================
def test_oom_challenge():
    """挑战1: 故意制造OOM，学习内存管理"""
    logger.info("\n" + "=" * 60)
    logger.info("挑战1: OOM内存管理测试")
    logger.info("=" * 60)
    
    try:
        # 加载超大模型制造OOM
        logger.info("加载超大模型测试...")
        model_name = "Qwen/Qwen2.5-14B-Instruct"  # 使用14B模型制造OOM

        # 计算理论显存占用
        logger.info("\n=== 理论显存计算 ===")
        param_count = 14e9  # 14B参数
        fp16_bytes = 2  # FP16每个参数2字节
        model_params_memory = param_count * fp16_bytes / (1024**3)  # GB

        # KV缓存估算
        seq_len = 100  # 假设序列长度
        hidden_dim = 5120  # Qwen2.5-14B隐藏层维度
        layers = 28  # Qwen2.5-14B层数
        kv_cache_memory = seq_len * hidden_dim * layers * 2 * 2 / (1024**3)  # GB (FP16)

        # 激活值和开销
        activation_memory = 2.0  # 激活值约2GB
        overhead_memory = 1.0  # 其他开销约1GB
        total_estimated = model_params_memory + kv_cache_memory + activation_memory + overhead_memory
        logger.info(f"模型参数: {param_count/1e9:.1f}B")
        logger.info(f"数据类型: FP16 (2 bytes/参数)")
        logger.info(f"模型参数显存: {model_params_memory:.2f}GB")
        logger.info(f"KV缓存估算: {kv_cache_memory:.1f}GB")
        logger.info(f"激活值估算: {activation_memory:.1f}GB")
        logger.info(f"其他开销估算: {overhead_memory:.1f}GB")
        logger.info(f"总估算显存: {total_estimated:.2f}GB")
        logger.info(f"A10 GPU显存: 24GB")
        logger.info(f"是否超限: {'是' if total_estimated > 24 else '否'}")
        
        # 尝试加载完整模型（可能OOM）
        try:
            # 使用ModelScope加载模型（国内网络更稳定）
            logger.info("从ModelScope下载14B模型...")
            model_dir = snapshot_download(model_name)  # 下载14B模型
            tokenizer = ModelScopeTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = ModelScopeModel.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,  # 使用FP32占用双倍内存
                device_map="cuda",  # 强制全部加载到GPU
                low_cpu_mem_usage=False,  # 禁用CPU卸载
                trust_remote_code=True
            )
            logger.info("✅ 大模型加载成功")

            # 测试内存使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                memory_peak = torch.cuda.max_memory_allocated() / 1024**3

                logger.info(f"\n=== 实际显存使用 ===")
                logger.info(f"GPU内存分配: {memory_allocated:.2f}GB")
                logger.info(f"GPU内存保留: {memory_reserved:.2f}GB")
                logger.info(f"GPU内存峰值: {memory_peak:.2f}GB")

                # 理论vs实际对比
                logger.info(f"\n=== 显存对比 ===")
                logger.info(f"理论估算: {total_estimated:.2f}GB")
                logger.info(f"实际使用: {memory_allocated:.2f}GB")
                logger.info(f"差异: {abs(memory_allocated - total_estimated):.2f}GB ({abs(memory_allocated - total_estimated)/total_estimated*100:.1f}%)")
            
            # 尝试大批量推理
            logger.info("测试大批量推理...")
            logger.info("显存分析 ===")
            batch_size = 100
            prompts = ["什么是人工智能？"] * batch_size  # 100个并发请求
            
            # 计算100个请求的KV缓存显存
            logger.info(f"批量大小: {batch_size}")
            logger.info(f"单个prompt token数: ~10 tokens")
            logger.info(f"序列长度: ~10 tokens")
            logger.info(f"隐藏层维度: ~4096 (Qwen2.5-7B)")
            print(f"单个prompt token数: ~10 tokens")
            print(f"序列长度: ~10 tokens")
            print(f"隐藏层维度: ~4096 (Qwen2.5-7B)")
            
            # KV缓存计算公式
            seq_len = 10  # 假设输入序列长度
            hidden_dim = 4096  # Qwen2.5-7B隐藏层维度
            layers = 28  # Qwen2.5-7B层数
            
            # 单个请求的KV缓存
            kv_cache_single = seq_len * hidden_dim * layers * 2 * 2  # bytes (FP16)
            kv_cache_single_gb = kv_cache_single / (1024**3)
            
            # 100个请求的KV缓存
            kv_cache_batch = kv_cache_single * batch_size
            kv_cache_batch_gb = kv_cache_batch / (1024**3)
            
            print(f"\n=== KV缓存显存计算 ===")
            print(f"单请求KV缓存: {kv_cache_single_gb:.3f}GB")
            print(f"{batch_size}请求KV缓存: {kv_cache_batch_gb:.2f}GB")
            print(f"模型参数: {model_params_memory:.2f}GB")
            print(f"总显存需求: {model_params_memory + kv_cache_batch_gb:.2f}GB")
            print(f"A10 GPU显存: 24GB")
            print(f"是否超限: {'是' if (model_params_memory + kv_cache_batch_gb) > 24 else '否'}")
            
            print(f"\n=== OOM风险分析 ===")
            print(f"风险因素:")
            print(f"1. KV缓存随batch_size线性增长")
            print(f"2. 每层都有K和V两个缓存")
            print(f"3. 生成新token时KV缓存持续增长")
            print(f"4. 激活值占用额外显存")
            
            print(f"\n=== 开始批量推理测试 ===")
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 记录推理前显存
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            
            # 记录推理后显存
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                memory_peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"推理前显存: {memory_before:.2f}GB")
                print(f"推理后显存: {memory_after:.2f}GB")
                print(f"峰值显存: {memory_peak:.2f}GB")
                print(f"KV缓存增长: {memory_peak - memory_before:.2f}GB")
            
            print("✅ 大批量推理成功")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM错误: {e}")
                logger.info("💡 内存管理技巧:")
                logger.info("1. 使用量化模型 (4-bit/8-bit)")
                logger.info("2. 减少批量大小")
                logger.info("3. 使用梯度检查点")
                logger.info("4. 清理GPU缓存")
            else:
                raise
                
    except Exception as e:
        logger.info(f"测试失败: {e}")

    # 清理GPU内存
    logger.info("清理GPU内存...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("GPU内存已清理")

    logger.info("\n挑战1完成")
    print("\n挑战1完成")

# ==========================================
# vLLM PagedAttention测试
# ==========================================
def test_vllm_pagedattention():
    """测试vLLM PagedAttention性能"""
    logger.info("\n" + "=" * 60)
    logger.info("挑战2: vLLM PagedAttention测试")
    logger.info("=" * 60)
    
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 测试提示
        prompts = [
            "什么是人工智能？",
            "机器学习的原理是什么？",
            "深度学习有什么应用？"
        ]
        
        # ====================
        # 1. 传统推理测试
        # ====================
        logger.info("\n=== 传统推理测试 ===")
        logger.info("从ModelScope下载1.5B模型...")
        model_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer = ModelScopeTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        traditional_model = ModelScopeModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 记录传统推理内存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            traditional_memory_before = torch.cuda.memory_allocated() / 1024**3
        
        # 传统推理
        start_time = time.time()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(traditional_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = traditional_model.generate(**inputs, max_new_tokens=50)
        traditional_time = time.time() - start_time
        
        if torch.cuda.is_available():
            traditional_memory_after = torch.cuda.memory_allocated() / 1024**3
            traditional_memory_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"传统推理: 时间={traditional_time:.3f}s, 内存={traditional_memory_peak:.2f}GB")
        
        del traditional_model
        torch.cuda.empty_cache()
        
        # ====================
        # 2. vLLM推理测试
        # ====================
        logger.info("\n=== vLLM PagedAttention测试 ===")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vllm_memory_before = torch.cuda.memory_allocated() / 1024**3
        
        # vLLM使用本地模型路径
        llm = LLM(
            model=model_dir,  # 使用ModelScope下载的本地路径
            trust_remote_code=True,
            max_model_len=2048,
            enable_prefix_caching=True  # 启用prompt缓存
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50
        )

        logger.info(f"开始推理 {len(prompts)} 个提示...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        vllm_time = time.time() - start_time

        if torch.cuda.is_available():
            vllm_memory_after = torch.cuda.memory_allocated() / 1024**3
            vllm_memory_peak = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(f"\n=== vLLM推理结果 ===\n")
        for i, output in enumerate(outputs):
            logger.info(f"\n提示 {i+1}: {output.prompt}")
            logger.info(f"回答: {output.outputs[0].text}")

        logger.info(f"\n=== 性能统计 ===\n")
        logger.info(f"vLLM总时间: {vllm_time:.3f}s")
        logger.info(f"vLLM平均时间: {vllm_time/len(prompts):.3f}s/请求")
        logger.info(f"vLLM吞吐量: {len(prompts)/vllm_time:.1f} 请求/秒")
        logger.info(f"vLLM内存峰值: {vllm_memory_peak:.2f}GB")
        
        # ====================
        # 3. 性能对比
        # ====================
        logger.info(f"\n=== 性能对比 ===")
        logger.info(f"时间对比: vLLM {vllm_time:.3f}s vs 传统 {traditional_time:.3f}s")
        logger.info(f"加速比: {traditional_time/vllm_time:.2f}x")
        logger.info(f"内存对比: vLLM {vllm_memory_peak:.2f}GB vs 传统 {traditional_memory_peak:.2f}GB")
        logger.info(f"内存节省: {(1 - vllm_memory_peak/traditional_memory_peak)*100:.1f}%")

        # ====================
        # 4. Prompt缓存测试
        # ====================
        logger.info(f"\n=== Prompt缓存测试 ===")
        similar_prompts = [
            "什么是人工智能？",  # 相似
            "人工智能的定义是什么？",  # 相似
            "AI是什么意思？"  # 相似
        ]

        # 第一次推理（无缓存）
        start_time = time.time()
        llm.generate(similar_prompts, sampling_params)
        first_time = time.time() - start_time
        logger.info(f"第一次推理（无缓存）: {first_time:.3f}s")

        # 第二次推理（有缓存）
        start_time = time.time()
        llm.generate(similar_prompts, sampling_params)
        cached_time = time.time() - start_time
        logger.info(f"第二次推理（有缓存）: {cached_time:.3f}s")
        logger.info(f"缓存加速: {first_time/cached_time:.2f}x")
        
        # ====================
        # 5. Continuous Batching测试（大prompt优化）
        # ====================
        logger.info(f"\n=== Continuous Batching测试 ===")

        # 生成大prompt
        long_prompt = """
        人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

        人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考，也可能超过人的智能。

        请根据以上内容，回答以下问题："""

        # 测试场景：混合长度prompt
        mixed_prompts = [
            long_prompt + " 什么是人工智能？",  # 大prompt
            "机器学习的原理是什么？",  # 小prompt
            long_prompt + " 深度学习有什么应用？",  # 大prompt
            "自然语言处理技术有哪些？"  # 小prompt
        ]

        logger.info(f"测试 {len(mixed_prompts)} 个混合长度prompt")
        logger.info(f"大prompt长度: {len(long_prompt)} 字符")
        logger.info(f"小prompt平均长度: {len('机器学习的原理是什么？')} 字符")

        # vLLM连续批处理
        start_time = time.time()
        outputs = llm.generate(mixed_prompts, sampling_params)
        vllm_continuous_time = time.time() - start_time

        logger.info(f"\n=== vLLM Continuous Batching结果 ===")
        logger.info(f"总时间: {vllm_continuous_time:.3f}s")
        logger.info(f"平均时间: {vllm_continuous_time/len(mixed_prompts):.3f}s/请求")

        # 传统批处理对比（需要等待所有prompt处理完成）
        logger.info(f"\n=== 传统静态批处理对比 ===")
        start_time = time.time()
        for i, prompt in enumerate(mixed_prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(traditional_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = traditional_model.generate(**inputs, max_new_tokens=50)
        traditional_batch_time = time.time() - start_time

        logger.info(f"传统批处理时间: {traditional_batch_time:.3f}s")
        logger.info(f"Continuous Batching加速: {traditional_batch_time/vllm_continuous_time:.2f}x")
        logger.info(f"说明: vLLM可以动态添加请求到batch，无需等待所有prompt处理完成")

        logger.info("\n✅ vLLM PagedAttention测试完成")
        
    except ImportError:
        print("⚠️ vLLM未安装，跳过此测试")
        print("安装命令: pip install vllm")
    except Exception as e:
        print(f"vLLM测试失败: {e}")

# ==========================================
# 高并发压测
# ==========================================
def test_high_concurrency():
    """高并发压测"""
    print("\n" + "=" * 60)
    print("挑战3: 高并发压测")
    print("=" * 60)
    
    try:
        # 使用Ray进行并发测试
        ray.init(ignore_reinit_error=True)
        
        @ray.remote(num_gpus=1)
        class ConcurrentService:
            def __init__(self):
                from modelscope import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            def inference(self, prompt):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 创建服务
        num_services = 1  # 单GPU
        services = [ConcurrentService.remote() for _ in range(num_services)]
        
        # 并发测试
        concurrency_levels = [10, 50, 100]
        for concurrency in concurrency_levels:
            print(f"\n测试并发数: {concurrency}")
            prompts = ["什么是AI？"] * concurrency
            
            start_time = time.time()
            results = ray.get([services[i % num_services].inference.remote(p) 
                             for i, p in enumerate(prompts)])
            total_time = time.time() - start_time
            
            print(f"总时间: {total_time:.3f}秒")
            print(f"QPS: {concurrency/total_time:.1f} 请求/秒")
            print(f"平均延迟: {total_time/concurrency*1000:.1f}ms")
        
        ray.shutdown()
        print("\n✅ 高并发压测完成")
        
    except Exception as e:
        print(f"并发测试失败: {e}")
        if ray.is_initialized():
            ray.shutdown()

# ==========================================
# 性能优化对比
# ==========================================
def test_performance_optimization():
    """性能优化对比测试"""
    print("\n" + "=" * 60)
    print("挑战4: 性能优化对比")
    print("=" * 60)
    
    try:
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        
        # 测试模型
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # 基准测试：FP32
        print("\n=== FP32基准测试 ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        prompt = "什么是人工智能？"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_fp32.device) for k, v in inputs.items()}
        
        # 预热
        with torch.no_grad():
            _ = model_fp32.generate(**inputs, max_new_tokens=5)
        
        # 正式测试
        start_time = time.time()
        with torch.no_grad():
            outputs = model_fp32.generate(**inputs, max_new_tokens=20)
        inference_time = time.time() - start_time
        
        memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        print(f"FP32: 时间={inference_time:.3f}s, 内存={memory:.2f}GB")
        
        del model_fp32
        torch.cuda.empty_cache()
        
        # 测试1: FP16 vs FP32
        print("\n=== 精度对比 ===")
        for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            model = ModelScopeModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            
            prompt = "什么是人工智能？"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 预热
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=5)
            
            # 正式测试
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
            inference_time = time.time() - start_time
            
            memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            print(f"{dtype_name}: 时间={inference_time:.3f}s, 内存={memory:.2f}GB")
            
            del model
            torch.cuda.empty_cache()
        
        # 测试2: 批量大小对比
        print("\n=== 批量大小对比 ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        for batch_size in [1, 4, 8]:
            prompts = ["什么是AI？"] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            inference_time = time.time() - start_time
            
            print(f"批量{batch_size}: 总时间={inference_time:.3f}s, 平均={inference_time/batch_size:.3f}s/请求")
        
        print("\n✅ 性能优化对比完成")
        
    except Exception as e:
        print(f"性能测试失败: {e}")

# ==========================================
# 主测试流程
# ==========================================
def main():
    """主测试流程"""
    print("\n🚀 开始明天综合测试")
    print("⏰ 预计总时间: 2小时")
    
    start_time = time.time()
    
    # 挑战1: OOM测试
    print("\n⏱️  0-30分钟: OOM挑战测试")
    test_oom_challenge()
    
    # 挑战2: vLLM测试
    print("\n⏱️  30-60分钟: vLLM PagedAttention测试")
    test_vllm_pagedattention()
    
    # 挑战3: 高并发测试
    print("\n⏱️  60-90分钟: 高并发压测")
    test_high_concurrency()
    
    # 挑战4: 性能优化
    print("\n⏱️  90-120分钟: 性能优化对比")
    test_performance_optimization()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 所有测试完成！总时间: {total_time/60:.1f}分钟")
    print("=" * 60)

if __name__ == "__main__":
    main()
