#!/usr/bin/env python3
"""
单节点vLLM推理测试
验证vLLM基本推理功能和性能
"""

import time
from vllm import LLM, SamplingParams
from modelscope import snapshot_download

def test_vllm_inference():
    """测试vLLM单节点推理"""
    print("=" * 60)
    print("单节点vLLM推理测试")
    print("=" * 60)
    
    try:
        # 下载模型
        print("下载模型...")
        model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
        
        # 初始化vLLM
        print("初始化vLLM...")
        llm = LLM(
            model=model_dir,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048
        )
        
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50
        )
        
        # 测试提示
        prompts = [
            "什么是人工智能？",
            "机器学习的原理是什么？",
            "深度学习有什么应用？"
        ]
        
        # 执行推理
        print(f"\n开始推理 {len(prompts)} 个提示...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        # 打印结果
        print(f"\n=== 推理结果 ===")
        for i, output in enumerate(outputs):
            print(f"\n提示 {i+1}: {output.prompt}")
            print(f"回答: {output.outputs[0].text}")
        
        # 性能统计
        print(f"\n=== 性能统计 ===")
        print(f"总时间: {total_time:.3f}s")
        print(f"平均时间: {total_time/len(prompts):.3f}s/请求")
        print(f"吞吐量: {len(prompts)/total_time:.1f} 请求/秒")
        
        print("\n✅ 单节点vLLM推理测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_inference()
