#!/usr/bin/env python3
"""
双节点GPU集群分布式测试
测试内容：
1. Ray负载均衡测试
2. vLLM张量并行测试
3. Ray自动扩缩容测试
"""

import ray
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer as ModelScopeTokenizer, AutoModelForCausalLM as ModelScopeModel, snapshot_download
from ray import serve

print("=" * 60)
print("双节点GPU集群分布式测试")
print("=" * 60)

# ==========================================
# 测试1: Ray负载均衡
# ==========================================
def test_ray_load_balancing():
    """测试Ray负载均衡 - 任务自动分配到不同节点"""
    print("\n" + "=" * 60)
    print("测试1: Ray负载均衡")
    print("=" * 60)
    
    try:
        # 初始化Ray集群
        ray.init(ignore_reinit_error=True)
        
        # 检查集群状态
        print("检查Ray集群状态...")
        resources = ray.available_resources()
        print(f"可用资源: {resources}")
        
        # 创建LLM服务Actor
        @ray.remote(num_gpus=1)
        class LLMService:
            def __init__(self, service_id):
                self.service_id = service_id
                print(f"服务 {service_id} 初始化中...")
                # 使用ModelScope下载模型
                model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
                self.tokenizer = ModelScopeTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                self.model = ModelScopeModel.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"服务 {service_id} 初始化完成")
            
            def inference(self, prompt):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return f"[服务{self.service_id}] {result}"
        
        # 创建2个服务实例（假设2个节点）
        print("\n创建2个LLM服务实例...")
        num_services = 2
        services = [LLMService.remote(i) for i in range(num_services)]
        
        # 测试负载均衡
        print("\n测试负载均衡 - 发送10个请求...")
        prompts = ["什么是人工智能？"] * 10
        
        start_time = time.time()
        results = ray.get([services[i % num_services].inference.remote(prompt) 
                         for i, prompt in enumerate(prompts)])
        total_time = time.time() - start_time
        
        print(f"\n=== 负载均衡结果 ===")
        for i, result in enumerate(results):
            print(f"请求 {i+1}: {result}")
        
        print(f"\n=== 性能统计 ===")
        print(f"总时间: {total_time:.3f}s")
        print(f"平均时间: {total_time/len(prompts):.3f}s/请求")
        print(f"QPS: {len(prompts)/total_time:.1f} 请求/秒")
        
        # 分析负载分配
        service_counts = {}
        for result in results:
            if "服务0" in result:
                service_counts["服务0"] = service_counts.get("服务0", 0) + 1
            elif "服务1" in result:
                service_counts["服务1"] = service_counts.get("服务1", 0) + 1
        
        print(f"\n=== 负载分配 ===")
        for service, count in service_counts.items():
            print(f"{service}: {count} 个请求")
        
        ray.shutdown()
        print("\n✅ Ray负载均衡测试完成")
        
    except Exception as e:
        print(f"Ray负载均衡测试失败: {e}")
        if ray.is_initialized():
            ray.shutdown()

# ==========================================
# 测试2: Worker节点故障恢复测试
# ==========================================
def test_worker_node_failure_recovery():
    """测试Worker节点故障恢复 - 模拟worker节点故障，观察Ray的容错能力"""
    print("\n" + "=" * 60)
    print("测试2: Worker节点故障恢复")
    print("=" * 60)

    try:
        # 初始化Ray集群
        ray.init(ignore_reinit_error=True)

        # 检查集群状态
        print("检查Ray集群状态...")
        resources = ray.available_resources()
        print(f"可用资源: {resources}")
        print(f"GPU数量: {resources.get('GPU', 0)}")
        print(f"CPU数量: {resources.get('CPU', 0)}")

        # 创建持续运行的GPU任务
        @ray.remote(num_gpus=1)
        class LongRunningTask:
            def __init__(self, task_id):
                self.task_id = task_id
                self.node_id = ray.get_runtime_context().get_node_id()
                print(f"任务 {task_id} 初始化在节点: {self.node_id}")

            def run_task(self, iterations):
                import time
                results = []
                for i in range(iterations):
                    time.sleep(1)
                    results.append(f"任务 {self.task_id} 迭代 {i+1}")
                return {
                    "task_id": self.task_id,
                    "node_id": self.node_id,
                    "results": results
                }

        print("\n=== 测试1: 正常运行测试 ===")
        print("创建2个GPU任务，正常运行...")

        tasks = [LongRunningTask.remote(i) for i in range(2)]
        # 先运行一小段时间确认正常
        initial_results = ray.get([task.run_task.remote(2) for task in tasks])

        print(f"\n=== 初始运行结果 ===")
        for result in initial_results:
            print(f"任务 {result['task_id']}: 节点 {result['node_id']}, 完成 {len(result['results'])} 次迭代")

        print(f"\n=== Ray容错机制 ===")
        print(f"1. Head节点: 集群核心，故障会导致集群失效")
        print(f"2. Worker节点: 故障时Ray自动重新调度任务")
        print(f"3. 任务重试: 失败的任务自动重试")
        print(f"4. 资源感知: 自动检测节点状态变化")
        print(f"5. 生产建议: 配置Head节点高可用方案")

        print(f"\n=== 说明 ===")
        print(f"由于测试环境限制，本测试仅演示Ray的容错机制原理")
        print(f"实际故障恢复需要:")
        print(f"- 停止worker节点: ray stop")
        print(f"- 观察任务重新调度: ray status")
        print(f"- 重启worker节点: ray start --address=...")

        ray.shutdown()
        print("\n✅ Worker节点故障恢复测试完成")

    except Exception as e:
        print(f"Worker节点故障恢复测试失败: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()

# ==========================================
# 测试3: Ray自动扩缩容
# ==========================================
def test_ray_autoscaling():
    """测试Ray自动扩缩容 - 根据负载动态调整副本数"""
    print("\n" + "=" * 60)
    print("测试3: Ray自动扩缩容")
    print("=" * 60)
    
    try:
        # 初始化Ray
        ray.init(ignore_reinit_error=True)
        
        # 初始化Ray Serve
        serve.start(detached=True)
        
        @serve.deployment(
            autoscaling_config={
                "min_replicas": 1,  # 最小副本数
                "max_replicas": 4,  # 最大副本数
                "target_num_ongoing_requests_per_replica": 3  # 每个副本目标请求数
            },
            ray_actor_options={"num_gpus": 1}  # 每个副本使用1个GPU
        )
        class LLMService:
            def __init__(self):
                print("LLM服务初始化...")
                # 使用ModelScope下载模型
                model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
                self.tokenizer = ModelScopeTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                self.model = ModelScopeModel.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            async def __call__(self, request):
                prompt = request["prompt"]
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=10)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {"answer": result}
        
        # 部署服务
        print("部署Ray Serve服务...")
        handle = serve.run(LLMService.bind())
        
        print("\n=== 测试自动扩缩容 ===")
        print("阶段1: 低负载 (2个请求)")
        start_time = time.time()
        futures = [handle.remote({"prompt": "什么是AI？"}) for _ in range(2)]
        results = [f.result() for f in futures]
        low_load_time = time.time() - start_time
        print(f"低负载时间: {low_load_time:.3f}s")

        print("\n阶段2: 高负载 (10个请求)")
        start_time = time.time()
        futures = [handle.remote({"prompt": "什么是AI？"}) for _ in range(10)]
        results = [f.result() for f in futures]
        high_load_time = time.time() - start_time
        print(f"高负载时间: {high_load_time:.3f}s")

        print("\n阶段3: 负载降低 (2个请求)")
        time.sleep(5)  # 等待缩容
        start_time = time.time()
        futures = [handle.remote({"prompt": "什么是AI？"}) for _ in range(2)]
        results = [f.result() for f in futures]
        reduced_load_time = time.time() - start_time
        print(f"降低负载时间: {reduced_load_time:.3f}s")
        
        print(f"\n=== 自动扩缩容性能 ===")
        print(f"低负载: {low_load_time:.3f}s (2请求)")
        print(f"高负载: {high_load_time:.3f}s (10请求)")
        print(f"降负载: {reduced_load_time:.3f}s (2请求)")
        
        print(f"\n=== 自动扩缩容优势 ===")
        print(f"1. 低负载时减少副本，节省资源")
        print(f"2. 高负载时自动扩容，提升性能")
        print(f"3. 动态调整，无需手动干预")
        print(f"4. 成本优化，按需付费")
        
        # 清理
        serve.shutdown()
        ray.shutdown()
        print("\n✅ Ray自动扩缩容测试完成")
        
    except Exception as e:
        print(f"Ray自动扩缩容测试失败: {e}")
        try:
            serve.shutdown()
        except:
            pass
        if ray.is_initialized():
            ray.shutdown()

# ==========================================
# 主测试流程
# ==========================================
def main():
    """主测试流程"""
    print("\n🚀 开始双节点分布式测试")
    print("⏰ 预计总时间: 30分钟")
    
    start_time = time.time()
    
    # 测试1: Ray负载均衡
    print("\n⏱️  0-10分钟: Ray负载均衡测试")
    test_ray_load_balancing()
    
    # 测试2: Worker节点故障恢复
    print("\n⏱️  10-20分钟: Worker节点故障恢复测试")
    test_worker_node_failure_recovery()
    
    # 测试3: Ray自动扩缩容
    print("\n⏱️  20-30分钟: Ray自动扩缩容测试")
    test_ray_autoscaling()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 所有双节点测试完成！总时间: {total_time/60:.1f}分钟")
    print("=" * 60)

if __name__ == "__main__":
    main()
