# 架构说明

## 系统架构

### 整体架构

Ray GPU集群采用Master-Worker架构，包含一个Head节点和多个Worker节点。

```
┌─────────────────────────────────────────────────┐
│              Ray Head Node                      │
│  - Ray Dashboard (8265)                         │
│  - GCS Server (6379)                            │
│  - Resource Scheduler                          │
│  - Object Store                                 │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼───────┐ ┌────▼────────┐
│ Worker Node 1 │ │ Worker Node 2│
│ - GPU (A10)   │ │ - GPU (A10)  │
│ - vLLM Service│ │ - vLLM Service│
│ - Object Store│ │ - Object Store│
└──────────────┘ └─────────────┘
```

## 组件说明

### 1. Head节点

**职责:**
- 集群元数据管理
- 任务调度和资源分配
- 监控和日志收集
- Dashboard服务

**主要组件:**
- **GCS (Global Control Service)**: 集群状态管理
- **Raylet**: 本地任务调度器
- **Dashboard**: Web监控界面
- **Object Store**: 分布式对象存储

### 2. Worker节点

**职责:**
- 执行计算任务
- GPU资源管理
- 本地对象存储
- 健康状态上报

**主要组件:**
- **Raylet**: 本地任务调度器
- **Object Store**: 本地对象存储
- **Worker**: 任务执行进程
- **vLLM Engine**: LLM推理引擎

## Ray调度机制

### 资源调度

Ray采用基于资源的调度策略：

1. **资源声明**: 任务声明需要的资源（CPU、GPU、内存）
2. **资源匹配**: 调度器将任务分配到资源充足的节点
3. **负载均衡**: 避免单节点过载
4. **抢占式调度**: 高优先级任务可以抢占资源

### 任务调度

**调度流程:**
```
1. 任务提交 → Ray Client
2. 资源请求 → GCS
3. 节点选择 → Resource Scheduler
4. 任务分配 → Worker Raylet
5. 任务执行 → Worker
6. 结果返回 → Object Store
7. 结果获取 → Ray Client
```

## vLLM推理优化

### PagedAttention

**原理:**
- 将KV Cache分页管理
- 按需分配显存
- 减少显存碎片

**优势:**
- 显存利用率提升2-4倍
- 支持更长上下文
- 减少OOM风险

### 连续批处理

**原理:**
- 将多个请求合并处理
- 共享KV Cache
- 减少显存占用

**优势:**
- 吞吐量提升2-3倍
- 延迟降低
- GPU利用率提高

## Ray Serve自动扩缩容

### 扩缩容策略

**扩容触发条件:**
- 请求队列长度超过阈值
- 平均延迟超过阈值
- CPU/GPU利用率超过阈值

**缩容触发条件:**
- 请求队列长度低于阈值
- 副本空闲时间超过阈值
- 成本优化需求

### 扩缩容流程

```
1. 监控指标收集
2. 阈值判断
3. 副本数调整
4. 流量重新分配
5. 状态同步
```

## 故障恢复机制

### Worker节点故障

**检测机制:**
- 心跳检测
- 状态监控
- 超时判断

**恢复流程:**
```
1. 检测节点故障
2. 标记节点不可用
3. 重新调度任务
4. 等待节点恢复
5. 重新加入集群
```

### 任务重试

**重试策略:**
- 指数退避
- 最大重试次数
- 任务状态保存

## 网络架构

### 通信模式

- **控制平面**: Head ↔ Worker (GCS)
- **数据平面**: Worker ↔ Worker (Object Store)
- **客户端**: Client ↔ Head (Dashboard)

### 端口配置

| 服务 | 端口 | 协议 | 用途 |
|------|------|------|------|
| GCS | 6379 | TCP | 集群控制 |
| Dashboard | 8265 | HTTP | Web监控 |
| Object Store | 自动分配 | TCP | 对象存储 |
| Client | 自动分配 | TCP | 客户端通信 |

## 数据流

### 推理请求流程

```
Client → Ray Serve → vLLM Service → GPU → Response
```

### 对象存储流程

```
Task → Local Object Store → Remote Object Store → Consumer
```

## 性能指标

### 关键指标

- **吞吐量**: QPS (Queries Per Second)
- **延迟**: P50/P95/P99
- **资源利用率**: CPU/GPU/内存
- **错误率**: 任务失败率
- **扩缩容时间**: 副本调整耗时

### 监控方式

- Ray Dashboard实时监控
- Prometheus指标导出
- 自定义日志分析

## 安全考虑

### 网络安全

- 私网通信
- 防火墙规则
- SSH密钥认证

### 资源隔离

- GPU资源隔离
- 内存限制
- 进程隔离

## 扩展性

### 水平扩展

- 添加Worker节点
- 无需重启Head节点
- 自动资源发现

### 垂直扩展

- 增加GPU数量
- 升级硬件配置
- 调整资源配额
