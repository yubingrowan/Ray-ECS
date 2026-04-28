#!/usr/bin/env python3
"""
简单的Transformer实现
用于理解Transformer的基本原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 线性变换
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, hidden_dim]
            key: [batch_size, seq_len, hidden_dim]
            value: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len, seq_len] (可选)
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size = query.size(0)
        
        # 计算Q, K, V
        Q = self.W_Q(query)  # [batch, seq_len, hidden_dim]
        print(f"Q shape: {Q.shape}")
        K = self.W_K(key)    # [batch, seq_len, hidden_dim]
        print(f"K shape: {K.shape}")
        V = self.W_V(value)  # [batch, seq_len, hidden_dim]
        print(f"V shape: {V.shape}")
        
        # 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print(f"scores shape: {scores.shape}")
        # scores: [batch, num_heads, seq_len, seq_len]
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 可视化注意力权重（只在第一次调用时）
        if not hasattr(self, 'visualized'):
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_weights[0, 0].detach().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title('Attention Weights (Head 0)')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.savefig('attention_weights.png')
            print("注意力权重已保存到 attention_weights.png")
            self.visualized = True

        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch, num_heads, seq_len, head_dim]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        print(f"attn_output shape: {attn_output.shape}")
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        # attn_output: [batch, seq_len, hidden_dim]
        
        # 输出投影
        output = self.W_O(attn_output)
        print(f"output shape: {output.shape}")
        
        return output


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len, seq_len] (可选)
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        # 多头注意力 + 残差连接 + LayerNorm
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x


class Transformer(nn.Module):
    """完整的Transformer"""
    
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len]
            mask: [batch_size, seq_len, seq_len] (可选)
        Returns:
            output: [batch_size, seq_len, vocab_size]
        """
        print(f"......forward input shape: {x.shape}")
        batch_size, seq_len = x.size()
        
        # 词嵌入 + 位置嵌入
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        token_emb = self.token_embedding(x)  # [batch, seq_len, hidden_dim]
        pos_emb = self.position_embedding(positions)  # [1, seq_len, hidden_dim]
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出
        output = self.fc_out(x)

        
        return output


def test_transformer():
    """测试Transformer"""
    print("=" * 60)
    print("测试Transformer")
    print("=" * 60)
    
    # 参数
    vocab_size = 10000
    hidden_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    max_seq_len = 128
    batch_size = 4
    seq_len = 32
    
    print(f"参数:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  ff_dim: {ff_dim}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print()
    
    # 创建模型
    model = Transformer(vocab_size, hidden_dim, num_heads, num_layers, ff_dim, max_seq_len)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()
    
    # 测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入shape: {x.shape}")
    
    # 前向传播
    output = model(x)
    print(f"输出shape: {output.shape}")
    print()
    
    # 测试多头注意力
    print("测试多头注意力:")
    mha = MultiHeadAttention(hidden_dim, num_heads)
    query = torch.randn(batch_size, seq_len, hidden_dim)
    key = torch.randn(batch_size, seq_len, hidden_dim)
    value = torch.randn(batch_size, seq_len, hidden_dim)
    
    attn_output = mha(query, key, value)
    print(f"  输入shape: {query.shape}")
    print(f"  输出shape: {attn_output.shape}")
    print()
    
    # 测试前馈网络
    print("测试前馈网络:")
    ffn = FeedForwardNetwork(hidden_dim, ff_dim)
    x_ffn = torch.randn(batch_size, seq_len, hidden_dim)
    ffn_output = ffn(x_ffn)
    print(f"  输入shape: {x_ffn.shape}")
    print(f"  输出shape: {ffn_output.shape}")
    print()
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_transformer()
