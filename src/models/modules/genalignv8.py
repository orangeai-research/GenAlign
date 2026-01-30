'''
Desc@: Official Implementation of GenAlign (KDD2026)
Time: 2026-01-28
Author: OrangeAI Research Team
Paper: Agent Modality: Generative Multimodal Alignment via Rectified Flow for Recommendation 
GenAlign: Dual Generative Alignment via Rectified Flow for Multimodal Recommendation
GenAlign: Generative Alignment via Rectified Flow for Multimodal Recommendation
Version8: a version of GenAlign with Transformer-based Velocity Network
'''

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedFlow(nn.Module):
    '''
    Desc@: Rectified Flow for Cross-Modal Alignment
    Args:
        dim: dimension of the input and output
        hidden_dim: dimension of the hidden layer
    Returns:
        output: the output of the Rectified Flow
    Version: v1 
    '''
    def __init__(self, dim, hidden_dim=None):
        super(RectifiedFlow, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def compute_loss(self, x0, x1):
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device)
        z_t = t * x1 + (1 - t) * x0
        v_target = x1 - x0 # 目标传输向量v_target：图像到文本的真实差值（最优传输方向）跨模态语义gap
        v_pred = self.forward(z_t, t) # 模型预测的 t 时刻传输方向
        loss = F.mse_loss(v_pred, v_target)
        return loss


class TransformerVelocityNet(nn.Module):
    """
    Desc@: Transformer-based Velocity Network (DiT-like Architecture)
    Uses Cross-Attention to inject Multimodal Condition
    """
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # Input Projection for x (State)
        self.x_proj = nn.Linear(dim, dim)
        
        # Input Projection for condition (Key/Value)
        self.c_proj = nn.Linear(dim, dim)

        # Transformer Decoder Layers (Self-Attn + Cross-Attention + FFN)
        # d_model=dim
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, 
                nhead=num_heads, 
                dim_feedforward=dim*4, 
                dropout=dropout, 
                activation='gelu', # GELU is standard for Transformers
                batch_first=True,
                norm_first=True # Pre-Norm is more stable
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Initialize weights (DiT style zero-init for last layer often helps)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, t, condition):
        """
        x: (batch, dim) -> Query source
        t: (batch, 1) -> Time signal
        condition: (batch, dim) -> Key/Value source
        """
        batch_size = x.size(0)
        
        # 1. Prepare Query (State + Time)
        # Project x and t to same dimension
        x_emb = self.x_proj(x)
        t_emb = self.time_mlp(t)
        
        # Add Time Embedding to State (Time Injection)
        # Treat as a sequence of length 1: (batch, 1, dim)
        query = (x_emb + t_emb).unsqueeze(1) 
        
        # 2. Prepare Key/Value (Condition)
        # Treat as a sequence of length 1: (batch, 1, dim)
        # If we had multiple modalities (Img, Text), we could concat them here: (batch, 2, dim)
        kv = self.c_proj(condition).unsqueeze(1)
        
        # 3. Transformer Forward
        h = query
        for block in self.transformer_blocks:
            # tgt=h (Query), memory=kv (Key/Value)
            h = block(tgt=h, memory=kv)
            
        # 4. Output Projection
        # Squeeze back to (batch, dim)
        return self.out_proj(h.squeeze(1))


class ConditionalRectifiedFlow(nn.Module):
    """
    Desc@: Conditional Rectified Flow with Transformer Backbone (DiT)
    """
    def __init__(self, dim, hidden_dim=None):
        super(ConditionalRectifiedFlow, self).__init__()
        
        # 使用 Transformer Backbone
        # num_heads 和 num_layers 可以作为超参调节
        # 这里使用轻量级配置: 2 Layers, 4 Heads
        self.net = TransformerVelocityNet(dim, num_heads=4, num_layers=2)

    def forward(self, x, t, condition):
        return self.net(x, t, condition)

    def compute_loss(self, target_id, condition_feat):
        """
        计算 Flow Matching Loss (MSE + Cosine)
        Args:
            target_id (x1): 目标 ID Embedding (Sparse Signal)
            condition_feat (c): 多模态特征 (Dense Signal)
        """
        batch_size = target_id.size(0)
        
        # 1. 定义起点 x0: 高斯噪声
        x0 = torch.randn_like(target_id)
        
        # 2. 定义终点 x1: 真实的协同 ID
        x1 = target_id
        
        # [Math Optimization] Mini-Batch Optimal Transport
        # 在 Batch 内重新配对 x0 和 x1，使得传输距离最小，路径更直
        # 简单的贪心策略：按 Norm 排序近似 OT
        # 或者直接计算 Pairwise Distance 矩阵然后求解（计算量大）
        # 这里使用一种简单有效的近似：Sorting
        # 对 x0 和 x1 分别按第一主成分投影排序，或者简单按 Norm 排序
        # 考虑到推荐系统 Embedding 的特性，我们不做复杂的 OT，
        # 而是保留随机配对，因为推荐系统中的 User/Item 对是固定的（Condition 是固定的）
        # 注意：这里是 Conditional Flow Matching，x1 (ID) 和 condition 是绑定的！
        # 我们不能打乱 x1 和 condition 的对应关系！
        # 唯一能动的是 x0 (噪声)。
        # 所以 OT 的含义是：对于给定的 (x1, c)，选择一个最优的 x0。
        # 理论上 x0 ~ N(0, I)，我们可以选择离 x1 最近的 x0 吗？
        # 不行，因为推理时我们无法预知哪个 x0 离目标近。
        # 所以对于 Conditional Flow，通常不做 OT 配对，除非是无条件生成。
        # 
        # 因此，这里的数学优化我们采用：**Logit-Normal Sampling for t**
        # 使得训练集中在中间区域（更难学的区域），而不是均匀分布
        
        # 3. 采样时间 t (Logit-Normal Sampling)
        # t = sigmoid(sigma * z + mu), z ~ N(0, 1)
        # 这种采样方式让 t 更集中在 0.5 附近，避免边缘 0 和 1 的无效训练
        mu, sigma = 0.0, 1.0
        t_noise = torch.randn(batch_size, 1, device=x0.device)
        t = torch.sigmoid(mu + sigma * t_noise)
        # 4. 构造插值路径 (Interpolation)
        z_t = t * x1 + (1 - t) * x0
        # 5. 计算目标速度 (Target Velocity)
        v_target = x1 - x0
        # 6. 模型预测速度 (Predicted Velocity)
        v_pred = self.forward(z_t, t, condition_feat)
        # 7. MSE Loss
        mse_loss = F.mse_loss(v_pred, v_target)
        
        # [Math Optimization] Velocity Consistency Regularization
        # 鼓励预测速度的模长（Magnitude）与目标速度的模长一致
        # 这有助于稳定训练，防止生成向量发散
        v_pred_norm = torch.norm(v_pred, dim=-1)
        v_target_norm = torch.norm(v_target, dim=-1)
        norm_loss = F.mse_loss(v_pred_norm, v_target_norm)
        
        # Total Loss
        loss = mse_loss + 0.1 * norm_loss
        
        return loss

    def generate_agent_modality(self, condition_feat, t_value=0.5, steps=1):
        """
        推理阶段：生成 t=t_value 的 Agent Modality
        支持多步欧拉采样 (Multi-step Euler Sampling)
        """
        batch_size = condition_feat.size(0)
        
        # 1. 起点：采样高斯噪声 (z_0)
        z = torch.randn_like(condition_feat)
        
        # 2. 计算步长
        dt = t_value / steps
        
        # 3. 逐步迭代
        for i in range(steps):
            # 当前时间 t
            t_current = i * dt
            t_tensor = torch.full((batch_size, 1), t_current, device=condition_feat.device)
            
            # 预测当前位置的速度 v
            v_pred = self.forward(z, t_tensor, condition_feat)
            
            # 更新位置：z_{t+dt} = z_t + v * dt
            z = z + v_pred * dt
        
        agent_modality = z
        return agent_modality

        