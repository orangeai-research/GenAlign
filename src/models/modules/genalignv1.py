'''
Desc@: Official Implementation of GenAlign (KDD2026)
Time: 2026-01-28
Author: OrangeAI Research Team
Paper: Agent Modality: Generative Multimodal Alignment via Rectified Flow for Recommendation 
Version 1
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


class ConditionalRectifiedFlow(nn.Module):
    """
    Desc@: Conditional Rectified Flow for Cross-Modal and Colabrative ID Singal Alignment
        Conditioned Rectified Flow for: Noise (x0) -> Agent (t=0.5) -> ID (x1), Conditioned on Multimodal Features
    Args:
        dim: dimension of the input and output
        hidden_dim: dimension of the hidden layer
    Returns:
        output: the output of the Conditional Rectified Flow (Agent Modality)
    Version: v1
    """
    def __init__(self, dim, hidden_dim=None):
        super(ConditionalRectifiedFlow, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2  # 稍微增大隐藏层，因为输入维度变大了
            
        # 输入: x(dim) + t(1) + condition(dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim) # 输出仅为 velocity (维度与 x 一致)
        )
    
    def forward(self, x, t, condition):
        # 将 噪声/中间态、时间、多模态条件 拼接
        xtc = torch.cat([x, t, condition], dim=-1)
        return self.net(xtc)

    def compute_loss(self, target_id, condition_feat):
        """
        计算 Flow Matching Loss
        Args:
            target_id (x1): 目标 ID Embedding (Sparse Signal)
            condition_feat (c): 多模态特征 (Dense Signal)
        """
        batch_size = target_id.size(0)
        
        # 1. 定义起点 x0: 高斯噪声
        x0 = torch.randn_like(target_id)
        
        # 2. 定义终点 x1: 真实的协同 ID
        x1 = target_id
        
        # 3. 采样时间 t
        t = torch.rand(batch_size, 1, device=x0.device)
        
        # 4. 构造插值路径 (Interpolation)
        z_t = t * x1 + (1 - t) * x0
        
        # 5. 计算目标速度 (Target Velocity)
        v_target = x1 - x0
        
        # 6. 模型预测速度 (Predicted Velocity)
        # 关键：模型必须看到 condition 才能知道往哪走
        v_pred = self.forward(z_t, t, condition_feat)
        
        # 7. MSE Loss
        loss = F.mse_loss(v_pred, v_target)
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

        