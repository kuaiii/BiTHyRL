# -*- coding: utf-8 -*-
"""BiT-HyRL 策略网络 (MLP with Residual Connections)。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块：帮助梯度流动，避免梯度消失。"""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x + residual)  # 残差连接 + LayerNorm
        return F.relu(x)


class MLPPolicy(nn.Module):
    """
    改进的策略网络：
    - 更大的隐藏层 (128)
    - 残差连接帮助梯度流动
    - LayerNorm 稳定训练
    - Dropout 防止过拟合
    """

    def __init__(self, num_features, hidden_dim=64, output_dim=1, dropout=0.1):
        super(MLPPolicy, self).__init__()
        # 扩展隐藏层维度
        actual_hidden = max(hidden_dim, 128)
        
        # 输入投影
        self.input_proj = nn.Linear(num_features, actual_hidden)
        self.input_ln = nn.LayerNorm(actual_hidden)
        
        # 残差块
        self.res_block1 = ResidualBlock(actual_hidden, dropout)
        self.res_block2 = ResidualBlock(actual_hidden, dropout)
        
        # 额外的特征混合层
        self.fc_mix = nn.Linear(actual_hidden, actual_hidden)
        self.mix_ln = nn.LayerNorm(actual_hidden)
        
        # 输出层
        self.actor = nn.Linear(actual_hidden, output_dim)
        
        # 温度参数：可学习的 softmax 温度，帮助探索
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        # 输入投影
        x = F.relu(self.input_ln(self.input_proj(x)))
        
        # 残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 特征混合
        x = F.relu(self.mix_ln(self.fc_mix(x)))
        
        # 输出分数
        scores = self.actor(x).squeeze(-1)
        
        # 应用温度缩放（clamp防止温度过小导致数值问题）
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        return probs
