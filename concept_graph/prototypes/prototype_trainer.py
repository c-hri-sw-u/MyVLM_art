"""
简要介绍 (ZH):
  原型训练器。通过 InfoNCE 在每个概念维度（艺术家/风格/题材/媒介）上对比学习，
  更新概念原型 p_c，使其更好地区分同维度内的概念。

Overview (EN):
  Prototype trainer using InfoNCE across concept dimensions. Updates per-concept prototypes to better separate
  concepts within the same dimension while keeping cross-dimension coherence.

TODOs (详细):
  1) 数据采样：按维度取正样本与 K 个负样本，计算 CLIP 特征并归一化
  2) 损失设计：维度内 InfoNCE；可选跨维度正则（避免原型跨维度过近）
  3) 原型更新：动量更新或直接梯度（若将 p_c 设为可训练参数）
  4) 检查点：保存每维度所有概念的 p_c，供推理与注入使用
"""

import torch


class PrototypeTrainer:
    def __init__(self, cfg, dataset, prototype_head):
        self.cfg = cfg
        self.dataset = dataset
        self.prototype_head = prototype_head
        # TODO: 初始化优化器/动量参数，准备维度内的概念列表

    def train(self):
        # TODO: 实现 InfoNCE 训练循环与原型更新逻辑
        pass
