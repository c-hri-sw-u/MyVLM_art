"""
简要介绍 (ZH):
  原型概念头。为每个概念维护 CLIP 特征空间的原型向量 p_c（如 ViT‑B/16, 512 维），
  计算查询图像的相似度 s_c=cos_sim(CLIP(x_q), p_c)，并按阈值 τ 激活。

Overview (EN):
  Prototype-based concept head. Maintains per-concept prototypes in CLIP feature space and computes cosine
  similarities for activation. Serves as the concept detection backbone for multi-granular concept graph.

Outputs & Interface:
  - extract_signal(image_paths) -> Dict[Path, Dict[concept_idx, Tensor([1-s, s])]]
    将相似度 s 映射为伪概率 [1-s, s] 以复用现有对象分支距离定义。

TODOs (详细):
  1) 加载/选择 CLIP 模型（建议 ViT‑B/16），实现特征抽取与归一化
  2) 原型初始化：按概念聚合若干参考图像，求均值并 L2 归一化；支持离线保存/加载
  3) extract_signal：批量处理图片，计算 s_c 并输出兼容字典格式
  4) 阈值 τ 管理：仅标记激活概念，但输出全量 s_c 以供注入层 gating
"""

import torch
from torch import Tensor
from typing import Dict, List


class PrototypeHead:
    def __init__(self, clip_model_name: str = "ViT-B/16", device: str = "cuda"):
        self.clip_model_name = clip_model_name
        self.device = device
        self.model = None
        self.preprocess = None
        self.prototypes = None  # 形状: [n_concepts, dim]
        # TODO: 加载 CLIP 模型与预处理；初始化 prototypes

    def extract_signal(self, image_paths: List):
        # TODO: 对每张图计算特征，与 prototypes 做余弦相似度，映射为 {concept_idx: Tensor([1-s, s])}
        return {}
