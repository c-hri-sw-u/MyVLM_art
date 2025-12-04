"""
简要介绍 (ZH):
  概念图数据集。单个样本同时包含艺术家、风格、题材、媒介四个维度标签；
  负责：
    - 从图片构造 CLIP 原型相似度（s_c）作为概念激活信号
    - 组织联合训练的文本目标（解释/分类）与 InfoNCE 负样本集合

Overview (EN):
  Concept-graph dataset. Each sample carries multi-dimensional labels (artist/style/genre/media). It forms
  prototype similarities per dimension for activation, assembles textual targets for explanation/classification,
  and prepares negative sets for InfoNCE across dimensions.

Schema:
  - inputs: { images, labels_per_dim }
  - outputs: { concept_signals (per concept), prompts, targets, negatives (per dimension) }

TODOs (详细):
  1) 定义 Dataset 类：__len__ / __getitem__，返回包含图像张量、概念信号、prompt/target、InfoNCE 负样本
  2) 原型计算：调用 concept_graph/prototypes/prototype_head.py 提供的接口，得到 s_c
  3) 文本构造：复用 reasoning/prompt_templates.py，根据激活概念生成解释或分类指令
  4) 负样本：同维度随机采 K 个概念作为 negatives，用于 InfoNCE
"""

import torch
from torch.utils.data import Dataset


class ConceptGraphDataset(Dataset):
    def __init__(self, inputs, labels_per_dim, prototype_head, processor=None, transforms=None, device="cuda"):
        self.inputs = inputs
        self.labels_per_dim = labels_per_dim
        self.prototype_head = prototype_head
        self.processor = processor
        self.transforms = transforms
        self.device = device
        # TODO: 预计算或按需计算原型相似度 s_c，并缓存概念信号

    def __len__(self):
        return len(self.inputs["images"]) if "images" in self.inputs else 0

    def __getitem__(self, idx):
        # TODO: 返回包含图像张量、概念信号、提示词与目标文本、InfoNCE 负样本集合的字典
        return {}
