"""
PrototypeTrainer
---------------
使用最小化的 InfoNCE 目标，在单一概念维度上微调 `PrototypeHead` 中的原型向量。
Overview (EN):
  Prototype trainer using InfoNCE across concept dimensions. Updates per-concept prototypes to better separate
  concepts within the same dimension while keeping cross-dimension coherence.

TODOs (详细):
  1) 数据采样：按维度取正样本与 K 个负样本，计算 CLIP 特征并归一化
  2) 损失设计：维度内 InfoNCE；可选跨维度正则（避免原型跨维度过近）
  3) 原型更新：动量更新或直接梯度（若将 p_c 设为可训练参数）
  4) 检查点：保存每维度所有概念的 p_c，供推理与注入使用

期望 cfg 属性：
    - dimension: 目标概念维度（如 "artist"）
    - batch_size: DataLoader 批大小
    - lr: 原型参数学习率
    - epochs: 迭代轮数
    - temperature: InfoNCE 温度系数
    - save_path (可选): 训练结束后保存原型检查点的路径
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class PrototypeTrainerConfig:
    dimension: str
    batch_size: int = 8
    lr: float = 5e-3
    epochs: int = 5
    temperature: float = 0.07
    save_path: Optional[str] = None


class PrototypeTrainer:
    def __init__(self,
                 cfg: PrototypeTrainerConfig,
                 dataset,
                 prototype_head):
        self.cfg = cfg
        self.dataset = dataset
        self.prototype_head = prototype_head
        self.dimension = cfg.dimension

        if self.dimension not in self.prototype_head.prototypes:
            raise ValueError(f"Dimension {self.dimension} not found in PrototypeHead.")

        self.device = self.prototype_head.device
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True
        )
        self._prepare_train_state()

    def _prepare_train_state(self):
        """
        将目标维度的原型向量设置为可训练参数，并初始化优化器。
        """
        tensor = self.prototype_head.prototypes[self.dimension].to(self.device)
        self.prototype_param = torch.nn.Parameter(tensor.clone())
        self.optimizer = torch.optim.Adam([self.prototype_param], lr=self.cfg.lr)
        self.label_to_idx: Dict[str, int] = self.prototype_head.concept_to_idx[self.dimension]

    def train(self) -> Dict[str, float]:
        self.prototype_head.model.eval()
        log: Dict[str, float] = {}

        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            step = 0
            for batch in self.dataloader:
                labels = batch["label"]
                images = batch["image_anchor"]

                # 过滤掉没有标签或不在映射里的样本
                valid_indices = [
                    idx for idx, label in enumerate(labels)
                    if label is not None and label in self.label_to_idx
                ]
                if len(valid_indices) == 0:
                    continue

                images = images[valid_indices].to(self.device)
                target = torch.tensor(
                    [self.label_to_idx[labels[idx]] for idx in valid_indices],
                    device=self.device,
                    dtype=torch.long
                )

                with torch.no_grad():
                    feats = self.prototype_head.model.encode_image(images)
                    feats = F.normalize(feats.float(), dim=-1)

                logits = feats @ self.prototype_param.T
                logits = logits / self.cfg.temperature
                loss = F.cross_entropy(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 每次更新后重新归一化原型向量，保持其在单位球上
                with torch.no_grad():
                    self.prototype_param.copy_(F.normalize(self.prototype_param, dim=-1))

                epoch_loss += loss.item()
                step += 1

            if step > 0:
                log[f"epoch_{epoch}_loss"] = epoch_loss / step

        # 回写更新后的原型
        self.prototype_head.prototypes[self.dimension] = self.prototype_param.detach().cpu()

        if self.cfg.save_path:
            save_path = Path(self.cfg.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "clip_model_name": self.prototype_head.clip_model_name,
                "prototypes": self.prototype_head.prototypes,
                "concept_to_idx": self.prototype_head.concept_to_idx,
                "idx_to_concept": self.prototype_head.idx_to_concept,
            }
            torch.save(payload, save_path)

        return log
