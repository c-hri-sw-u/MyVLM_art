from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp


@dataclass
class PrototypeTrainerConfig:
    dimension: str
    batch_size: int = 8
    lr: float = 5e-3
    epochs: int = 5
    temperature: float = 0.07
    save_path: Optional[str] = None
    negatives_k: int = 16
    hard_negatives: bool = True
    margin: float = 0.0
    lambda_margin: float = 0.0
    debug_log: bool = False
    log_interval: int = 50


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
        Set requires_grad=True for the prototypes of the target dimension. Initialize the optimizer.
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
                labels_valid = [labels[idx] for idx in valid_indices]
                pos_idx = torch.tensor(
                    [self.label_to_idx[l] for l in labels_valid],
                    device=self.device,
                    dtype=torch.long
                )

                with torch.no_grad():
                    with amp.autocast(enabled=self.prototype_head.precision == "fp16"):
                        feats = self.prototype_head.model.encode_image(images)
                    feats = F.normalize(feats.float(), dim=-1)

                num_proto = self.prototype_param.size(0)
                # negatives_k 已在训练脚本解析，若超界，这里再做一次保护
                k = min(max(1, self.cfg.negatives_k), max(0, num_proto - 1))
                if k == 0:
                    continue

                sims_full = feats @ self.prototype_param.T
                # 仅使用原型作为负样本，不引入样本级负样本
                if self.cfg.hard_negatives:
                    mask = torch.ones_like(sims_full, dtype=torch.bool)
                    mask.scatter_(1, pos_idx.view(-1, 1), False)
                    masked = sims_full.masked_fill(~mask, float("-inf"))
                    neg_idx_tensor = masked.topk(k, dim=1, largest=True).indices
                else:
                    neg_indices = []
                    for p in pos_idx.tolist():
                        pool = list(range(num_proto))
                        pool.remove(p)
                        sample = pool if len(pool) <= k else random.sample(pool, k)
                        neg_indices.append(sample)
                    neg_idx_tensor = torch.tensor(neg_indices, device=self.device, dtype=torch.long)

                pos_sim = sims_full.gather(1, pos_idx.view(-1, 1))
                neg_sim = sims_full.gather(1, neg_idx_tensor)
                logits = torch.cat([pos_sim, neg_sim], dim=1) / self.cfg.temperature
                target = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
                ce = F.cross_entropy(logits, target)

                max_neg = neg_sim.max(dim=1).values
                margin_loss = torch.clamp(self.cfg.margin - pos_sim.squeeze(1) + max_neg, min=0).mean()
                loss = ce + self.cfg.lambda_margin * margin_loss

                if self.cfg.debug_log and (step % max(1, self.cfg.log_interval) == 0):
                    pos_mean = float(pos_sim.mean().item())
                    max_neg_mean = float(max_neg.mean().item())
                    margin_mean = float(torch.clamp(self.cfg.margin - pos_sim.squeeze(1) + max_neg, min=0).mean().item())
                    print(f"[epoch {epoch} step {step}] pos_sim={pos_mean:.4f} max_neg_sim={max_neg_mean:.4f} margin={margin_mean:.4f} loss={loss.item():.4f}")

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
