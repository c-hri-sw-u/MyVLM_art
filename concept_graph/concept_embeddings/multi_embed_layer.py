"""
简要介绍 (ZH):
  多 token 概念嵌入注入层。为每个概念学习最多 K 个嵌入 token，并按原型相似度/激活强度动态选择
  实际追加的 token 数，将其拼接到目标 VLM 层输出，实现变长序列的可解释注入。

Overview (EN):
  Multi-token concept embedding injection layer. Each concept owns up to K trainable tokens; at runtime we gate
  how many tokens to append based on prototype similarity/activation strength and concatenate them into the target
  VLM layer output. Compatible with variable-length sequences and existing MyVLM flow.

Integration Points:
  - Layer replacement: myvlm/myvlm.py:34–43
  - Training loop & checkpointing: myvlm/myvlm.py:63–106
  - Inference-time embedding set: inference/inference_utils.py:17–28
  - Attention regularization references:
      * LLaVA: myvlm/myllava.py:49–58
      * MiniGPT‑v2: myvlm/myminigpt_v2.py:35–44

TODOs (详细):
  1) 实现类 MultiTokenConceptLayer，持有：
     - base_layer: 目标被包裹的原始层
     - embedding_dim: 视觉嵌入维度（如 LLaVA 为 4096）
     - max_tokens_per_concept: 每概念 token 上限（默认 4）
     - threshold/ gating: 基于相似度 s_c 的阈值与映射策略（例如 K=ceil(K_max * g_c)）
     - dtype/device: 训练与推理的精度/设备
  2) forward(hidden_state, concept_signal):
     - 计算概念距离/相似度，得到每概念的激活强度
     - 为每个被激活的概念选择前 K 个 token，并按对象/单人/多人分支将 token 逐一拼接到 layer_out
     - 返回拼接后的张量（保持与原始层 dtype 对齐）
  3) values/keys 形状：
     - keys: [n_keys, dim]
     - values: [n_concepts, n_tokens, dim]
     - key_idx_to_value_idx: 将多个 key 映射到同一概念 index（并据此选择其 token 列表）
  4) _compute_distances 与对象分支兼容：
     - 若 concept_signal 为原型相似度 s_c，映射为“伪概率” dict {concept_idx: Tensor([1-s_c, s_c])}
       以复用现有对象逻辑的距离定义（myvlm/myvlm_layer.py:244–255）
  5) 正则挂点：
     - 在训练模式下，为新追加的概念 token 的自注意力添加 L2 正则项，接口与现有实现保持一致
  6) 辅助：梯度裁剪、归一化、阈值打印（推理模式）
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union, List, Dict
from concept_embedding_training.data_utils import cosine_distance


class MultiTokenConceptLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        max_tokens_per_concept: int = 4,
        threshold: float = 0.7,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.layer = layer
        self.embedding_dim = embedding_dim
        self.max_tokens_per_concept = max_tokens_per_concept
        self.threshold = threshold
        self.torch_dtype = torch_dtype
        self.device = device
        self.keys = None
        self.values = None
        self.key_idx_to_value_idx: Dict[int, int] = {}
        self.training = True

    def initialize_values(self, n_concepts: int) -> None:
        values = torch.randn(n_concepts, self.max_tokens_per_concept, self.embedding_dim, device=self.device)
        values = values / values.norm(dim=-1, keepdim=True)
        self.values = nn.Parameter(values.to(dtype=self.torch_dtype), requires_grad=True)

    def set_keys(self, keys: torch.Tensor) -> None:
        self.keys = keys.to(self.device)

    def set_key_to_value_mapping(self, mapping: Dict[int, int]) -> None:
        self.key_idx_to_value_idx = dict(mapping)

    def forward(self, *args) -> torch.Tensor:
        hidden_state = args[0]
        concept_signal = args[1] if len(args) > 1 else None
        layer_out = self.layer(hidden_state)
        if concept_signal is None or self.values is None:
            return layer_out
        self.keys = self.keys.to(self.device) if self.keys is not None else self.keys

        if isinstance(concept_signal, list) and isinstance(concept_signal[0], dict):
            extended = []
            for sample_idx, sample_sig in enumerate(concept_signal):
                sample_out = layer_out[sample_idx]
                added = set()
                for concept_idx, probas in sample_sig.items():
                    dist = 1 - probas[0][1]
                    if dist <= self.threshold and concept_idx not in added:
                        g = torch.clamp(probas[0][1], 0.0, 1.0)
                        k = max(1, int(torch.ceil(g * self.max_tokens_per_concept).item()))
                        tokens = self.values[concept_idx][:k]
                        tokens = F.normalize(tokens, dim=-1, p=2).to(dtype=layer_out.dtype, device=layer_out.device)
                        sample_out = torch.vstack([sample_out, tokens])
                        added.add(concept_idx)
                extended.append(sample_out)
            return torch.stack(extended, dim=0).to(dtype=self.torch_dtype)

        if isinstance(concept_signal, torch.Tensor):
            extended = []
            for sample_idx, q in enumerate(concept_signal):
                dists = self._compute_distances(concept_signal=concept_signal, query=q.to(self.device))
                smallest_dist, chosen_key = dists.min(0)
                sample_out = layer_out[sample_idx]
                used = set()
                for i in range(chosen_key.shape[0]):
                    concept_idx = self.key_idx_to_value_idx.get(chosen_key[i].item(), None)
                    if concept_idx is None or concept_idx in used:
                        continue
                    if smallest_dist[i] <= self.threshold:
                        g = torch.clamp(1.0 - smallest_dist[i], 0.0, 1.0)
                        k = max(1, int(torch.ceil(g * self.max_tokens_per_concept).item()))
                        tokens = self.values[concept_idx][:k]
                        tokens = F.normalize(tokens, dim=-1, p=2).to(dtype=layer_out.dtype, device=layer_out.device)
                        sample_out = torch.vstack([sample_out, tokens])
                        used.add(concept_idx)
                extended.append(sample_out)
            if len(extended) > 0:
                return torch.stack(extended, dim=0).to(dtype=self.torch_dtype)
            return layer_out

        return layer_out

    def _compute_distances(self, concept_signal: Union[torch.Tensor, List], query: torch.Tensor) -> Union[torch.Tensor, List]:
        if isinstance(concept_signal, list) and isinstance(concept_signal[0], dict):
            dists = []
            for sample_probas in concept_signal:
                dists.append({k: 1 - v[0][1] for k, v in sample_probas.items()})
        else:
            dists = torch.stack([cosine_distance(query, key).view(-1, 1) for key in self.keys]).view(-1, len(query))
        return dists

    # TODO: 实现 keys/values 初始化与映射（复用 myvlm/myvlm_layer.py 的初始化流程语义）
    # TODO: 实现对象/多人/单人三种分支的拼接逻辑（按样本循环，避免重复添加同一概念）
    # TODO: 实现基于 s_c 的 gating 策略（线性/分段/Top‑K），并支持最小 token 数（如至少 1）
    # TODO: 实现追加 token 的归一化与 dtype 对齐，支持推理打印阈值与距离
