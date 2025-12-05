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

Status:
  - Core gating and dynamic multi-token injection implemented.
  - Keys/values tensors and mapping are available via `initialize_values`, `set_keys`, and `set_key_to_value_mapping`.

Remaining Work:
  - Add attention regularization hook for newly appended concept tokens during training.
  - 归一化/对比度调节：在 export_signals.py 或 ConceptGraphDataset 注入前，先对每个维度的相似度做 z-score/温度缩放，使不同维度的分布更接近，然后再用统一阈值。
  - 两层筛选机制：
    - “是否追加某概念的嵌入 token”：
        - 阈值筛选：分数达标才视为激活（ concept_graph/concept_embeddings/multi_embed_layer.py:78–93, 95–116 ）
        - “概念内 token gating”：已激活时，按激活强度 g 映射出要追加的 token 数
    - 概念级筛选：
        - 如果某个dimension(如genre), 没有可激活的token, 则追加该dimension最高的Top1 的概念
        - 如果某个dimension(如genre), 有很可激活的token, 则筛选出该dimension最高的Top2 的概念
  - Consolidate key/value initialization and persistence with the upstream MyVLM layer lifecycle.
  - Object/single/multi-person branches are not required for this dataset; keep as optional extension if needed.
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