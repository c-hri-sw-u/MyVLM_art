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
  - 注意力正则：为新追加的概念 token 加入 attention 正则钩子（训练期）。可利用 `concept_token_idxs`（vlms/llava/model/language_model/llava_llama.py:45）收集位置并计算均衡约束。
  - 信号归一化/对比度调节：在概念原型相似度导出阶段做统一归一化与温度缩放，减少跨维度分布差异。
    - 现有：`export_signals.py` 支持 `--normalize=zero_one|clamp_zero|none`（concept_graph/prototypes/export_signals.py:86–100）。
    - 待扩展：加入 `zscore` 与 `temperature=T` 两种模式，并在数据集加载端保持一致（concept_graph/concept_embeddings/trainer.py:207–232）。
  - 两层筛选机制（概念级 + 概念内）：
    - 概念是否激活（已有）：阈值筛选，`dist <= τ` 视为激活（multi_embed_layer.py:87–101, 104–120）。
    - 概念内 token gating（已有）：按激活强度 `g` 映射 token 数 `k = ceil(g · K_max)`，并确保 `k ≥ 1`。
    - 无激活回退：若某维度无激活且存在分数≥`τ − δ`的候选，仅回退该维度 `Top‑1`；否则跳过该维度，避免误注入。
    - 有激活时 Top‑K：对已激活候选按分数降序截断至 `Top‑K_per_dim`，同时受全局预算约束。
  - 全局预算与公平分配：
    - `max_concepts_per_sample` 控制单样本总追加概念数（如 3）；维度优先策略建议为 `artist≥1, style≥1, genre≥1`，其余按分数与剩余预算分配。
  - 生命周期整合：统一 `keys/values/mapping` 的初始化、持久化与推理时加载流程，确保与 MyVLM 层生命周期一致。
  - 配置暴露：在 `configs/train_config.py` 增加阈值 `τ`、容差 `δ`、`Top‑K_per_dim`、`max_tokens_per_concept`、`max_concepts_per_sample`、归一化与温度参数，使训练/推理可控。
  - 验证与度量：记录每维度激活准确率、误注入率、预算命中率；分析失败样例并回溯阈值/温度设置。

实现指引 (Instructions):
  1) 扩展信号导出：在 `concept_graph/prototypes/export_signals.py` 增加 `--normalize=zscore` 与 `--temperature=T`，并保存归一化后的分数。
  2) 配置项补充：在 `configs/train_config.py` 增加 gating/预算相关参数，并在训练器/推理端传递给本层。
  3) 两层筛选实现：在本层 `forward()` 中按如下顺序执行：
     - 阈值筛选→每维 `Top‑K_per_dim`→全局预算裁剪→概念内 token gating（`g→k`）。
  4) 注意力正则：训练时通过 `concept_token_idxs` 标注追加 token 的位置，计算均衡正则并纳入总损失。
  5) 验证流程：统计并可视化各维度的激活/回退/裁剪比例，调参 `τ, δ, K_per_dim, max_concepts_per_sample, T` 以稳定性能。
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
        max_concepts_per_sample: int = 0,
        backoff_delta: float = 0.0,
    ):
        super().__init__()
        self.layer = layer
        self.embedding_dim = embedding_dim
        self.max_tokens_per_concept = max_tokens_per_concept
        self.threshold = threshold
        self.torch_dtype = torch_dtype
        self.device = device
        self.max_concepts_per_sample = int(max(0, max_concepts_per_sample))
        self.backoff_delta = float(max(0.0, backoff_delta))
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
                candidates = []
                for concept_idx, probas in sample_sig.items():
                    s = float(probas[0][1].item()) if hasattr(probas[0][1], "item") else float(probas[0][1])
                    dist = 1.0 - s
                    if dist <= self.threshold:
                        candidates.append((concept_idx, s))
                if len(candidates) == 0 and self.backoff_delta > 0.0:
                    for concept_idx, probas in sample_sig.items():
                        s = float(probas[0][1].item()) if hasattr(probas[0][1], "item") else float(probas[0][1])
                        if s >= max(0.0, self.threshold - self.backoff_delta):
                            candidates.append((concept_idx, s))
                    candidates.sort(key=lambda t: t[1], reverse=True)
                    if len(candidates) > 1:
                        candidates = candidates[:1]
                else:
                    candidates.sort(key=lambda t: t[1], reverse=True)
                if self.max_concepts_per_sample > 0 and len(candidates) > self.max_concepts_per_sample:
                    candidates = candidates[:self.max_concepts_per_sample]
                used = set()
                for concept_idx, s in candidates:
                    if concept_idx in used:
                        continue
                    g = max(0.0, min(1.0, s))
                    k = max(1, int(torch.ceil(torch.tensor(g) * self.max_tokens_per_concept).item()))
                    tokens = self.values[concept_idx][:k]
                    tokens = F.normalize(tokens, dim=-1, p=2).to(dtype=layer_out.dtype, device=layer_out.device)
                    sample_out = torch.vstack([sample_out, tokens])
                    used.add(concept_idx)
                extended.append(sample_out)
            return torch.stack(extended, dim=0).to(dtype=self.torch_dtype)

        if isinstance(concept_signal, torch.Tensor):
            extended = []
            for sample_idx, q in enumerate(concept_signal):
                dists = self._compute_distances(concept_signal=concept_signal, query=q.to(self.device))
                smallest_dist, chosen_key = dists.min(0)
                sample_out = layer_out[sample_idx]
                pairs = []
                for i in range(chosen_key.shape[0]):
                    concept_idx = self.key_idx_to_value_idx.get(chosen_key[i].item(), None)
                    if concept_idx is None:
                        continue
                    dist_i = float(smallest_dist[i].item()) if hasattr(smallest_dist[i], "item") else float(smallest_dist[i])
                    s_i = 1.0 - dist_i
                    if dist_i <= self.threshold:
                        pairs.append((concept_idx, s_i))
                if len(pairs) == 0 and self.backoff_delta > 0.0:
                    ds = []
                    for i in range(chosen_key.shape[0]):
                        concept_idx = self.key_idx_to_value_idx.get(chosen_key[i].item(), None)
                        if concept_idx is None:
                            continue
                        dist_i = float(smallest_dist[i].item()) if hasattr(smallest_dist[i], "item") else float(smallest_dist[i])
                        s_i = 1.0 - dist_i
                        if s_i >= max(0.0, self.threshold - self.backoff_delta):
                            ds.append((concept_idx, s_i))
                    ds.sort(key=lambda t: t[1], reverse=True)
                    if len(ds) > 1:
                        ds = ds[:1]
                    pairs = ds
                else:
                    pairs.sort(key=lambda t: t[1], reverse=True)
                if self.max_concepts_per_sample > 0 and len(pairs) > self.max_concepts_per_sample:
                    pairs = pairs[:self.max_concepts_per_sample]
                used = set()
                for concept_idx, s_i in pairs:
                    if concept_idx in used:
                        continue
                    g = max(0.0, min(1.0, s_i))
                    k = max(1, int(torch.ceil(torch.tensor(g) * self.max_tokens_per_concept).item()))
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
