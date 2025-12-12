import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Union, List, Dict, Tuple
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
        topk_per_dim: int = 0,
        fairness: bool = False,
        priority: Optional[List[str]] = None,
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
        self.topk_per_dim = int(max(0, topk_per_dim))
        self.fairness = bool(fairness)
        self.priority = [p.strip() for p in (priority or []) if p.strip()]
        self.keys = None
        self.values = None
        self.key_idx_to_value_idx: Dict[int, int] = {}
        self.training = True
        # dimension ranges: {dim: (start_idx, end_idx)} end exclusive
        self.dim_ranges: Dict[str, Tuple[int, int]] = {}

    def initialize_values(self, n_concepts: int) -> None:
        values = torch.randn(n_concepts, self.max_tokens_per_concept, self.embedding_dim, device=self.device)
        values = values / values.norm(dim=-1, keepdim=True)
        self.values = nn.Parameter(values.to(dtype=self.torch_dtype), requires_grad=True)

    def set_keys(self, keys: torch.Tensor) -> None:
        self.keys = keys.to(self.device)

    def set_key_to_value_mapping(self, mapping: Dict[int, int]) -> None:
        self.key_idx_to_value_idx = dict(mapping)

    def set_dim_ranges(self, ranges: Dict[str, Tuple[int, int]]) -> None:
        self.dim_ranges = dict(ranges)

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
                # group by dimension ranges
                per_dim_sel: Dict[str, List[Tuple[int, float]]] = {}
                dims = list(self.dim_ranges.keys())
                for dim in dims:
                    start, end = self.dim_ranges[dim]
                    activated: List[Tuple[int, float]] = []
                    # threshold selection
                    for concept_idx, probas in sample_sig.items():
                        if concept_idx < start or concept_idx >= end:
                            continue
                        s = float(probas[0][1].item()) if hasattr(probas[0][1], "item") else float(probas[0][1])
                        if s >= self.threshold:
                            activated.append((concept_idx, s))
                    activated.sort(key=lambda t: t[1], reverse=True)
                    # backoff per dim if none
                    if len(activated) == 0 and self.backoff_delta > 0.0:
                        candidates: List[Tuple[int, float]] = []
                        for concept_idx, probas in sample_sig.items():
                            if concept_idx < start or concept_idx >= end:
                                continue
                            s = float(probas[0][1].item()) if hasattr(probas[0][1], "item") else float(probas[0][1])
                            if s >= max(0.0, self.threshold - self.backoff_delta):
                                candidates.append((concept_idx, s))
                        candidates.sort(key=lambda t: t[1], reverse=True)
                        if len(candidates) > 0:
                            activated = [candidates[0]]
                    # topk per dim
                    if self.topk_per_dim > 0 and len(activated) > self.topk_per_dim:
                        activated = activated[:self.topk_per_dim]
                    per_dim_sel[dim] = activated
                # fairness merge with global budget
                selected: List[Tuple[str, int, float]] = []
                budget = self.max_concepts_per_sample
                if budget <= 0:
                    # no budget: flatten all
                    for dim in dims:
                        for idx, s in per_dim_sel.get(dim, []):
                            selected.append((dim, idx, s))
                else:
                    # fairness: keep one per dim in priority order
                    used_total = 0
                    prio = self.priority if self.priority else dims
                    out_per_dim: Dict[str, List[Tuple[int, float]]] = {d: [] for d in dims}
                    for d in prio:
                        opts = per_dim_sel.get(d, [])
                        if len(opts) > 0 and used_total < budget:
                            out_per_dim[d].append(opts[0])
                            used_total += 1
                    remaining: List[Tuple[str, int, float]] = []
                    for d in dims:
                        start_i = len(out_per_dim[d])
                        for i in range(start_i, len(per_dim_sel.get(d, []))):
                            idx, s = per_dim_sel[d][i]
                            remaining.append((d, idx, s))
                    remaining.sort(key=lambda t: t[2], reverse=True)
                    for d, idx, s in remaining:
                        if used_total >= budget:
                            break
                        out_per_dim[d].append((idx, s))
                        used_total += 1
                    for d in dims:
                        for idx, s in out_per_dim[d]:
                            selected.append((d, idx, s))
                # append tokens for selected
                used = set()
                for _, concept_idx, s in selected:
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
                    if s_i >= self.threshold:
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
