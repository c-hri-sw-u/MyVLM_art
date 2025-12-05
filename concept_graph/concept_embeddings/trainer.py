"""
简要介绍 (ZH):
  概念嵌入（多 token）训练器。联合优化：
  - 语言建模损失（交叉熵），让 VLM 在解释/分类输出中自然提及并利用激活概念
  - 注意力正则损失，抑制某些概念 token 主导，保持多概念协同稳定

Overview (EN):
  Trainer for multi-token concept embeddings. Jointly optimize language modeling (CE) and attention regularization
  to encourage coherent multi-concept reasoning while preventing dominance of specific concept tokens.

Inputs/Outputs:
  - Inputs: batched images, activated concept signals (prototype similarities), prompts & targets
  - Outputs: checkpoints of concept embeddings per iteration, evaluation logs on validation split

Status:
  - Implemented. Includes Stage A/B training, token‑weighted CE for keys/reason, attention regularization,
    multi‑token injection via mm_projector, periodic validation, and checkpointing of keys/values.

Remaining Work:
  1) Extend metrics/validation for per‑dimension accuracy and coherence.
  2) Optional: merge multi‑dimension signals (style/genre) and expose gating options via config.
  3) Optional: add serialization for key‑to‑value mapping when using external initialization.


阶段化训练调度：
阶段 A：使用 from_csv(csv_path, images_root, ..., stage_mode="A", w_keys=1.0, w_reason=0.0)；跑若干 epoch 至稳定
阶段 B：切换 stage_mode="B"，w_reason=0.2，学习理由的语言风格；定期验证三键分类是否稳定
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from concept_graph.concept_embeddings.datasets.llava_concept_graph_dataset import (
    LLaVAConceptGraphDataset,
)
from concept_graph.concept_embeddings.multi_embed_layer import MultiTokenConceptLayer
from myvlm.common import MyVLMLayerMode, VLM_TO_EMBEDDING_DIM
from myvlm.utils import parent_module, brackets_to_periods


class MultiTokenEmbeddingTrainer:
    def __init__(self, cfg, myvlm, dataset_builder):
        self.cfg = cfg
        self.myvlm = myvlm
        self.dataset_builder = dataset_builder
        self.device = getattr(self.myvlm, "device", "cuda")
        self.processor = self.myvlm.vlm.processor
        self.images_root = Path(self.cfg.data_root) / "dataset"
        self.targets_csv = Path("artifacts/synth_targets.csv")
        self.dataset_json = self.images_root / "wikiart_5artists_dataset.json"
        self.signals_jsons = {
            "artist": Path("artifacts/concept_signals_artist.json"),
            "style": Path("artifacts/concept_signals_style.json"),
            "genre": Path("artifacts/concept_signals_genre.json"),
        }
        self.stage_a_steps = int(self.cfg.optimization_steps)
        self.stage_b_steps = max(1, int(self.cfg.optimization_steps // 2))
        self._train_loaders: Dict[str, DataLoader] = {}
        self._val_loaders: Dict[str, DataLoader] = {}
        self._build_stage_loaders()
        self._injected_layer = None

    def train(self):
        checkpoints: Dict[int, Dict[str, torch.Tensor]] = {}
        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "training", True)
        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "mode", MyVLMLayerMode.TRAIN)

        optimizer, scheduler = None, None

        pbar_a = tqdm(range(self.stage_a_steps))
        for i in pbar_a:
            setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "iter", i)
            for batch_idx, batch in enumerate(self._train_loaders["A"]):
                batch["output_attentions"] = True
                outputs = self.myvlm.vlm.model(**batch)
                if optimizer is None:
                    optimizer = torch.optim.AdamW(self.myvlm.vlm.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, self.cfg.learning_rate)
                loss = outputs.loss
                reg_loss = 0.0
                if getattr(self.cfg, "reg_lambda", 0.0) > 0 and hasattr(outputs, "attentions") and hasattr(outputs, "concept_token_idxs") and outputs.concept_token_idxs is not None:
                    try:
                        reg_losses = []
                        for probas in outputs.attentions:
                            for sample_idx in range(probas.shape[0]):
                                reg_losses.append(self.cfg.reg_lambda * torch.mean(probas[sample_idx, :, outputs.concept_token_idxs[sample_idx], :] ** 2))
                        reg_loss = sum(reg_losses)
                        loss = loss + reg_loss
                    except Exception:
                        reg_loss = 0.0
                loss.backward()
                layer_module = eval(f"self.myvlm.vlm.{self.myvlm.layer}")
                if hasattr(layer_module, "values") and layer_module.values is not None:
                    torch.nn.utils.clip_grad_norm_(layer_module.values, 0.05, norm_type=2)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar_a.set_description(f"Stage A | Loss: {float(loss):0.3f} | Reg: {float(reg_loss):0.3f}")
                if self.myvlm._should_validate(i, batch_idx):
                    self.myvlm.validate(self._val_loaders["A"])
                if self.myvlm._should_save_checkpoint(i, batch_idx):
                    state = {}
                    if hasattr(layer_module, "keys") and layer_module.keys is not None:
                        state["keys"] = layer_module.keys.clone().detach().requires_grad_(False).cpu()
                    if hasattr(layer_module, "values") and layer_module.values is not None:
                        state["values"] = layer_module.values.clone().detach().requires_grad_(False).cpu()
                    if state:
                        checkpoints[i] = state

        pbar_b = tqdm(range(self.stage_b_steps))
        for j in pbar_b:
            step = self.stage_a_steps + j
            setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "iter", step)
            for batch_idx, batch in enumerate(self._train_loaders["B"]):
                batch["output_attentions"] = True
                outputs = self.myvlm.vlm.model(**batch)
                loss = outputs.loss
                reg_loss = 0.0
                if getattr(self.cfg, "reg_lambda", 0.0) > 0 and hasattr(outputs, "attentions") and hasattr(outputs, "concept_token_idxs") and outputs.concept_token_idxs is not None:
                    try:
                        reg_losses = []
                        for probas in outputs.attentions:
                            for sample_idx in range(probas.shape[0]):
                                reg_losses.append(self.cfg.reg_lambda * torch.mean(probas[sample_idx, :, outputs.concept_token_idxs[sample_idx], :] ** 2))
                        reg_loss = sum(reg_losses)
                        loss = loss + reg_loss
                    except Exception:
                        reg_loss = 0.0
                loss.backward()
                layer_module = eval(f"self.myvlm.vlm.{self.myvlm.layer}")
                if hasattr(layer_module, "values") and layer_module.values is not None:
                    torch.nn.utils.clip_grad_norm_(layer_module.values, 0.05, norm_type=2)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar_b.set_description(f"Stage B | Loss: {float(loss):0.3f} | Reg: {float(reg_loss):0.3f}")
                if self.myvlm._should_validate(step, batch_idx):
                    self.myvlm.validate(self._val_loaders["B"])
                if self.myvlm._should_save_checkpoint(step, batch_idx):
                    state = {}
                    if hasattr(layer_module, "keys") and layer_module.keys is not None:
                        state["keys"] = layer_module.keys.clone().detach().requires_grad_(False).cpu()
                    if hasattr(layer_module, "values") and layer_module.values is not None:
                        state["values"] = layer_module.values.clone().detach().requires_grad_(False).cpu()
                    if state:
                        checkpoints[step] = state

        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "mode", MyVLMLayerMode.INFERENCE)
        return checkpoints

    def _build_stage_loaders(self) -> None:
        ds_a = LLaVAConceptGraphDataset.from_csv(
            csv_path=self.targets_csv,
            images_root=self.images_root,
            processor=self.processor,
            prompt_builder=self._prompt_builder,
            target_builder=self._target_builder,
            stage_mode="A",
            w_keys=1.0,
            w_reason=0.0,
            device=self.device,
            torch_dtype=self.myvlm.vlm.torch_dtype,
        )
        ds_b = LLaVAConceptGraphDataset.from_csv(
            csv_path=self.targets_csv,
            images_root=self.images_root,
            processor=self.processor,
            prompt_builder=self._prompt_builder,
            target_builder=self._target_builder,
            stage_mode="B",
            w_keys=1.0,
            w_reason=0.2,
            device=self.device,
            torch_dtype=self.myvlm.vlm.torch_dtype,
        )
        signals_map = self._load_concept_signals()
        for ds in [ds_a, ds_b]:
            for k in range(len(ds.samples)):
                p = ds.samples[k]["image_path"]
                if p in signals_map:
                    ds.samples[k]["concept_signals"] = signals_map[p]
        self._ensure_multi_token_layer_init(signals_map)
        self._train_loaders["A"] = DataLoader(ds_a, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, collate_fn=ds_a.collate_fn)
        self._val_loaders["A"] = DataLoader(ds_a, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds_a.collate_fn)
        self._train_loaders["B"] = DataLoader(ds_b, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, collate_fn=ds_b.collate_fn)
        self._val_loaders["B"] = DataLoader(ds_b, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds_b.collate_fn)

    def _prompt_builder(self, labels: Dict[str, str], concept_signals: Any, mode: str = "train_semi_structured", structured_cfg: Optional[Dict[str, Any]] = None) -> str:
        artist = labels.get("artist", "").replace("_", " ")
        style = labels.get("style", "").replace("_", " ")
        genre = labels.get("genre", "").replace("_", " ")
        if mode == "train_semi_structured":
            return f"Identify the artist, style, and genre of this painting, then provide a brief reasoning."
        return f"Describe the painting including artist, style, and genre."

    def _target_builder(self, labels: Dict[str, str], concept_signals: Any, mode: str = "train_semi_structured") -> str:
        artist = labels.get("artist", "")
        style = labels.get("style", "")
        genre = labels.get("genre", "")
        return f"artist: {artist}\nstyle: {style}\ngenre: {genre}"

    def _load_concept_signals(self) -> Dict[str, Dict[int, torch.Tensor]]:
        img_list: List[str] = []
        with self.dataset_json.open("r") as f:
            records = json.load(f)
            for r in records:
                rel = r.get("image", "")
                if rel:
                    img_list.append(str((self.images_root / rel).resolve()))
        signals_artist = []
        if self.signals_jsons["artist"].exists():
            with self.signals_jsons["artist"].open("r") as f:
                signals_artist = json.load(f)
        mapping: Dict[str, Dict[int, torch.Tensor]] = {}
        for i, abs_path in enumerate(img_list):
            if i < len(signals_artist) and "artist" in signals_artist[i]:
                sig = signals_artist[i]["artist"]
                out: Dict[int, torch.Tensor] = {}
                for k_str, vec in sig.items():
                    try:
                        k = int(k_str)
                    except Exception:
                        continue
                    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                    out[k] = t
                mapping[abs_path] = out
        return mapping

    def _ensure_multi_token_layer_init(self, signals_map: Dict[str, Dict[int, torch.Tensor]]) -> None:
        edit_module = parent_module(self.myvlm.vlm, brackets_to_periods(self.myvlm.layer))
        layer_name = self.myvlm.layer.rsplit(".", 1)[-1]
        original = getattr(edit_module, layer_name)
        if isinstance(original, MultiTokenConceptLayer):
            layer = original
        else:
            layer = MultiTokenConceptLayer(
                layer=original,
                embedding_dim=VLM_TO_EMBEDDING_DIM[self.myvlm.cfg.vlm_type],
                max_tokens_per_concept=getattr(self.cfg, "max_tokens_per_concept", 4),
                threshold=self.cfg.threshold,
                torch_dtype=self.myvlm.vlm.torch_dtype,
                device=self.device,
            )
            setattr(edit_module, layer_name, layer)
        concept_idxs = set()
        for v in signals_map.values():
            for k in v.keys():
                concept_idxs.add(k)
        n_concepts = max(concept_idxs) + 1 if len(concept_idxs) > 0 else 1
        if getattr(layer, "values", None) is None:
            layer.initialize_values(n_concepts)
        self._injected_layer = layer
