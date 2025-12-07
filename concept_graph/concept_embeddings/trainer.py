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
import hashlib
import gc

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
        self.stage_b_steps = int(self.cfg.optimization_steps)  # Stage B 和 Stage A 相同 epochs
        self._train_loaders: Dict[str, DataLoader] = {}
        self._val_loaders: Dict[str, DataLoader] = {}
        self._build_stage_loaders()
        self._injected_layer = None

    def train(self):
        checkpoints: Dict[int, Dict[str, torch.Tensor]] = {}
        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "training", True)
        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "mode", MyVLMLayerMode.TRAIN)

        # 初始化 wandb（如果可用且配置启用）
        use_wandb = WANDB_AVAILABLE and getattr(self.cfg, "use_wandb", False)
        if use_wandb:
            wandb_project = getattr(self.cfg, "wandb_project", "myvlm-art-concept-embedding")
            wandb_run_name = getattr(self.cfg, "wandb_run_name", f"concept_{self.cfg.concept_name}_seed_{self.cfg.seed}")
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "concept_name": self.cfg.concept_name,
                    "seed": self.cfg.seed,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                    "stage_a_steps": self.stage_a_steps,
                    "stage_b_steps": self.stage_b_steps,
                    "reg_lambda": getattr(self.cfg, "reg_lambda", 0.0),
                    "threshold": self.cfg.threshold,
                },
            )
            tqdm.write(f"[wandb] Initialized: project={wandb_project}, run={wandb_run_name}")

        optimizer, scheduler = None, None
        global_step = 0  # 全局步数计数器

        pbar_a = tqdm(total=self.stage_a_steps, desc="Stage A")
        for i in range(self.stage_a_steps):
            setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "iter", i)
            dl_a = self._train_loaders["A"]
            pbar_batches_a = tqdm(dl_a, total=len(dl_a), leave=False, desc=f"Train A {i+1}/{self.stage_a_steps}")
            for batch_idx, batch in enumerate(pbar_batches_a):
                if int(getattr(self.cfg, "max_train_batches", 0)) > 0 and batch_idx >= int(getattr(self.cfg, "max_train_batches", 0)):
                    break
                attn_interval = int(getattr(self.cfg, "attn_reg_interval", 1))
                batch["output_attentions"] = bool(getattr(self.cfg, "reg_lambda", 0.0) > 0) and (batch_idx % attn_interval == 0)
                if optimizer is None:
                    optimizer = torch.optim.AdamW(self.myvlm.vlm.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, self.cfg.learning_rate)
                with torch.cuda.amp.autocast():
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
                do_step = ((batch_idx + 1) % int(getattr(self.cfg, "grad_accum_steps", 1)) == 0) or (batch_idx == len(dl_a) - 1)
                if do_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                pbar_batches_a.set_description(f"Train A {i+1}/{self.stage_a_steps} | Loss: {float(loss):0.3f} | Reg: {float(reg_loss):0.3f}")
                # wandb 日志记录
                if use_wandb:
                    wandb.log({
                        "stage": "A",
                        "epoch": i + 1,
                        "batch": batch_idx,
                        "global_step": global_step,
                        "train/loss": float(loss),
                        "train/reg_loss": float(reg_loss),
                        "train/lm_loss": float(outputs.loss),
                        "lr": scheduler.get_last_lr()[0] if scheduler else self.cfg.learning_rate,
                    }, step=global_step)
                global_step += 1
                if self.myvlm._should_validate(i, batch_idx):
                    tqdm.write(f"Validating A step {i+1}")
                    self.myvlm.validate(self._val_loaders["A"], desc=f"Validation A {i+1}/{self.stage_a_steps}")
                if self.myvlm._should_save_checkpoint(i, batch_idx):
                    state = {}
                    if hasattr(layer_module, "keys") and layer_module.keys is not None:
                        state["keys"] = layer_module.keys.clone().detach().requires_grad_(False).cpu()
                    if hasattr(layer_module, "values") and layer_module.values is not None:
                        state["values"] = layer_module.values.clone().detach().requires_grad_(False).cpu()
                    if state:
                        checkpoints[i] = state
                        from datetime import datetime
                        from zoneinfo import ZoneInfo
                        ts = datetime.now(ZoneInfo('America/New_York')).strftime("%Y%m%d_%H%M%S")
                        path = self.cfg.output_path / f"checkpoints_{self.cfg.concept_name}_seed_{self.cfg.seed}_{ts}.pt"
                        torch.save(checkpoints, path)
                        pbar_batches_a.set_postfix(save="ok")
                        tqdm.write(f"Checkpoint saved: {path} (step {i})")
            pbar_batches_a.close()
            pbar_a.update(1)

        pbar_b = tqdm(total=self.stage_b_steps, desc="Stage B")
        # Stage B 开始前彻底清理显存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 诊断：检查 Stage B 数据集
        if use_wandb:
            sample_b = self._train_loaders["B"].dataset.samples[0] if len(self._train_loaders["B"].dataset.samples) > 0 else {}
            tqdm.write(f"[Stage B] 数据集大小: {len(self._train_loaders['B'].dataset)}")
            tqdm.write(f"[Stage B] 示例 target_keys: {sample_b.get('labels_per_dim', {})}")
        for j in range(self.stage_b_steps):
            step = self.stage_a_steps + j
            setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "iter", step)
            dl_b = self._train_loaders["B"]
            pbar_batches_b = tqdm(dl_b, total=len(dl_b), leave=False, desc=f"Train B {j+1}/{self.stage_b_steps}")
            for batch_idx, batch in enumerate(pbar_batches_b):
                if int(getattr(self.cfg, "max_train_batches", 0)) > 0 and batch_idx >= int(getattr(self.cfg, "max_train_batches", 0)):
                    break
                attn_interval = int(getattr(self.cfg, "attn_reg_interval", 1))
                batch["output_attentions"] = bool(getattr(self.cfg, "reg_lambda", 0.0) > 0) and (batch_idx % attn_interval == 0)
                # Stage B: 移除 token_weights，模型不支持此参数
                token_weights = batch.pop("token_weights", None)
                labels = batch.get("labels", None)  # 保存 labels 用于加权 loss
                with torch.cuda.amp.autocast():
                    outputs = self.myvlm.vlm.model(**batch)
                
                # 使用 token_weights 计算加权 loss（显存优化版）
                if token_weights is not None and labels is not None:
                    # shift: 预测 token[i+1] 基于 token[0:i]
                    shift_labels = labels[..., 1:].contiguous()
                    shift_weights = token_weights[..., 1:].contiguous()
                    vocab_size = outputs.logits.size(-1)
                    
                    # 直接在原始 logits 上操作，避免创建完整副本
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    per_token_loss = loss_fct(
                        outputs.logits[..., :-1, :].reshape(-1, vocab_size),
                        shift_labels.reshape(-1)
                    )
                    
                    # 应用权重
                    weights_flat = shift_weights.reshape(-1)
                    valid_mask = (shift_labels.reshape(-1) != -100).float()
                    weight_sum = (weights_flat * valid_mask).sum()
                    
                    if weight_sum > 0:
                        loss = (per_token_loss * weights_flat).sum() / weight_sum
                    else:
                        loss = outputs.loss
                    
                    # 立即释放中间变量
                    del per_token_loss, weights_flat, valid_mask, shift_labels, shift_weights
                else:
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
                do_step = ((batch_idx + 1) % int(getattr(self.cfg, "grad_accum_steps", 1)) == 0) or (batch_idx == len(dl_b) - 1)
                if do_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                pbar_batches_b.set_description(f"Train B {j+1}/{self.stage_b_steps} | Loss: {float(loss):0.3f} | Reg: {float(reg_loss):0.3f}")
                # wandb 日志记录
                if use_wandb:
                    # 原始 loss（未加权）vs 加权 loss
                    raw_loss = outputs.loss.item() if hasattr(outputs.loss, 'item') else float(outputs.loss)
                    weighted_loss_val = float(loss) - float(reg_loss)  # 减去 reg_loss 得到纯 lm loss
                    wandb.log({
                        "stage": "B",
                        "epoch": j + 1,
                        "batch": batch_idx,
                        "global_step": global_step,
                        "train/loss": float(loss),              # 总 loss（加权 + reg）
                        "train/weighted_lm_loss": weighted_loss_val,  # 加权语言建模 loss
                        "train/raw_lm_loss": raw_loss,          # 原始未加权 loss
                        "train/reg_loss": float(reg_loss),
                        "lr": scheduler.get_last_lr()[0] if scheduler else self.cfg.learning_rate,
                    }, step=global_step)
                global_step += 1
                if self.myvlm._should_validate(step, batch_idx):
                    tqdm.write(f"Validating B step {step}")
                    self.myvlm.validate(self._val_loaders["B"], desc=f"Validation B {j+1}/{self.stage_b_steps}")
                if self.myvlm._should_save_checkpoint(step, batch_idx):
                    state = {}
                    if hasattr(layer_module, "keys") and layer_module.keys is not None:
                        state["keys"] = layer_module.keys.clone().detach().requires_grad_(False).cpu()
                    if hasattr(layer_module, "values") and layer_module.values is not None:
                        state["values"] = layer_module.values.clone().detach().requires_grad_(False).cpu()
                    if state:
                        checkpoints[step] = state
                        from datetime import datetime
                        from zoneinfo import ZoneInfo
                        ts = datetime.now(ZoneInfo('America/New_York')).strftime("%Y%m%d_%H%M%S")
                        path = self.cfg.output_path / f"checkpoints_{self.cfg.concept_name}_seed_{self.cfg.seed}_{ts}.pt"
                        torch.save(checkpoints, path)
                        pbar_batches_b.set_postfix(save="ok")
                        tqdm.write(f"Checkpoint saved: {path} (step {step})")
            pbar_batches_b.close()
            pbar_b.update(1)

        setattr(eval(f"self.myvlm.vlm.{self.myvlm.layer}"), "mode", MyVLMLayerMode.INFERENCE)
        
        # 关闭 wandb
        if use_wandb:
            wandb.finish()
            tqdm.write("[wandb] Run finished.")
        
        return checkpoints

    def _build_stage_loaders(self) -> None:
        # Stage-specific structured configs for prompts
        self._structured_cfg_a: Dict[str, Any] = {}
        self._structured_cfg_b: Dict[str, Any] = {
            "variants": [
                "Identify the artist, style, and genre. First give three lines 'artist: ...', 'style: ...', 'genre: ...'. Then write 2-3 short sentences explaining the attribution.",
                "Return three key-value lines for artist, style, genre, followed by 2-3 brief sentences of visual reasoning about brushwork, color, and composition.",
                "Give the keys (artist/style/genre) as three lines, then add 2-3 concise sentences describing evidence for the attribution and style.",
            ],
            "weights": [0.5, 0.3, 0.2],
            "seed": int(getattr(self.cfg, "seed", 0)),
            "max_reason_tokens": int(getattr(self.cfg, "max_reason_tokens", 64)),
        }
        ds_a = LLaVAConceptGraphDataset.from_csv(
            csv_path=self.targets_csv,
            images_root=self.images_root,
            processor=self.processor,
            prompt_builder=self._prompt_builder_a,
            target_builder=self._target_builder,
            stage_mode="A",
            w_keys=1.0,
            w_reason=0.0,
            structured_cfg=self._structured_cfg_a,
            device=self.device,
            torch_dtype=self.myvlm.vlm.torch_dtype,
        )
        ds_b = LLaVAConceptGraphDataset.from_csv(
            csv_path=self.targets_csv,
            images_root=self.images_root,
            processor=self.processor,
            prompt_builder=self._prompt_builder_b,
            target_builder=self._target_builder,
            stage_mode="B",
            w_keys=1.0,
            w_reason=0.2,
            structured_cfg=self._structured_cfg_b,
            device=self.device,
            torch_dtype=self.myvlm.vlm.torch_dtype,
        )
        signals_map = self._load_concept_signals()
        for ds in [ds_a, ds_b]:
            for k in range(len(ds.samples)):
                p = ds.samples[k]["image_path"]
                if p in signals_map:
                    ds.samples[k]["concept_signals"] = signals_map[p]
        subset_n = int(getattr(self.cfg, "train_subset_n", 0))
        subset_stride = max(1, int(getattr(self.cfg, "train_subset_stride", 1)))
        if subset_n > 0:
            for ds in [ds_a, ds_b]:
                ds.samples = ds.samples[0:subset_n:subset_stride]
        self._ensure_multi_token_layer_init(signals_map)
        self._train_loaders["A"] = DataLoader(ds_a, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, collate_fn=ds_a.collate_fn)
        self._val_loaders["A"] = DataLoader(ds_a, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds_a.collate_fn)
        self._train_loaders["B"] = DataLoader(ds_b, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, collate_fn=ds_b.collate_fn)
        self._val_loaders["B"] = DataLoader(ds_b, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds_b.collate_fn)

    def _prompt_builder_a(self, labels: Dict[str, str], concept_signals: Any, mode: str = "train_semi_structured", structured_cfg: Optional[Dict[str, Any]] = None) -> str:
        return "You are an art expert. Output exactly three lines: 'artist: ...', 'style: ...', 'genre: ...'. No additional text. If uncertain, use 'Unknown'."

    def _prompt_builder_b(self, labels: Dict[str, str], concept_signals: Any, mode: str = "train_semi_structured", structured_cfg: Optional[Dict[str, Any]] = None) -> str:
        variants = (structured_cfg or {}).get("variants", [])
        weights = (structured_cfg or {}).get("weights", [])
        seed = int((structured_cfg or {}).get("seed", 0))
        if not variants or not weights or len(variants) != len(weights):
            return "Identify the artist, style, and genre by first giving three lines 'artist: ...', 'style: ...', 'genre: ...', then provide 2-3 short sentences of reasoning."
        total = sum(float(w) for w in weights)
        if total <= 0:
            return variants[0]
        cum = []
        s = 0.0
        for w in weights:
            s += float(w) / total
            cum.append(s)
        key = f"{labels.get('artist','')}|{labels.get('style','')}|{labels.get('genre','')}|{seed}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        u = (h % 10**8) / float(10**8)
        idx = 0
        for i, c in enumerate(cum):
            if u <= c:
                idx = i
                break
        return variants[idx]

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
        signals_artist, signals_style, signals_genre = [], [], []
        if self.signals_jsons["artist"].exists():
            with self.signals_jsons["artist"].open("r") as f:
                signals_artist = json.load(f)
        if self.signals_jsons["style"].exists():
            with self.signals_jsons["style"].open("r") as f:
                signals_style = json.load(f)
        if self.signals_jsons["genre"].exists():
            with self.signals_jsons["genre"].open("r") as f:
                signals_genre = json.load(f)
        def _count_dim(items: List[Dict[str, Any]], key: str) -> int:
            n = 0
            for rec in items:
                sig = rec.get(key, {})
                for k in sig.keys():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if idx + 1 > n:
                        n = idx + 1
            return n
        n_artist = _count_dim(signals_artist, "artist")
        n_style = _count_dim(signals_style, "style")
        n_genre = _count_dim(signals_genre, "genre")
        offset_artist = 0
        offset_style = n_artist
        offset_genre = n_artist + n_style
        # store dimension ranges for downstream layer
        self._dim_ranges = {
            "artist": (offset_artist, offset_style),
            "style": (offset_style, offset_genre),
            "genre": (offset_genre, offset_genre + n_genre),
        }
        mapping: Dict[str, Dict[int, torch.Tensor]] = {}
        for i, abs_path in enumerate(img_list):
            out: Dict[int, torch.Tensor] = {}
            if i < len(signals_artist) and "artist" in signals_artist[i]:
                sig = signals_artist[i]["artist"]
                for k_str, vec in sig.items():
                    try:
                        k = int(k_str) + offset_artist
                    except Exception:
                        continue
                    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                    out[k] = t
            if i < len(signals_style) and "style" in signals_style[i]:
                sig = signals_style[i]["style"]
                for k_str, vec in sig.items():
                    try:
                        k = int(k_str) + offset_style
                    except Exception:
                        continue
                    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                    out[k] = t
            if i < len(signals_genre) and "genre" in signals_genre[i]:
                sig = signals_genre[i]["genre"]
                for k_str, vec in sig.items():
                    try:
                        k = int(k_str) + offset_genre
                    except Exception:
                        continue
                    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                    out[k] = t
            if len(out) > 0:
                mapping[abs_path] = out
        return mapping

    def _ensure_multi_token_layer_init(self, signals_map: Dict[str, Dict[int, torch.Tensor]]) -> None:
        edit_module = parent_module(self.myvlm.vlm, brackets_to_periods(self.myvlm.layer))
        layer_name = self.myvlm.layer.rsplit(".", 1)[-1]
        original = getattr(edit_module, layer_name)
        # If original is MyVLMLayer wrapper, unwrap to the base layer for multi-token injection
        base_layer = getattr(original, "layer", original)
        if isinstance(original, MultiTokenConceptLayer):
            layer = original
        else:
            layer = MultiTokenConceptLayer(
                layer=base_layer,
                embedding_dim=VLM_TO_EMBEDDING_DIM[self.myvlm.cfg.vlm_type],
                max_tokens_per_concept=getattr(self.cfg, "max_tokens_per_concept", 4),
                threshold=self.cfg.threshold,
                torch_dtype=self.myvlm.vlm.torch_dtype,
                device=self.device,
                max_concepts_per_sample=getattr(self.cfg, "max_concepts_per_sample", 0),
                backoff_delta=getattr(self.cfg, "backoff_delta", 0.0),
                topk_per_dim=getattr(self.cfg, "topk_per_dim", 0),
                fairness=getattr(self.cfg, "fairness", False),
                priority=[s.strip() for s in getattr(self.cfg, "priority", "artist,style,genre").split(",") if s.strip()],
            )
            setattr(edit_module, layer_name, layer)
        concept_idxs = set()
        for v in signals_map.values():
            for k in v.keys():
                concept_idxs.add(k)
        n_concepts = max(concept_idxs) + 1 if len(concept_idxs) > 0 else 1
        if getattr(layer, "values", None) is None:
            layer.initialize_values(n_concepts)
        # set dimension ranges for fairness merge
        if hasattr(self, "_dim_ranges") and isinstance(self._dim_ranges, dict):
            layer.set_dim_ranges(self._dim_ranges)
        self._injected_layer = layer
