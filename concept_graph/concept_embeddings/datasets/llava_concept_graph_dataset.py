from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from vlms.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vlms.llava.conversation import conv_templates, SeparatorStyle
from vlms.llava.mm_utils import tokenizer_image_token


class LLaVAConceptGraphDataset(Dataset):
    def __init__(
        self,
        base_samples: List[Dict[str, Any]],
        processor: Any,
        prompt_builder: Any,
        target_builder: Any,
        template_mode: str = "train_semi_structured",
        precomputed_targets: Optional[Dict[str, str]] = None,
        precomputed_keys: Optional[Dict[str, str]] = None,
        precomputed_reasons: Optional[Dict[str, str]] = None,
        stage_mode: str = "A",
        w_keys: float = 1.0,
        w_reason: float = 0.2,
        structured_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.samples = base_samples
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.target_builder = target_builder
        self.template_mode = template_mode
        self.precomputed_targets = precomputed_targets or {}
        self.precomputed_keys = precomputed_keys or {}
        self.precomputed_reasons = precomputed_reasons or {}
        self.stage_mode = stage_mode
        self.w_keys = w_keys
        self.w_reason = w_reason
        self.structured_cfg = structured_cfg or {}
        self.device = device
        self.torch_dtype = torch_dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = s.get("image")
        if image is None:
            image = Image.open(s["image_path"]).convert("RGB")
        labels = s["labels_per_dim"]
        concept_signals = s.get("concept_signals", None)
        prompt = self.prompt_builder(labels, concept_signals, mode=self.template_mode, structured_cfg=self.structured_cfg)
        inputs = self.processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        inputs = inputs.to(self.device, self.torch_dtype)
        img_path = s["image_path"]
        keys = None
        reason = ""
        if img_path in self.precomputed_keys:
            keys = self.precomputed_keys[img_path]
        elif img_path in self.precomputed_targets:
            keys = self.precomputed_targets[img_path]
        else:
            keys = self.target_builder(labels, concept_signals, mode=self.template_mode)
        if self.stage_mode != "A":
            if img_path in self.precomputed_reasons:
                reason = self.precomputed_reasons[img_path] or ""
                max_r = int(self.structured_cfg.get("max_reason_tokens", 0))
                if max_r > 0 and len(reason) > 0:
                    try:
                        r_ids = tokenizer_image_token(reason, self.processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                        if r_ids.shape[0] > max_r:
                            r_ids = r_ids[:max_r]
                            reason = self.processor.tokenizer.decode(r_ids, skip_special_tokens=True)
                    except Exception:
                        pass
        target_text = keys if self.stage_mode == "A" else (keys + ("\n" + reason if reason else ""))
        batch = {
            "images": inputs,
            "prompt": prompt,
            "target_text": target_text,
            "target_keys": keys,
            "target_reason": reason,
            "w_keys": self.w_keys,
            "w_reason": self.w_reason,
            "concept_signals": concept_signals,
            "labels_per_dim": labels,
            "image_path": s["image_path"],
            "stage_mode": self.stage_mode,
        }
        return batch

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["images"] for b in batch], dim=0)
        concept_signals = [b.get("concept_signals") for b in batch]
        input_ids_list = []
        labels_list = []
        weights_list = []
        for b in batch:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + b["prompt"]
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_full = conv.get_prompt() + " " + b["target_text"]
            ids = tokenizer_image_token(
                prompt_full,
                self.processor.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            sep = conv.sep + conv.roles[1] + ": "
            parts = prompt_full.split(sep)
            parts[0] += sep
            instr_len = len(tokenizer_image_token(parts[0], self.processor.tokenizer)) - 2
            targets = ids.clone()
            targets[:, :1] = -100
            targets[:, 1:1 + instr_len] = -100
            input_ids_list.append(ids[0])
            labels_list.append(targets[0])
            if self.stage_mode == "B":
                keys_text = b.get("target_keys", "") or ""
                reason_text = b.get("target_reason", "") or ""
                k_ids = tokenizer_image_token(keys_text, self.processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                r_ids = tokenizer_image_token(reason_text, self.processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') if reason_text else torch.tensor([], dtype=torch.long)
                keys_len = int(k_ids.shape[0])
                reason_len = int(r_ids.shape[0])
                w = torch.zeros(ids.shape[1], dtype=torch.float32, device=self.device)
                start = 1 + instr_len
                end_k = start + keys_len
                end_r = end_k + reason_len
                w[start:end_k] = float(self.w_keys)
                if reason_len > 0:
                    w[end_k:end_r] = float(self.w_reason)
                weights_list.append(w)
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = input_ids.ne(pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        token_weights = None
        if self.stage_mode == "B" and len(weights_list) > 0:
            token_weights = pad_sequence(weights_list, batch_first=True, padding_value=0.0)
        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "concept_signals": concept_signals,
            "token_weights": token_weights,
        }

    @staticmethod
    def from_csv(
        csv_path: Path,
        images_root: Path,
        processor: Any,
        prompt_builder: Any,
        target_builder: Any,
        stage_mode: str = "A",
        w_keys: float = 1.0,
        w_reason: float = 0.2,
        template_mode: str = "train_semi_structured",
        structured_cfg: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        rows = []
        pre_keys: Dict[str, str] = {}
        pre_reasons: Dict[str, str] = {}
        with Path(csv_path).open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("image_path", "")
                img_path = str((images_root / rel).resolve())
                labels = {
                    "artist": row.get("artist", ""),
                    "style": row.get("style", ""),
                    "genre": row.get("genre", ""),
                    "media": [],
                }
                rows.append({"image_path": img_path, "labels_per_dim": labels, "concept_signals": None})
                pre_keys[img_path] = row.get("target_keys", "")
                pre_reasons[img_path] = row.get("target_reason", "")
        return LLaVAConceptGraphDataset(
            base_samples=rows,
            processor=processor,
            prompt_builder=prompt_builder,
            target_builder=target_builder,
            template_mode=template_mode,
            precomputed_targets=None,
            precomputed_keys=pre_keys,
            precomputed_reasons=pre_reasons,
            stage_mode=stage_mode,
            w_keys=w_keys,
            w_reason=w_reason,
            structured_cfg=structured_cfg,
            device=device,
            torch_dtype=torch_dtype,
        )
"""
简要介绍 (ZH):
  LLaVA 概念图数据集（阶段化 A/B）。从 CSV/JSON 载入图像与标签，
  生成 prompt 与目标文本；在阶段 B 为 keys 与 reason 段分别赋权，用于加权 CE。

Overview (EN):
  LLaVA concept-graph dataset with phased training. Loads images and labels, builds prompts and targets,
  and in Stage B provides per‑segment token weights (keys vs. reason) to enable weighted CE loss.

Inputs/Outputs:
  - Input: base_samples with image_path/labels; optional precomputed keys/reason
  - Output (collate): images, input_ids, attention_mask, labels, concept_signals, token_weights (Stage B)
"""
