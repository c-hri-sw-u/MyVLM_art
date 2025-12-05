from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


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
