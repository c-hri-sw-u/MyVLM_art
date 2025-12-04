from typing import Dict, Any, Optional, List

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
        self.structured_cfg = structured_cfg or {}
        self.device = device
        self.torch_dtype = torch_dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = s["image"]
        labels = s["labels_per_dim"]
        concept_signals = s.get("concept_signals", None)
        prompt = self.prompt_builder(labels, concept_signals, mode=self.template_mode, structured_cfg=self.structured_cfg)
        inputs = self.processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        inputs = inputs.to(self.device, self.torch_dtype)
        if s["image_path"] in self.precomputed_targets:
            target = self.precomputed_targets[s["image_path"]]
        else:
            target = self.target_builder(labels, concept_signals, mode=self.template_mode)
        batch = {
            "images": inputs,
            "prompt": prompt,
            "target_text": target,
            "concept_signals": concept_signals,
            "labels_per_dim": labels,
            "image_path": s["image_path"],
        }
        return batch
