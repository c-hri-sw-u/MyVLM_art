from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


class LLaVAConceptGraphDataset(Dataset):
    def __init__(
        self,
        base_samples: List[Dict[str, Any]],
        processor: Any,
        prompts_builder: Any,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.samples = base_samples
        self.processor = processor
        self.prompts_builder = prompts_builder
        self.device = device
        self.torch_dtype = torch_dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = s["image"]
        labels = s["labels_per_dim"]
        concept_signals = s.get("concept_signals", None)
        prompts = self.prompts_builder(labels)  # 应返回若干候选；这里选择一个
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        inputs = self.processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        inputs = inputs.to(self.device, self.torch_dtype)
        # 目标文本（简单版）：稳定覆盖全部维度
        target_text_pool = [
            f"This painting is by {labels['artist']}, in {labels['style']}, showing a {labels['genre']}."
        ]
        target = np.random.choice(target_text_pool, size=1, replace=False)[0]
        batch = {
            "images": inputs,
            "prompt": prompt,
            "target_text": target,
            "concept_signals": concept_signals,
            "labels_per_dim": labels,
            "image_path": s["image_path"],
        }
        return batch

