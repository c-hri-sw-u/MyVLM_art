"""
简要介绍 (ZH):
  概念图通用数据集包装器。解析 WikiArt 风格 JSON，统一输出 image_path 与多维标签，
  可选注入预计算概念信号，用于下游数据集与推理。

Overview (EN):
  Generic concept‑graph dataset. Parses WikiArt‑like JSON to yield image paths and multi‑dimensional labels,
  optionally attaches precomputed concept signals for downstream datasets and reasoning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class ConceptGraphDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        images_root: Union[str, Path],
        prototype_head: Optional[Any] = None,
        precomputed_signals: Optional[Dict[int, Any]] = None,
        transforms: Optional[Any] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.images_root = Path(images_root)
        with self.dataset_path.open("r") as f:
            records = json.load(f)
        self.records: List[Dict[str, Any]] = []
        for r in records:
            img_rel = r.get("image", "")
            img_rel_path = Path(img_rel)
            if img_rel_path.is_absolute():
                img_path = img_rel_path
            else:
                parts = img_rel_path.parts
                if len(parts) > 0 and parts[0] == self.images_root.name:
                    img_rel_path = Path(*parts[1:])
                img_path = self.images_root / img_rel_path
            c = r.get("concepts", {})
            media = c.get("media", [])
            if isinstance(media, str):
                media = [] if media == "" else [media]
            self.records.append(
                {
                    "image_path": img_path,
                    "labels_per_dim": {
                        "artist": c.get("artist", r.get("artist", "")),
                        "style": c.get("style", ""),
                        "genre": c.get("genre", ""),
                        "media": media,
                    },
                    "title": r.get("title", ""),
                    "date": r.get("date", ""),
                }
            )
        self.prototype_head = prototype_head
        self.precomputed_signals = precomputed_signals or {}
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        signals = None
        if isinstance(self.precomputed_signals, list):
            if idx < len(self.precomputed_signals):
                signals = self.precomputed_signals[idx]
        elif isinstance(self.precomputed_signals, dict):
            if idx in self.precomputed_signals:
                signals = self.precomputed_signals[idx]
        if signals is None and self.prototype_head is not None:
            out = self.prototype_head.extract_signal([rec["image_path"]])
            signals = out.get(rec["image_path"], None)
        return {
            "image": img,
            "image_path": str(rec["image_path"]),
            "labels_per_dim": rec["labels_per_dim"],
            "concept_signals": signals,
            "title": rec["title"],
            "date": rec["date"],
        }

