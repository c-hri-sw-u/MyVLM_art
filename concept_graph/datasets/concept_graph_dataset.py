import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset
import torch


class ConceptGraphDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        images_root: Union[str, Path],
        prototype_head: Optional[Any] = None,
        precomputed_signals: Optional[Dict[int, Any]] = None,
        precomputed_signals_path: Optional[Union[str, Path]] = None,
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
        if precomputed_signals is not None:
            self.precomputed_signals = precomputed_signals
        elif precomputed_signals_path is not None:
            self.precomputed_signals = self._load_precomputed_signals(precomputed_signals_path)
        else:
            self.precomputed_signals = {}
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        signals = None
        if idx in self.precomputed_signals:
            signals = self.precomputed_signals[idx]
        elif self.prototype_head is not None:
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

    def _load_precomputed_signals(self, path: Union[str, Path]) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
        path = Path(path)
        with path.open("r") as f:
            payload = json.load(f)
        raw = payload.get("signals", payload)
        parsed: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = {}
        for idx_str, dim_dict in raw.items():
            idx = int(idx_str)
            parsed[idx] = {}
            for dimension, concept_scores in dim_dict.items():
                parsed[idx][dimension] = {}
                for concept_idx_str, scores in concept_scores.items():
                    parsed[idx][dimension][int(concept_idx_str)] = torch.tensor(scores, dtype=torch.float32)
        return parsed
