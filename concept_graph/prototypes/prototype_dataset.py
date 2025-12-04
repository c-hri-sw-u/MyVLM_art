from typing import Dict, Any, List

from torch.utils.data import Dataset


class PrototypeDataset(Dataset):
    def __init__(self, base_samples: List[Dict[str, Any]], clip_preprocess: Any, dimension: str = "style"):
        self.samples = base_samples
        self.clip_preprocess = clip_preprocess
        self.dimension = dimension
        # TODO: 可在此构造维度内的概念索引，便于抽取正负样本

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = s["image"]
        labels = s["labels_per_dim"]
        x = self.clip_preprocess(image)
        # 简化：仅返回 anchor；正负样本选择逻辑由外部 sampler/coach 负责
        return {
            "image_anchor": x,
            "label": labels.get(self.dimension, None),
            "labels_per_dim": labels,
            "image_path": s["image_path"],
        }

