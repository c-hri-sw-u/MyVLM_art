import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image
from torch.utils.data import Dataset


def build_base_samples(dataset_json: Path,
                       dimensions: Optional[List[str]] = None,
                       images_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    解析 WikiArt 数据集 JSON，构建包含图像与多维标签的 base_samples。
    """
    dataset_json = Path(dataset_json)
    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset json not found: {dataset_json}")

    with dataset_json.open('r') as f:
        entries = json.load(f)

    # 若未显式指定维度，默认使用 concepts 中所有键
    dims = set(dimensions or [])
    base_samples: List[Dict[str, Any]] = []
    for entry in entries:
        relative_path = entry.get("image")
        if relative_path is None:
            continue
        data_root = Path(images_root) if images_root is not None else dataset_json.parent
        image_path = (data_root / relative_path).resolve()
        
        if not image_path.exists():
            continue

        # 将图像加载为 RGB，避免后续重复磁盘 IO
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        concepts: Dict[str, Any] = entry.get("concepts", {})
        if not dims:
            dims.update(concepts.keys())
        labels_per_dim = {dim: concepts.get(dim) for dim in dims}

        base_samples.append({
            "image_path": image_path,
            "image": image,
            "labels_per_dim": labels_per_dim
        })
    return base_samples


class PrototypeDataset(Dataset):
    def __init__(self,
                 base_samples: List[Dict[str, Any]],
                 clip_preprocess: Any,
                 dimension: str = "style"):
        self.samples = base_samples
        self.clip_preprocess = clip_preprocess
        self.dimension = dimension

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        labels = s["labels_per_dim"]
        label = labels.get(self.dimension)
        image_tensor = self.clip_preprocess(s["image"])
        return {
            "image_anchor": image_tensor,
            "label": label,
            "labels_per_dim": labels,
            "image_path": str(s["image_path"]),
        }
