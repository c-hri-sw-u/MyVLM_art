from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained
from torch import Tensor
from torch.cuda import amp


class PrototypeHead:
    def __init__(self,
                 clip_model_name: str = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
                 device: Union[str, torch.device] = "cuda",
                 precision: str = "fp16",
                 batch_size: int = 16):
        self.clip_model_name = clip_model_name
        self.device = torch.device(device)
        self.precision = precision
        self.batch_size = batch_size

        self.model, self.preprocess = create_model_from_pretrained(clip_model_name, precision=precision)
        self.model.to(self.device)
        self.model.eval()

        # 结构：{dimension: Tensor[n_concepts, dim]}
        self.prototypes: Dict[str, Tensor] = {}
        self.concept_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_concept: Dict[str, List[str]] = {}

    def build_prototypes(self,
                         concept_to_paths: Dict[str, Dict[str, List[Union[str, Path]]]],
                         save_path: Optional[Path] = None) -> Dict[str, Tensor]:
        """
        根据 {dimension: {concept: [paths]}} 构建原型，必要时保存到磁盘。
        """
        self.prototypes = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}

        with torch.no_grad():
            for dimension, concept_paths in concept_to_paths.items():
                proto_list: List[Tensor] = []
                concept_names: List[str] = []
                for concept, paths in concept_paths.items():
                    path_list = [Path(p) for p in paths if Path(p).exists()]
                    if len(path_list) == 0:
                        continue
                    features, _ = self._encode_image_paths(path_list)
                    if features.numel() == 0:
                        continue
                    prototype = F.normalize(features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                    proto_list.append(prototype.cpu())
                    concept_names.append(concept)

                if proto_list:
                    stacked = torch.stack(proto_list, dim=0)
                    self.prototypes[dimension] = stacked
                    idx_map = {name: idx for idx, name in enumerate(concept_names)}
                    self.concept_to_idx[dimension] = idx_map
                    self.idx_to_concept[dimension] = concept_names

        if save_path is not None:
            payload = {
                "clip_model_name": self.clip_model_name,
                "prototypes": self.prototypes,
                "concept_to_idx": self.concept_to_idx,
                "idx_to_concept": self.idx_to_concept,
            }
            torch.save(payload, save_path)

        return self.prototypes

    def load_prototypes(self, ckpt_path: Path):
        payload = torch.load(ckpt_path, map_location="cpu")
        self.prototypes = payload["prototypes"]
        self.concept_to_idx = payload["concept_to_idx"]
        self.idx_to_concept = payload["idx_to_concept"]

    def extract_signal(self,
                       image_paths: List[Union[str, Path]],
                       dimension: str) -> Dict[Path, Dict[int, Tensor]]:
        """
        对输入图像计算与指定维度所有原型的相似度，输出兼容 concept_head 接口的伪概率。
        """
        if dimension not in self.prototypes:
            raise ValueError(f"Dimension {dimension} has no prototypes. "
                             f"Available: {list(self.prototypes.keys())}")

        prototypes = self.prototypes[dimension].to(self.device)
        path_objs = [Path(p) for p in image_paths]
        outputs: Dict[Path, Dict[int, Tensor]] = {}

        with torch.no_grad():
            features, processed_paths = self._encode_image_paths(path_objs)
            sims = torch.matmul(features, prototypes.T)
            for idx, path in enumerate(processed_paths):
                concept_scores: Dict[int, Tensor] = {}
                for concept_idx, score in enumerate(sims[idx]):
                    concept_scores[concept_idx] = torch.stack(
                        [1 - score, score], dim=0
                    ).cpu()
                outputs[path] = concept_scores
            missing = set(path_objs) - set(processed_paths)
            for missing_path in missing:
                outputs[missing_path] = {}
        return outputs

    def _encode_image_paths(self, paths: List[Path]) -> Tuple[Tensor, List[Path]]:
        """
        Helper：批量编码若干图像路径，输出 L2 归一化后的 embeddings。
        """
        embeds: List[Tensor] = []
        processed: List[Path] = []
        for i in range(0, len(paths), self.batch_size):
            batch_paths = paths[i:i + self.batch_size]
            batch_tensors = []
            for path in batch_paths:
                if not path.exists():
                    continue
                with Image.open(path) as img:
                    image = img.convert("RGB")
                batch_tensors.append(self.preprocess(image))
                processed.append(path)

            if len(batch_tensors) == 0:
                continue

            batch = torch.stack(batch_tensors, dim=0).to(self.device)
            with amp.autocast(enabled=self.precision == "fp16"):
                feats = self.model.encode_image(batch)
            feats = F.normalize(feats.float(), dim=-1)
            embeds.append(feats)

        if len(embeds) == 0:
            raise ValueError("No valid images found to encode for prototype computation.")
        return torch.cat(embeds, dim=0), processed
