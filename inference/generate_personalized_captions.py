import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pyrallis
import torch

from configs.myvlm_art_config import MyVLMArtConfig
from myvlm.common import VLMType, PersonalizationTask, VLM_TO_PROMPTS, VLM_TO_LAYER, VLM_TO_EMBEDDING_DIM
from myvlm.utils import parent_module, brackets_to_periods
from vlms.llava_wrapper import LLaVAWrapper
from concept_graph.concept_embeddings.multi_embed_layer import MultiTokenConceptLayer


def _resolve_dataset_json(images_root: Path, preferred: str = "") -> Path:
    if preferred:
        cand = images_root / preferred
        if cand.exists():
            return cand
    pats = ["*wikiart*5artists*.json", "*dataset*.json", "*test*.json", "*.json"]
    seen = set()
    cands: List[Path] = []
    for pat in pats:
        for p in sorted(images_root.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True):
            if str(p) in seen:
                continue
            seen.add(str(p))
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No dataset JSON found under {images_root}")
    return cands[0]


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


def load_concept_signals(images_root: Path, dataset_json: str = "") -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Tuple[int, int]]]:
    dataset_json = _resolve_dataset_json(images_root, preferred=dataset_json)
    img_list: List[str] = []
    with dataset_json.open("r") as f:
        records = json.load(f)
        for r in records:
            rel = r.get("image", "")
            if rel:
                img_list.append(str((images_root / rel).resolve()))
    signals_artist, signals_style, signals_genre = [], [], []
    p_artist = Path("artifacts/concept_signals_artist.json")
    p_style = Path("artifacts/concept_signals_style.json")
    p_genre = Path("artifacts/concept_signals_genre.json")
    if p_artist.exists():
        with p_artist.open("r") as f:
            signals_artist = json.load(f)
    if p_style.exists():
        with p_style.open("r") as f:
            signals_style = json.load(f)
    if p_genre.exists():
        with p_genre.open("r") as f:
            signals_genre = json.load(f)
    n_artist = _count_dim(signals_artist, "artist")
    n_style = _count_dim(signals_style, "style")
    n_genre = _count_dim(signals_genre, "genre")
    offset_artist = 0
    offset_style = n_artist
    offset_genre = n_artist + n_style
    dim_ranges = {
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
    return mapping, dim_ranges


def ensure_multi_token_layer(vlm: LLaVAWrapper,
                             cfg: MyVLMArtConfig,
                             dim_ranges: Dict[str, Tuple[int, int]],
                             n_concepts: int) -> MultiTokenConceptLayer:
    edit_module = parent_module(vlm, brackets_to_periods(VLM_TO_LAYER[cfg.vlm_type]))
    layer_name = VLM_TO_LAYER[cfg.vlm_type].rsplit(".", 1)[-1]
    original = getattr(edit_module, layer_name)
    if isinstance(original, MultiTokenConceptLayer):
        layer = original
    else:
        layer = MultiTokenConceptLayer(
            layer=original,
            embedding_dim=VLM_TO_EMBEDDING_DIM[cfg.vlm_type],
            max_tokens_per_concept=getattr(cfg, "max_tokens_per_concept", 4),
            threshold=cfg.threshold,
            torch_dtype=vlm.torch_dtype,
            device=vlm.device,
            max_concepts_per_sample=getattr(cfg, "max_concepts_per_sample", 0),
            backoff_delta=getattr(cfg, "backoff_delta", 0.0),
            topk_per_dim=getattr(cfg, "topk_per_dim", 0),
            fairness=getattr(cfg, "fairness", False),
            priority=[s.strip() for s in getattr(cfg, "priority", "artist,style,genre").split(",") if s.strip()],
        )
        setattr(edit_module, layer_name, layer)
    # Set values: load from checkpoint if available, else initialize
    ckpt_path = cfg.output_path / f'concept_embeddings_{cfg.vlm_type}_{cfg.personalization_task}.pt'
    if ckpt_path.exists():
        iteration_to_concept = torch.load(ckpt_path, map_location='cpu')
        if len(iteration_to_concept) > 0:
            last_iter = max(iteration_to_concept.keys())
            vals = iteration_to_concept[last_iter].get("values", None)
            if vals is not None and vals.ndim == 3 and vals.shape[-1] == VLM_TO_EMBEDDING_DIM[cfg.vlm_type]:
                setattr(layer, "values", torch.nn.Parameter(vals.to(device=vlm.device, dtype=vlm.torch_dtype), requires_grad=False))
            else:
                if getattr(layer, "values", None) is None:
                    layer.initialize_values(n_concepts)
        else:
            if getattr(layer, "values", None) is None:
                layer.initialize_values(n_concepts)
    else:
        if getattr(layer, "values", None) is None:
            layer.initialize_values(n_concepts)
    layer.set_dim_ranges(dim_ranges)
    return layer


def _collect_image_paths(images_root: Path, dataset_json: str = "") -> List[Path]:
    dataset_json = _resolve_dataset_json(images_root, preferred=dataset_json)
    with dataset_json.open("r") as f:
        records = json.load(f)
    paths: List[Path] = []
    for r in records:
        rel = r.get("image", "")
        if rel:
            paths.append((images_root / rel).resolve())
    return paths


@pyrallis.wrap()
def main(cfg: MyVLMArtConfig):
    assert cfg.vlm_type == VLMType.LLAVA, "This script currently targets LLaVA for captioning."
    assert cfg.personalization_task == PersonalizationTask.CAPTIONING, "Use captioning task for personalized captions."
    vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
    images_root = Path(cfg.data_root)
    signals_map, dim_ranges = load_concept_signals(images_root=images_root)
    concept_idxs = set()
    for v in signals_map.values():
        for k in v.keys():
            concept_idxs.add(k)
    n_concepts = max(concept_idxs) + 1 if len(concept_idxs) > 0 else 1
    ensure_multi_token_layer(vlm=vlm, cfg=cfg, dim_ranges=dim_ranges, n_concepts=n_concepts)
    prompt_template = VLM_TO_PROMPTS[cfg.vlm_type][cfg.personalization_task][0]
    outputs: Dict[str, Dict[str, str]] = {}
    for img_path in _collect_image_paths(images_root, dataset_json=getattr(cfg, "dataset_json", "")):
        prompt = prompt_template.format(concept=cfg.concept_identifier)
        inputs = vlm.preprocess(image_path=img_path, prompt=prompt)
        signal = signals_map.get(str(img_path), None)
        caption = vlm.generate(inputs=inputs, concept_signals=signal)[0]
        outputs[str(img_path)] = {prompt: caption}
    out_dir = cfg.output_path / "inference_outputs"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f'inference_outputs_{cfg.vlm_type}_{cfg.personalization_task}.json'
    with out_path.open("w") as f:
        json.dump(outputs, f, indent=4, sort_keys=True)
    print(f"Saved outputs to: {out_path}")


if __name__ == "__main__":
    main()
