from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

from configs.myvlm_art_config import MyVLMArtConfig
from myvlm.common import VLMType, PersonalizationTask
from concept_graph.reasoning.inference_prompt_templates import get_prompts, build_prompt_structured
from inference.generate_personalized_captions import load_concept_signals, ensure_multi_token_layer
from concept_graph.prototypes.prototype_head import PrototypeHead
from vlms.llava_wrapper import LLaVAWrapper


def _select_activated(sample_sig: Dict[int, torch.Tensor], dim_ranges: Dict[str, Tuple[int, int]], threshold: float, backoff_delta: float, topk_per_dim: int) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for dim, (start, end) in dim_ranges.items():
        pairs: List[Tuple[int, float]] = []
        for idx, probas in sample_sig.items():
            if idx < start or idx >= end:
                continue
            s = float(probas[0][1])
            if s >= threshold:
                pairs.append((idx, s))
        pairs.sort(key=lambda t: t[1], reverse=True)
        if len(pairs) == 0 and backoff_delta > 0.0:
            cands: List[Tuple[int, float]] = []
            for idx, probas in sample_sig.items():
                if idx < start or idx >= end:
                    continue
                s = float(probas[0][1])
                if s >= max(0.0, threshold - backoff_delta):
                    cands.append((idx, s))
            cands.sort(key=lambda t: t[1], reverse=True)
            if len(cands) > 0:
                pairs = [cands[0]]
        if topk_per_dim > 0 and len(pairs) > topk_per_dim:
            pairs = pairs[:topk_per_dim]
        out[dim] = [p[0] for p in pairs]
    return out


def _occlusion_saliency(prototype: PrototypeHead, image_path: Path, dimension: str, concept_idx: int, grid: int = 16) -> np.ndarray:
    with Image.open(image_path).convert("RGB") as img:
        w, h = img.size
        base = prototype.extract_signal([image_path], dimension)[image_path][concept_idx][1].item()
        heat = np.zeros((grid, grid), dtype=np.float32)
        cell_w, cell_h = max(1, w // grid), max(1, h // grid)
        for i in range(grid):
            for j in range(grid):
                occluded = img.copy()
                draw = ImageDraw.Draw(occluded)
                draw.rectangle((j * cell_w, i * cell_h, (j + 1) * cell_w, (i + 1) * cell_h), fill=(128, 128, 128))
                tmp_path = image_path
                score = prototype.extract_signal([tmp_path], dimension)[tmp_path].get(concept_idx, torch.tensor([0.0, 0.0]))[1].item()
                heat[i, j] = max(0.0, base - score)
        return heat


def _overlay_saliency(image_path: Path, heatmap: np.ndarray, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    hmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    W, H = img.size
    hm = Image.fromarray((hmap * 255).astype(np.uint8)).resize((W, H))
    r = Image.new("L", (W, H), 0)
    r.paste(hm)
    a = Image.new("L", (W, H), 0)
    a.paste(hm)
    overlay = Image.merge("RGBA", (r, Image.new("L", (W, H), 0), Image.new("L", (W, H), 0), a))
    base = img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.6).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path)


def _saliency_stats(heatmap: np.ndarray) -> Dict[str, Any]:
    hmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    mean_val = float(hmap.mean())
    max_val = float(hmap.max())
    idx = int(hmap.argmax())
    h, w = hmap.shape
    max_pos = {"i": idx // w, "j": idx % w}
    return {"mean": mean_val, "max": max_val, "max_pos": max_pos, "grid": [h, w]}


def _aggregate_attentions(attns: Any) -> Optional[torch.Tensor]:
    if attns is None:
        return None
    if isinstance(attns, (list, tuple)):
        if len(attns) == 0:
            return None
        first = attns[0]
        if isinstance(first, (list, tuple)):
            xs: List[torch.Tensor] = []
            for step in attns:
                step_tensors = [t.float() for t in step if torch.is_tensor(t)]
                if len(step_tensors) == 0:
                    continue
                xs.append(torch.stack(step_tensors).mean(0))
            if len(xs) == 0:
                return None
            return torch.stack(xs).mean(0)
        else:
            ts = [t.float() for t in attns if torch.is_tensor(t)]
            if len(ts) == 0:
                return None
            return torch.stack(ts).mean(0)
    if torch.is_tensor(attns):
        return attns.float()
    return None


def _attention_vector_from_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
    try:
        return t.float().mean(dim=(0, 1, 2))
    except Exception:
        return None


def _resample_to_len(v: torch.Tensor, n: int) -> torch.Tensor:
    K = int(v.shape[0])
    if K <= 0:
        return torch.zeros(n, dtype=v.dtype, device=v.device)
    if K == n:
        return v
    pos = torch.linspace(0, K - 1, steps=n, device=v.device)
    left = pos.floor().long()
    right = torch.clamp(left + 1, max=K - 1)
    alpha = (pos - left.float()).clamp(0, 1)
    return (1 - alpha) * v[left] + alpha * v[right]


def _attn_to_heat(attns: Any, grid: int) -> Optional[np.ndarray]:
    vecs: List[torch.Tensor] = []
    if torch.is_tensor(attns):
        v = _attention_vector_from_tensor(attns)
        if v is not None:
            vecs.append(v)
    elif isinstance(attns, (list, tuple)):
        for x in attns:
            if torch.is_tensor(x):
                v = _attention_vector_from_tensor(x)
                if v is not None:
                    vecs.append(v)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    if torch.is_tensor(y):
                        v = _attention_vector_from_tensor(y)
                        if v is not None:
                            vecs.append(v)
    if len(vecs) == 0:
        return None
    n = int(grid) * int(grid)
    rs = [
        _resample_to_len(v, n)
        for v in vecs
        if v is not None and v.numel() > 0
    ]
    if len(rs) == 0:
        return None
    M = torch.stack(rs).mean(dim=0)
    heat = M.detach().cpu().numpy().reshape(int(grid), int(grid))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


def _llava_occlusion_saliency(vlm_wrapper, temp_inputs: Any, sig: Dict[int, torch.Tensor], grid: int = 12) -> np.ndarray:
    with torch.no_grad():
        base = vlm_wrapper.model(
            input_ids=temp_inputs.input_ids,
            images=temp_inputs.image_tensor,
            concept_signals=vlm_wrapper.prepare_concept_signals(sig),
            labels=temp_inputs.targets,
            return_dict=True,
        )
    base_loss = float(getattr(base, "loss", 0.0))
    x = temp_inputs.image_tensor.clone()
    _, C, H, W = x.shape
    cell_h, cell_w = max(1, H // grid), max(1, W // grid)
    fill = x.mean(dim=(2, 3), keepdim=True)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            occl = x.clone()
            occl[:, :, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w] = fill
            with torch.no_grad():
                out = vlm_wrapper.model(
                    input_ids=temp_inputs.input_ids,
                    images=occl,
                    concept_signals=vlm_wrapper.prepare_concept_signals(sig),
                    labels=temp_inputs.targets,
                    return_dict=True,
                )
            loss = float(getattr(out, "loss", base_loss))
            heat[i, j] = max(0.0, loss - base_loss)
    return heat


def run_reasoning(vlm_wrapper, activated_concepts: Optional[Dict[str, Dict[str, List[int]]]], images: List[str], concept_embeddings, cfg: MyVLMArtConfig, structured: bool = False, structured_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    assert cfg.vlm_type == VLMType.LLAVA
    assert cfg.personalization_task == PersonalizationTask.CAPTIONING
    images_root = Path(cfg.data_root)
    signals_map, dim_ranges = load_concept_signals(images_root=images_root, dataset_json=str(getattr(cfg, "dataset_json", "")))
    concept_idxs: List[int] = []
    for v in signals_map.values():
        for k in v.keys():
            concept_idxs.append(k)
    n_concepts = max(concept_idxs) + 1 if len(concept_idxs) > 0 else 1
    layer = ensure_multi_token_layer(vlm=vlm_wrapper, cfg=cfg, dim_ranges=dim_ranges, n_concepts=n_concepts)
    ckpt_path = Path(concept_embeddings) if not isinstance(concept_embeddings, dict) else None
    if ckpt_path is not None and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and len(payload) > 0:
            try:
                last = max(payload.keys())
                vals = payload[last].get("values", None)
            except Exception:
                vals = payload.get("values", None)
            if vals is not None:
                setattr(layer, "values", torch.nn.Parameter(vals.to(device=vlm_wrapper.device, dtype=vlm_wrapper.torch_dtype), requires_grad=False))
    idx_to_concept_map: Dict[str, List[str]] = {}
    for dim, ck in {
        "artist": Path("artifacts/prototypes_artist_trained.pt"),
        "style": Path("artifacts/prototypes_style_trained.pt"),
        "genre": Path("artifacts/prototypes_genre_trained.pt"),
    }.items():
        if ck.exists():
            head = PrototypeHead(device=cfg.device)
            head.load_prototypes(ck)
            idx_to_concept_map[dim] = head.idx_to_concept.get(dim, [])
    outputs: Dict[str, Dict[str, Any]] = {}
    for i, img in enumerate(images):
        p = Path(img)
        try:
            print(f"[{i+1}/{len(images)}] {p.name}")
        except Exception:
            pass
        sig = signals_map.get(str(p), {})
        if activated_concepts and str(p) in activated_concepts:
            ac = activated_concepts[str(p)]
        else:
            ac = _select_activated(sig, dim_ranges, cfg.threshold, cfg.backoff_delta, int(getattr(cfg, "topk_per_dim", 0)))
        # build per-dimension rankings
        rankings: Dict[str, List[Tuple[int, float]]] = {}
        for dim in ["artist", "style", "genre"]:
            start, end = dim_ranges[dim]
            pairs: List[Tuple[int, float]] = []
            for idx, probas in sig.items():
                if idx < start or idx >= end:
                    continue
                s = float(probas[0][1])
                pairs.append((idx, s))
            pairs.sort(key=lambda t: t[1], reverse=True)
            rankings[dim] = pairs
        # map global idx to dimension-local names if prototype ckpt available
        meta = {"activation": {}}
        for dim in ["artist", "style", "genre"]:
            sel_g = ac.get(dim, [])
            rank_g = rankings.get(dim, [])
            start = dim_ranges[dim][0]
            names = idx_to_concept_map.get(dim, [])
            sel_n: List[str] = []
            for gi in sel_g:
                li = gi - start
                sel_n.append(names[li] if names and 0 <= li < len(names) else str(gi))
            rank_n: List[str] = []
            for gi, _ in rank_g:
                li = gi - start
                rank_n.append(names[li] if names and 0 <= li < len(names) else str(gi))
            meta["activation"][dim] = {
                "selected_global": sel_g,
                "selected_names": sel_n,
                "ranked_global": rank_g,
                "ranked_names": rank_n,
            }
        if structured:
            prompt = build_prompt_structured({"labels_per_dim": {k: ac.get(k, []) for k in ["artist", "style", "genre"]}}, structured_cfg)
            prompts = [prompt]
            try:
                print("Prompt mode=STRUCTURED")
            except Exception:
                pass
        else:
            prompts = get_prompts({"labels_per_dim": {k: ac.get(k, []) for k in ["artist", "style", "genre"]}})
            try:
                print("Prompt mode=NATURAL")
            except Exception:
                pass
        res: Dict[str, str] = {}
        attn_cache: Dict[str, Any] = {}
        for prompt in prompts:
            inputs = vlm_wrapper.preprocess(image_path=p, prompt=prompt)
            with torch.cuda.amp.autocast():
                out = vlm_wrapper.model.generate(
                    inputs=inputs['input_ids'],
                    images=inputs['image_tensor'],
                    concept_signals=vlm_wrapper.prepare_concept_signals(sig),
                    do_sample=True if getattr(vlm_wrapper, "temperature", 0.2) > 0 else False,
                    temperature=getattr(vlm_wrapper, "temperature", 0.2),
                    top_p=getattr(vlm_wrapper, "top_p", 0.7),
                    stopping_criteria=[inputs['stopping_criteria']],
                    max_new_tokens=int(getattr(cfg, "max_reason_tokens", 200)),
                    output_attentions=False,
                    return_dict_in_generate=True,
                )
            text = vlm_wrapper.processor.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
            res[prompt] = text
            attn_cache[prompt] = getattr(out, "attentions", None)
        out_entry: Dict[str, Any] = {"__meta__": meta}
        out_entry.update(res)
        outputs[str(p)] = out_entry
        if bool(getattr(cfg, "save_saliency", False)):
            source = str(getattr(cfg, "saliency_source", "prototype")).strip().lower()
            try:
                print(f"保存显著图: source={source}, grid={int(getattr(cfg, 'saliency_grid', 16))}")
            except Exception:
                pass
            try:
                if source not in {"prototype", "llava_structured", "llava", "both"}:
                    print(f"未知saliency_source: {source}，可选值: prototype/llava_structured/llava/both")
            except Exception:
                pass
            if source in ("prototype", "both"):
                # 原型概念驱动显著图（当前实现）
                for dim in ["artist", "style", "genre"]:
                    ck = {
                        "artist": Path("artifacts/prototypes_artist_trained.pt"),
                        "style": Path("artifacts/prototypes_style_trained.pt"),
                        "genre": Path("artifacts/prototypes_genre_trained.pt"),
                    }[dim]
                    if ck.exists():
                        proto = PrototypeHead(device=cfg.device)
                        proto.load_prototypes(ck)
                        sel = ac.get(dim, [])
                        if sel:
                            concept_idx = sel[0] - dim_ranges[dim][0]
                            try:
                                print(f"生成prototype显著图: {p.name} dim={dim} concept_idx={concept_idx} grid={int(getattr(cfg, 'saliency_grid', 16))}")
                            except Exception:
                                pass
                            heat = _occlusion_saliency(proto, p, dim, concept_idx, grid=int(getattr(cfg, "saliency_grid", 16)))
                            out_img = cfg.output_path / "saliency" / "prototype" / f"{p.stem}_{dim}.png"
                            _overlay_saliency(p, heat, out_img)
                            stats = _saliency_stats(heat)
                            out_entry.setdefault("__meta__", {}).setdefault("saliency", {})[dim] = {
                                "path": str(out_img.resolve()),
                                "stats": stats,
                            }
                            try:
                                print(f"保存prototype显著图: {out_img}")
                            except Exception:
                                pass
                        else:
                            try:
                                print(f"跳过prototype显著图: {p.name} dim={dim} 未选择概念")
                            except Exception:
                                pass
                    else:
                        try:
                            print(f"跳过prototype显著图: {p.name} dim={dim} 原型权重缺失 {ck}")
                        except Exception:
                            pass
            if source in ("llava_structured", "llava", "both"):
                try:
                    try:
                        print("生成LLaVA显著图: 遮挡损失")
                    except Exception:
                        pass
                    s_prompt = build_prompt_structured({"labels_per_dim": {k: ac.get(k, []) for k in ["artist", "style", "genre"]}}, structured_cfg)
                    s_text = out_entry.get(s_prompt, "")
                    tmp_inputs = vlm_wrapper.prepare_inputs(image=Image.open(p), prompt=s_prompt, target=s_text)
                    heat = _llava_occlusion_saliency(vlm_wrapper, tmp_inputs, sig, grid=int(getattr(cfg, "saliency_grid", 16)))
                    out_img = cfg.output_path / "saliency" / "llava" / f"{p.stem}.png"
                    _overlay_saliency(p, heat, out_img)
                    stats = _saliency_stats(heat)
                    out_entry.setdefault("__meta__", {}).setdefault("saliency_llava", {})["structured"] = {"path": str(out_img.resolve()), "stats": stats}
                    try:
                        print(f"保存LLaVA显著图: {out_img}")
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        print(f"LLaVA显著图失败: {e}")
                    except Exception:
                        pass
    out_dir = cfg.output_path / "reasoning_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reasoning.json").open("w") as f:
        json.dump(outputs, f, indent=4, sort_keys=True)
    return outputs
