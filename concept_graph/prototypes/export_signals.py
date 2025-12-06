"""
简要介绍 (ZH):
  概念信号导出工具。读取原型检查点与数据集，批量计算每张图的概念相似度，并输出 JSON/CSV。

Overview (EN):
  Concept signal exporter. Loads prototype checkpoints and dataset, computes per‑image concept similarities,
  and writes JSON/CSV for downstream gating and training.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import csv

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from concept_graph.prototypes.prototype_head import PrototypeHead


def load_ckpt_meta(ckpt: Path) -> Dict:
    payload = torch.load(ckpt, map_location="cpu")
    return payload


def read_image_paths(dataset_json: Path, images_root: Path) -> List[Path]:
    with dataset_json.open("r") as f:
        records = json.load(f)
    paths: List[Path] = []
    for r in records:
        rel = r.get("image", "")
        p = Path(rel)
        if p.is_absolute():
            ip = p
        else:
            ip = images_root / p
        paths.append(ip.resolve())
    return paths


def to_jsonable(signal: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for k, v in signal.items():
        out[str(k)] = [float(x) for x in v.tolist()]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=False)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dimensions", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, default="json")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--normalize", type=str, default="none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip", type=str, default="true")
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    images_root = Path(args.images_root) if args.images_root else dataset_json.parent
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.output)
    fmt = args.format.strip().lower() if args.format else (out_path.suffix.lstrip(".") or "json")
    topk = max(1, int(args.topk))
    normalize_mode = args.normalize.strip().lower()
    temperature = float(args.temperature)
    do_clip = str(args.clip).lower() in ["1", "true", "yes", "y"]

    meta = load_ckpt_meta(ckpt_path)
    model_name = meta.get("clip_model_name", "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")
    head = PrototypeHead(clip_model_name=model_name)
    head.load_prototypes(ckpt_path)

    dims = args.dimensions.strip()
    if dims:
        dims_list = [d.strip() for d in dims.split(",") if d.strip()]
    else:
        dims_list = list(head.prototypes.keys())

    paths = read_image_paths(dataset_json, images_root)
    idx_of: Dict[Path, int] = {p: i for i, p in enumerate(paths)}
    with dataset_json.open("r") as f:
        records = json.load(f)

    def norm_s(x: float) -> float:
        if do_clip:
            x = max(-1.0, min(1.0, x))
        if normalize_mode == "zero_one":
            x = 0.5 * (x + 1.0)
        elif normalize_mode == "clamp_zero":
            x = max(0.0, x)
        x = max(-1.0, min(1.0, x))
        if normalize_mode == "zscore":
            pass
        return x

    if fmt == "json":
        results: List[Dict] = [dict() for _ in range(len(paths))]
        for dim in dims_list:
            signals = head.extract_signal(paths, dim)
            for p, sig in signals.items():
                i = idx_of.get(p)
                if i is None:
                    continue
                items = []
                for k_idx, vec in sig.items():
                    s = float(vec[1].item()) if hasattr(vec[1], "item") else float(vec[1])
                    items.append((k_idx, s))
                if normalize_mode == "zscore":
                    vals = [t[1] for t in items]
                    mean = sum(vals) / max(1, len(vals))
                    var = sum((v - mean) * (v - mean) for v in vals) / max(1, len(vals))
                    std = var ** 0.5
                    normed = []
                    for k_idx, s in items:
                        z = (s - mean) / (std + 1e-6)
                        p = 1.0 / (1.0 + torch.exp(torch.tensor(-z))).item()
                        if temperature != 1.0:
                            p = float(max(0.0, min(1.0, p)) ** (1.0 / max(1e-6, temperature)))
                        normed.append((k_idx, float(max(0.0, min(1.0, p)))))
                    items = normed
                else:
                    normed = []
                    for k_idx, s in items:
                        s2 = norm_s(s)
                        if temperature != 1.0:
                            s2 = float(max(0.0, min(1.0, s2)) ** (1.0 / max(1e-6, temperature)))
                        normed.append((k_idx, float(max(0.0, min(1.0, s2)))))
                    items = normed
                sig_out: Dict[int, torch.Tensor] = {}
                for k_idx, s in items:
                    d_norm = float(1.0 - s)
                    t = torch.tensor([d_norm, s], dtype=torch.float32)
                    sig_out[int(k_idx) if not isinstance(k_idx, int) else k_idx] = t
                results[i][dim] = to_jsonable(sig_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f)
        return

    rows: List[List[str]] = []
    header: List[str] = ["image_path", "artist", "style", "genre"]
    legends: Dict[str, List[str]] = {d: head.idx_to_concept.get(d, []) for d in dims_list}
    for d in dims_list:
        for k in range(1, topk + 1):
            header.append(f"{d}_top{k}_name")
            header.append(f"{d}_top{k}_score")

    for p in paths:
        try:
            rel_path = str(p.resolve().relative_to(images_root.resolve()))
        except Exception:
            rel_path = p.name
        i = idx_of.get(p)
        artist = style = genre = ""
        if i is not None:
            r = records[i]
            c = r.get("concepts", {})
            artist = c.get("artist", r.get("artist", ""))
            style = c.get("style", "")
            genre = c.get("genre", "")
        row: List[str] = [rel_path, artist, style, genre]
        for d in dims_list:
            sig = head.extract_signal([p], d).get(p, {})
            items = []
            for k_idx, vec in sig.items():
                idx = int(k_idx) if not isinstance(k_idx, int) else k_idx
                name = legends[d][idx] if idx < len(legends[d]) else str(idx)
                s = float(vec[1].item()) if hasattr(vec[1], "item") else float(vec[1])
                items.append((name, s))
            if normalize_mode == "zscore":
                vals = [t[1] for t in items]
                mean = sum(vals) / max(1, len(vals))
                var = sum((v - mean) * (v - mean) for v in vals) / max(1, len(vals))
                std = var ** 0.5
                items2 = []
                for name, s in items:
                    z = (s - mean) / (std + 1e-6)
                    p = 1.0 / (1.0 + torch.exp(torch.tensor(-z))).item()
                    if temperature != 1.0:
                        p = float(max(0.0, min(1.0, p)) ** (1.0 / max(1e-6, temperature)))
                    items2.append((name, float(max(0.0, min(1.0, p)))))
                items = items2
            else:
                items2 = []
                for name, s in items:
                    s2 = norm_s(s)
                    if temperature != 1.0:
                        s2 = float(max(0.0, min(1.0, s2)) ** (1.0 / max(1e-6, temperature)))
                    items2.append((name, float(max(0.0, min(1.0, s2)))))
                items = items2
            items.sort(key=lambda t: t[1], reverse=True)
            for k in range(topk):
                if k < len(items):
                    name, s = items[k]
                    row.append(name)
                    row.append(f"{s:.6f}")
                else:
                    row.append("")
                    row.append("")
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
