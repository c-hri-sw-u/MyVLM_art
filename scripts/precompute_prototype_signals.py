#!/usr/bin/env python
"""Precompute prototype-based concept signals and cache them to disk."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from concept_graph.prototypes.prototype_head import PrototypeHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute prototype signals")
    parser.add_argument("dataset_json", type=Path, help="Path to dataset json (e.g., wikiart_5artists_dataset.json)")
    parser.add_argument("images_root", type=Path, help="Root directory containing the images referenced in the json")
    parser.add_argument("prototype_ckpt", type=Path, help="Path to saved prototype checkpoint (.pt)")
    parser.add_argument("output_path", type=Path, help="Where to save the concept signals cache (JSON)")
    parser.add_argument("--dimensions", nargs="+", default=["artist", "style", "genre"],
                        help="Which concept dimensions to process. Must exist inside prototype checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device for CLIP inference (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Batch size for CLIP encoding")
    parser.add_argument("--chunk-size", type=int, default=64, dest="chunk_size",
                        help="Number of images per extract_signal call")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default=None,
                        help="Override precision (default: fp16 if cuda else fp32)")
    return parser.parse_args()


def load_records(dataset_json: Path, images_root: Path) -> List[Path]:
    with dataset_json.open("r") as f:
        entries = json.load(f)
    image_paths: List[Path] = []
    for entry in entries:
        rel = entry.get("image")
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.is_absolute():
            img_path = rel_path
        else:
            parts = rel_path.parts
            if len(parts) > 0 and parts[0] == images_root.name:
                rel_path = Path(*parts[1:])
            img_path = images_root / rel_path
        if img_path.exists():
            image_paths.append(img_path.resolve())
    return image_paths


def chunked(seq: List[Path], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    args = parse_args()
    precision = args.precision
    if precision is None:
        precision = "fp16" if torch.device(args.device).type == "cuda" else "fp32"

    head = PrototypeHead(
        device=args.device,
        precision=precision,
        batch_size=args.batch_size,
    )
    head.load_prototypes(args.prototype_ckpt)

    image_paths = load_records(args.dataset_json, args.images_root)
    if not image_paths:
        raise RuntimeError("No valid image paths found. Check dataset_json and images_root")

    path_to_index = {path: idx for idx, path in enumerate(image_paths)}
    cache: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for dim in args.dimensions:
        if dim not in head.prototypes:
            raise ValueError(f"Dimension '{dim}' not found in loaded prototypes. Available: {list(head.prototypes.keys())}")
        print(f"Processing dimension '{dim}' with {len(head.prototypes[dim])} concepts...")
        for chunk in tqdm(list(chunked(image_paths, args.chunk_size)), desc=f"{dim} chunks"):
            signals = head.extract_signal(chunk, dimension=dim)
            for path, concept_dict in signals.items():
                idx = path_to_index[path]
                dim_cache = {}
                for concept_idx, tensor_scores in concept_dict.items():
                    dim_cache[str(concept_idx)] = tensor_scores.tolist()
                cache.setdefault(str(idx), {})[dim] = dim_cache

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        json.dump({
            "meta": {
                "dataset_json": str(args.dataset_json.resolve()),
                "images_root": str(args.images_root.resolve()),
                "prototype_ckpt": str(args.prototype_ckpt.resolve()),
                "dimensions": args.dimensions,
            },
            "signals": cache,
        }, f)
    print(f"Saved signals for {len(cache)} images to {args.output_path}")


if __name__ == "__main__":
    main()
