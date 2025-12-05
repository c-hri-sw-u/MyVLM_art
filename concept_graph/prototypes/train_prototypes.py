import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from concept_graph.prototypes.prototype_head import PrototypeHead
from concept_graph.prototypes.prototype_dataset import build_base_samples, PrototypeDataset
from concept_graph.prototypes.prototype_trainer import PrototypeTrainer, PrototypeTrainerConfig


def build_concept_to_paths(dataset_json: Path, images_root: Path, dimension: str):
    with dataset_json.open("r") as f:
        records = json.load(f)
    mapping = {}
    for r in records:
        c = r.get("concepts", {})
        label = c.get(dimension)
        rel = r.get("image", "")
        if label is None or rel == "":
            continue
        p = images_root / Path(rel)
        if not p.exists():
            continue
        mapping.setdefault(label, []).append(str(p.resolve()))
    return {dimension: mapping}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=False)
    parser.add_argument("--dimension", type=str, default="style")
    parser.add_argument("--clip_model_name", type=str, default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--negatives_k", type=str, default="16")
    parser.add_argument("--hard_negatives", type=str, default="true")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lambda_margin", type=float, default=0.5)
    parser.add_argument("--debug_log", type=str, default="false")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    if args.images_root:
        images_root = Path(args.images_root)
    else:
        images_root = dataset_json.parent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = PrototypeHead(clip_model_name=args.clip_model_name, device=device, batch_size=16)

    mapping = build_concept_to_paths(dataset_json, images_root, args.dimension)
    head.build_prototypes(mapping, save_path=Path(args.save_path))

    base = build_base_samples(dataset_json, dimensions=[args.dimension])
    ds = PrototypeDataset(base_samples=base, clip_preprocess=head.preprocess, dimension=args.dimension)

    # 预解析 negatives_k，支持 "max" 语义（类别数-1）
    num_classes = head.prototypes[args.dimension].size(0) if args.dimension in head.prototypes else 0
    if str(args.negatives_k).lower() == "max" and num_classes > 0:
        negatives_k = max(1, num_classes - 1)
    else:
        try:
            negatives_k = int(args.negatives_k)
        except Exception:
            negatives_k = 16

    cfg = PrototypeTrainerConfig(
        dimension=args.dimension,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        temperature=args.temperature,
        negatives_k=negatives_k,
        hard_negatives=str(args.hard_negatives).lower() in ["1", "true", "yes", "y"],
        margin=args.margin,
        lambda_margin=args.lambda_margin,
        debug_log=str(args.debug_log).lower() in ["1", "true", "yes", "y"],
        log_interval=args.log_interval,
        save_path=args.save_path,
    )
    trainer = PrototypeTrainer(cfg, ds, head)
    log = trainer.train()
    print(json.dumps(log))


if __name__ == "__main__":
    main()
