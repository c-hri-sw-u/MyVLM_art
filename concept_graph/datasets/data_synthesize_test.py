import argparse
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from concept_graph.datasets.data_synthesize import (
    _default_paths,
    ConceptGraphDataset,
    build_prompt,
    build_target,
    synthesize_targets,
)


def main():
    parser = argparse.ArgumentParser(description="Smoke test: synthesize one sample and print prompt + raw output")
    parser.add_argument("--reveal_labels", type=str, default="true")
    parser.add_argument("--reasoning", type=str, default="false")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--images_root", type=str, default=None)
    args = parser.parse_args()

    reveal = str(args.reveal_labels).lower() in ["1", "true", "yes", "y"]
    include_reasoning = str(args.reasoning).lower() in ["1", "true", "yes", "y"]
    structured_cfg = {"keys_start": "[BEGIN_KEYS]", "keys_end": "[END_KEYS]", "reason_start": "[BEGIN_REASON]", "reason_end": "[END_REASON]"}

    dataset_path, images_root, model, _ = _default_paths()
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    if args.images_root:
        images_root = Path(args.images_root)

    ds = ConceptGraphDataset(dataset_path=dataset_path, images_root=images_root)
    s = ds[0]
    labels = s["labels_per_dim"]
    signals = s.get("concept_signals")
    prompt = build_prompt(labels, signals, mode="train_semi_structured", reveal_labels=reveal, include_reasoning=include_reasoning, structured_cfg=structured_cfg)
    print("Prompt:\n" + prompt)
    cache_keys, cache_reason, cache_raw = synthesize_targets([s], build_prompt, build_target, model=model, reveal_labels=reveal, include_reasoning=include_reasoning, structured_cfg=structured_cfg)
    raw_out = next(iter(cache_raw.values()))
    print("Raw Output:\n" + raw_out)


if __name__ == "__main__":
    main()
