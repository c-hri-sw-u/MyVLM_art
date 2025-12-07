import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, List, Optional

import pyrallis
import torch

from configs.myvlm_art_config import MyVLMArtConfig
from myvlm.common import VLMType, PersonalizationTask
from vlms.llava_wrapper import LLaVAWrapper
from concept_graph.reasoning.reasoning_runner import run_reasoning

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


def _load_images(images_root: Path, dataset_json: str = "") -> List[str]:
    p = _resolve_dataset_json(images_root, preferred=dataset_json)
    records = json.load(p.open())
    images = []
    for r in records:
        rel = r.get("image", "")
        if rel:
            images.append(str((images_root / rel).resolve()))
    return images


@pyrallis.wrap()
def main(cfg: MyVLMArtConfig, images: Optional[List[str]] = None, checkpoints_path: Optional[str] = None, prompt: Optional[str] = None):
    assert cfg.vlm_type == VLMType.LLAVA
    assert cfg.personalization_task == PersonalizationTask.CAPTIONING

    vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
    images_root = Path(cfg.data_root)
    if images is None or len(images) == 0:
        images = _load_images(images_root, dataset_json=str(getattr(cfg, "dataset_json", "")))
    if checkpoints_path:
        ckpt_path = checkpoints_path
    else:
        base_dir = cfg.output_root / cfg.concept_name / f"seed_{cfg.seed}"
        cands: List[Path] = []
        if base_dir.exists():
            cands.extend(sorted(base_dir.glob("checkpoints_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
            cands.extend(sorted(base_dir.glob("concept_embeddings_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        ckpt_path = str(cands[0]) if len(cands) > 0 else str(base_dir / f"checkpoints_{cfg.concept_name}_seed_{cfg.seed}.pt")

    cfg_prompt = getattr(cfg, "prompt", None)
    prompt_mode = (prompt if prompt is not None else (cfg_prompt if cfg_prompt is not None else "natural")).strip().lower()
    lim = int(getattr(cfg, "limit", 0))
    if lim > 0:
        images = images[:lim]
    print(f"Reasoning start | images={len(images)} | mode={prompt_mode} | limit={lim}")
    print(f"Checkpoint: {ckpt_path}")
    if prompt_mode == "both":
        print("Running NATURAL prompts...")
        out_nat = run_reasoning(
            vlm_wrapper=vlm,
            activated_concepts=None,
            images=images,
            concept_embeddings=ckpt_path,
            cfg=cfg,
            structured=False,
        )
        print("Running STRUCTURED prompts...")
        out_struct = run_reasoning(
            vlm_wrapper=vlm,
            activated_concepts=None,
            images=images,
            concept_embeddings=ckpt_path,
            cfg=cfg,
            structured=True,
            structured_cfg={"sentinel_start": "<BEGIN_JSON>", "sentinel_end": "<END_JSON>"},
        )
        outputs = {}
        keys = set(out_nat.keys()) | set(out_struct.keys())
        for k in keys:
            out = {}
            for d in [out_nat.get(k, {}), out_struct.get(k, {})]:
                out.update(d)
            outputs[k] = out
    else:
        outputs = run_reasoning(
            vlm_wrapper=vlm,
            activated_concepts=None,
            images=images,
            concept_embeddings=ckpt_path,
            cfg=cfg,
            structured=(prompt_mode == "structured") or bool(getattr(cfg, "structured_output", False)),
            structured_cfg={"sentinel_start": "<BEGIN_JSON>", "sentinel_end": "<END_JSON>"},
        )

    out_dir = cfg.output_root / cfg.concept_name / f"seed_{cfg.seed}" / "reasoning_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reasoning.json").open("w") as f:
        json.dump(outputs, f, indent=4, sort_keys=True)
    print(f"Saved reasoning to: {out_dir / 'reasoning.json'}")


if __name__ == "__main__":
    main()
