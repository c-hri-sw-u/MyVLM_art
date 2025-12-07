import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, List, Optional

import pyrallis
import torch
from PIL import Image, ImageDraw

from configs.myvlm_art_config import MyVLMArtConfig
from myvlm.common import VLMType, PersonalizationTask
from vlms.llava_wrapper import LLaVAWrapper
from concept_graph.reasoning.inference_prompt_templates import get_prompts, build_prompt_structured


def _load_images(images_root: Path, dataset_json: str = "wikiart_5artists_dataset.json") -> List[str]:
    records = json.load((images_root / dataset_json).open())
    images = []
    for r in records:
        rel = r.get("image", "")
        if rel:
            images.append(str((images_root / rel).resolve()))
    return images


def _overlay_saliency(image_path: Path, heatmap: torch.Tensor, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    hmap = heatmap.detach().cpu().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    W, H = img.size
    hm = Image.fromarray((hmap * 255).astype('uint8')).resize((W, H))
    r = Image.new("L", (W, H), 0)
    r.paste(hm)
    a = Image.new("L", (W, H), 0)
    a.paste(hm)
    overlay = Image.merge("RGBA", (r, Image.new("L", (W, H), 0), Image.new("L", (W, H), 0), a))
    base = img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.6).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path)


def _saliency_stats(heatmap: torch.Tensor) -> Dict[str, Any]:
    h = heatmap.detach().cpu()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    mean_val = float(h.mean().item())
    max_val = float(h.max().item())
    idx = int(torch.argmax(h).item())
    H, W = h.shape
    return {"mean": mean_val, "max": max_val, "max_pos": {"i": idx // W, "j": idx % W}, "grid": [H, W]}


def _llava_occlusion_saliency(vlm_wrapper: LLaVAWrapper, temp_inputs: Any, grid: int = 12) -> torch.Tensor:
    with torch.no_grad():
        base = vlm_wrapper.model(
            input_ids=temp_inputs.input_ids,
            images=temp_inputs.image_tensor,
            labels=temp_inputs.targets,
            return_dict=True,
        )
    base_loss = float(getattr(base, "loss", 0.0))
    x = temp_inputs.image_tensor.clone()
    _, C, H, W = x.shape
    cell_h, cell_w = max(1, H // grid), max(1, W // grid)
    fill = x.mean(dim=(2, 3), keepdim=True)
    heat = torch.zeros((grid, grid), dtype=torch.float32, device=x.device)
    for i in range(grid):
        for j in range(grid):
            occl = x.clone()
            occl[:, :, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w] = fill
            with torch.no_grad():
                out = vlm_wrapper.model(
                    input_ids=temp_inputs.input_ids,
                    images=occl,
                    labels=temp_inputs.targets,
                    return_dict=True,
                )
            loss = float(getattr(out, "loss", base_loss))
            heat[i, j] = max(0.0, loss - base_loss)
    return heat


@pyrallis.wrap()
def main(cfg: MyVLMArtConfig, images: Optional[List[str]] = None, prompt: Optional[str] = None):
    assert cfg.vlm_type == VLMType.LLAVA
    assert cfg.personalization_task == PersonalizationTask.CAPTIONING

    vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
    images_root = Path(cfg.data_root)
    if images is None or len(images) == 0:
        images = _load_images(images_root, dataset_json=str(getattr(cfg, "dataset_json", "wikiart_5artists_dataset.json")))
    cfg_prompt = getattr(cfg, "prompt", None)
    prompt_mode = (prompt if prompt is not None else (cfg_prompt if cfg_prompt is not None else "natural")).strip().lower()
    lim = int(getattr(cfg, "limit", 0))
    if lim > 0:
        images = images[:lim]
    print(f"Reasoning baseline | images={len(images)} | mode={prompt_mode} | limit={lim}")

    def run_once(structured: bool) -> Dict[str, Dict[str, Any]]:
        outputs: Dict[str, Dict[str, Any]] = {}
        for i, img in enumerate(images):
            p = Path(img)
            try:
                print(f"[{i+1}/{len(images)}] {p.name}")
            except Exception:
                pass
            if structured:
                prompt_s = build_prompt_structured({"labels_per_dim": {"artist": [], "style": [], "genre": []}}, {"sentinel_start": "<BEGIN_JSON>", "sentinel_end": "<END_JSON>"})
                prompts = [prompt_s]
                try:
                    print("Prompt mode=STRUCTURED")
                except Exception:
                    pass
            else:
                prompts = get_prompts({"labels_per_dim": {"artist": [], "style": [], "genre": []}})
                try:
                    print("Prompt mode=NATURAL")
                except Exception:
                    pass
            res: Dict[str, str] = {}
            for prompt in prompts:
                inputs = vlm.preprocess(image_path=p, prompt=prompt)
                with torch.cuda.amp.autocast():
                    out = vlm.model.generate(
                        inputs=inputs['input_ids'],
                        images=inputs['image_tensor'],
                        do_sample=True if getattr(vlm, "temperature", 0.2) > 0 else False,
                        temperature=getattr(vlm, "temperature", 0.2),
                        top_p=getattr(vlm, "top_p", 0.7),
                        stopping_criteria=[inputs['stopping_criteria']],
                        max_new_tokens=int(getattr(cfg, "max_reason_tokens", 200)),
                        output_attentions=False,
                        return_dict_in_generate=True,
                    )
                text = vlm.processor.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
                res[prompt] = text
            out_entry: Dict[str, Any] = {"__meta__": {"activation": {}}}
            out_entry.update(res)
            outputs[str(p)] = out_entry
            if structured and bool(getattr(cfg, "save_saliency", False)):
                source = str(getattr(cfg, "saliency_source", "llava")).strip().lower()
                if source in ("llava", "llava_structured", "both"):
                    try:
                        s_prompt = prompt_s
                        s_text = res.get(s_prompt, "")
                        tmp_inputs = vlm.prepare_inputs(image=Image.open(p), prompt=s_prompt, target=s_text)
                        heat = _llava_occlusion_saliency(vlm, tmp_inputs, grid=int(getattr(cfg, "saliency_grid", 12)))
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
        return outputs

    if prompt_mode == "both":
        out_nat = run_once(structured=False)
        out_struct = run_once(structured=True)
        outputs = {}
        keys = set(out_nat.keys()) | set(out_struct.keys())
        for k in keys:
            out = {}
            for d in [out_nat.get(k, {}), out_struct.get(k, {})]:
                out.update(d)
            outputs[k] = out
    else:
        outputs = run_once(structured=(prompt_mode == "structured"))

    out_dir = cfg.output_root / cfg.concept_name / f"seed_{cfg.seed}" / "reasoning_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reasoning_baseline.json").open("w") as f:
        json.dump(outputs, f, indent=4, sort_keys=True)
    print(f"Saved baseline reasoning to: {out_dir / 'reasoning_baseline.json'}")


if __name__ == "__main__":
    main()
