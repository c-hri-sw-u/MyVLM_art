import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import requests
import random
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

def _load_env_file():
    if ENV_PATH.exists():
        try:
            with ENV_PATH.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k not in os.environ or not os.environ.get(k):
                        os.environ[k] = v
        except Exception:
            pass

# Support running as a script (python /path/to/data_synthesize.py) without -m
try:
    from concept_graph.datasets.synthesize_prompt_templates import build_prompt, build_target
    from concept_graph.datasets.concept_graph_dataset import ConceptGraphDataset
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))
    from concept_graph.datasets.synthesize_prompt_templates import build_prompt, build_target
    from concept_graph.datasets.concept_graph_dataset import ConceptGraphDataset


def call_openrouter(messages: List[Dict[str, Any]], model: str, temperature: float = 0.0) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    _load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("open_router_api_key", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY/open_router_api_key not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "MyVLM_art Synthesis",
    }
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
        "temperature": float(temperature),
        "top_p": 0.9,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _encode_image_base64(image_path: Path) -> str:
    import base64
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def synthesize_targets(samples: List[Dict[str, Any]], prompt_builder, target_builder, model: str, reveal_labels: bool = True, include_reasoning: bool = False, structured_cfg: Dict[str, Any] = None):
    cache_keys = {}
    cache_reason = {}
    for s in samples:
        labels = s["labels_per_dim"]
        prompt = prompt_builder(labels, s.get("concept_signals"), mode="train_semi_structured", reveal_labels=reveal_labels, include_reasoning=include_reasoning, structured_cfg=structured_cfg)
        image_path = Path(s["image_path"]).resolve()
        image_data_uri = _encode_image_base64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]
        }]
        temp = 0.2 if include_reasoning else 0.0
        content = call_openrouter(messages, model=model, temperature=temp)
        # Post-filtering: extract keys block if reasoning present
        if include_reasoning:
            ks_start = (structured_cfg or {}).get("keys_start", "[BEGIN_KEYS]")
            ks_end = (structured_cfg or {}).get("keys_end", "[END_KEYS]")
            rs_start = (structured_cfg or {}).get("reason_start", "[BEGIN_REASON]")
            rs_end = (structured_cfg or {}).get("reason_end", "[END_REASON]")
            start_idx = content.find(ks_start)
            end_idx = content.find(ks_end)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                keys_block = content[start_idx + len(ks_start):end_idx].strip()
                content_keys = keys_block
            else:
                content_keys = content
            r_start = content.find(rs_start)
            r_end = content.find(rs_end)
            if r_start != -1 and r_end != -1 and r_end > r_start:
                reason_block = content[r_start + len(rs_start):r_end].strip()
                cache_reason[s["image_path"]] = reason_block
            else:
                cache_reason[s["image_path"]] = ""
        else:
            content_keys = content
        # Ensure keys presence; fallback deterministic
        lines_ok = all(k in (content_keys or "").lower() for k in ["artist:", "style:", "genre:"])
        cache_keys[s["image_path"]] = content_keys if lines_ok else target_builder(labels, s.get("concept_signals"), mode="train_semi_structured")
    return cache_keys, cache_reason


def save_targets(cache: Dict[str, str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _default_paths():
    dataset_path = Path("/home/ubuntu/MyVLM_art/data/dataset/wikiart_5artists_dataset.json")
    images_root = Path("/home/ubuntu/MyVLM_art/data/dataset/")
    model = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
    out_path = Path("/home/ubuntu/MyVLM_art/artifacts/synth_targets_test.json")
    return dataset_path, images_root, model, out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize target texts with OpenRouter (multimodal)")
    parser.add_argument("--reveal_labels", type=str, default="true", help="Whether to reveal ground-truth labels in the prompt (true/false)")
    parser.add_argument("--reasoning", type=str, default="false", help="Whether to include a reasoning block with sentinels (true/false)")
    args = parser.parse_args()
    reveal = str(args.reveal_labels).lower() in ["1", "true", "yes", "y"]
    include_reasoning = str(args.reasoning).lower() in ["1", "true", "yes", "y"]
    structured_cfg = {"keys_start": "[BEGIN_KEYS]", "keys_end": "[END_KEYS]", "reason_start": "[BEGIN_REASON]", "reason_end": "[END_REASON]"}

    dataset_path, images_root, model, out_path = _default_paths()
    if not dataset_path.exists():
        print(f"Dataset JSON not found: {dataset_path}")
        return
    ds = ConceptGraphDataset(dataset_path=dataset_path, images_root=images_root)
    if len(ds) == 0:
        print("Dataset is empty")
        return
    idx = random.randrange(len(ds))
    sample = ds[idx]
    labels = sample["labels_per_dim"]
    signals = sample.get("concept_signals")
    prompt = build_prompt(labels, signals, mode="train_semi_structured", reveal_labels=reveal, include_reasoning=include_reasoning, structured_cfg=structured_cfg)
    print("Prompt:\n" + prompt)
    try:
        cache_keys, cache_reason = synthesize_targets([sample], build_prompt, build_target, model=model, reveal_labels=reveal, include_reasoning=include_reasoning, structured_cfg=structured_cfg)
        keys_out = next(iter(cache_keys.values()))
        print("Keys Block:\n" + keys_out)
        if include_reasoning:
            reason_out = next(iter(cache_reason.values()))
            if reason_out:
                print("Reason Block:\n" + reason_out)
            else:
                print("Reason Block:\n<empty>")
    except Exception as e:
        print(f"Synthesis failed: {e}")
        fallback = build_target(labels, signals, mode="train_semi_structured")
        cache_keys = {sample["image_path"]: fallback}
        print("Fallback Keys:\n" + fallback)
    save_targets(cache_keys, out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
