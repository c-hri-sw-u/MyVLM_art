import os
import json
from pathlib import Path
from typing import Dict, Any, List

import requests


def call_openrouter(messages: List[Dict[str, str]], model: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('open_router_api_key', '')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
        "temperature": 0.0,
        "top_p": 0.9,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def synthesize_targets(samples: List[Dict[str, Any]], prompt_builder, target_builder, model: str) -> Dict[str, str]:
    cache = {}
    for s in samples:
        labels = s["labels_per_dim"]
        prompt = prompt_builder(labels, s.get("concept_signals"), mode="train_semi_structured")
        messages = [{"role": "user", "content": prompt}]
        content = call_openrouter(messages, model=model)
        # Optional post-filtering to ensure keys presence; fall back to deterministic target
        if all(k in content.lower() for k in ["artist:", "style:", "genre:", "medium:"]):
            cache[s["image_path"]] = content
        else:
            cache[s["image_path"]] = target_builder(labels, s.get("concept_signals"), mode="train_semi_structured")
    return cache


def save_targets(cache: Dict[str, str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

