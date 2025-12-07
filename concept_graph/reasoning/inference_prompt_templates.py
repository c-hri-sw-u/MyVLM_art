"""
简要介绍 (ZH):
  解释性推理的提示词模板库。根据已激活的多粒度概念（艺术家/风格/题材/媒介），
  自动生成用于 LLaVA 的对话式提示词，用于“解释为何是某艺术家”或“描述作品特征”。

Overview (EN):
  Prompt templates for explainable reasoning. Given activated multi-granular concepts, compose conversational
  prompts for LLaVA to produce natural language justifications and descriptive outputs.

Status:
  - Natural and structured prompts implemented; more templates can be added.

Remaining Work:
  1) Add variants: classification explanation, style-only description, genre/medium hybrids, counterfactuals.
  2) Improve templating and length control; expose options via config.
  3) Integrate with `reasoning_runner.py` for multi-template selection.
"""

from typing import Any, Dict, Optional


def _get_dims(x: Dict[str, Any]) -> Dict[str, Any]:
    d = x.get("labels_per_dim", x)
    artist = d.get("artist") or "Unknown"
    style = d.get("style") or "Unknown"
    genre = d.get("genre") or "Unknown"
    media = d.get("media") or []
    if isinstance(media, str):
        media_list = [media] if media else []
    else:
        media_list = media
    media_str = ", ".join(media_list) if media_list else "Unknown"
    return {"artist": artist, "style": style, "genre": genre, "media": media_str}


def build_prompt(activated_concepts: Dict[str, Any], mode: str = "natural", structured_cfg: Optional[Dict[str, Any]] = None) -> str:
    if mode == "structured":
        return build_prompt_structured(activated_concepts, structured_cfg)
    return build_prompt_natural(activated_concepts)


def build_prompt_structured(activated_concepts: Dict[str, Any], structured_cfg: Optional[Dict[str, Any]] = None) -> str:
    dims = _get_dims(activated_concepts)
    start = (structured_cfg or {}).get("sentinel_start", "<BEGIN_JSON>")
    end = (structured_cfg or {}).get("sentinel_end", "<END_JSON>")
    return (
        "Between "
        + start
        + " and "
        + end
        + ", return only a compact JSON with keys artist, style, genre, evidence. "
        "Use 'Unknown' if uncertain. In 'evidence', write 2–3 concise sentences explaining your reasoning. Do not include any text outside the JSON."
    )


def build_prompt_natural(activated_concepts: Dict[str, Any]) -> str:
    dims = _get_dims(activated_concepts)
    return (
        "Describe this painting in one cohesive paragraph. "
        "Naturally mention the artist, style, and genre in the prose (no JSON, no key-value lines). "
        "Then briefly justify your identification by referring to brushwork, palette, and composition."
    )


def get_prompts(activated_concepts: Dict[str, Any]):
    return [build_prompt_natural(activated_concepts)]
