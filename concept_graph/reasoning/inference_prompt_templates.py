"""
简要介绍 (ZH):
  解释性推理的提示词模板库。根据已激活的多粒度概念（艺术家/风格/题材/媒介），
  自动生成用于 LLaVA 的对话式提示词，用于“解释为何是某艺术家”或“描述作品特征”。

Overview (EN):
  Prompt templates for explainable reasoning. Given activated multi-granular concepts, compose conversational
  prompts for LLaVA to produce natural language justifications and descriptive outputs.

TODOs (详细):
  1) 设计多种模板：分类解释、风格描述、题材/媒介结合说明、反事实对比等
  2) 模板填充：将激活概念按维度插入占位符，控制长度与可读性
  3) 与 reasoning_runner.py 对接：提供函数 get_prompts(activated_concepts)
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
        "Return a JSON object with exactly these keys: artist, style, genre, medium, evidence, summary. "
        "Wrap the JSON between "
        + start
        + " and "
        + end
        + ". "
        "Use concise English. Set evidence to a list of three short clauses describing brushwork, color palette, and composition. "
        "Prefer the following activated concepts when assigning values. If uncertain, use 'Unknown'. "
        f"Activated: artist='{dims['artist']}', style='{dims['style']}', genre='{dims['genre']}', medium='{dims['media']}'. "
        "Do not include any text before or after the JSON."
    )


def build_prompt_natural(activated_concepts: Dict[str, Any]) -> str:
    dims = _get_dims(activated_concepts)
    summary = (
        f"ConceptSummary: Artist={dims['artist']}; Style={dims['style']}; Genre={dims['genre']}; Medium={dims['media']}"
    )
    return (
        "Explain the attribution and stylistic reasoning of this painting in clear English. "
        "Discuss brushwork, color palette, and composition, and relate them to the likely artist, style, genre, and medium. "
        "Write 120–180 words in two coherent paragraphs. "
        + summary
    )


def get_prompts(activated_concepts: Dict[str, Any]):
    return [build_prompt_natural(activated_concepts)]
