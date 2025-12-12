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
