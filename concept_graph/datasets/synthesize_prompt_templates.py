from typing import Dict, Any


def build_prompt(labels_per_dim: Dict[str, Any], concept_signals: Dict[str, Any] = None, mode: str = "train_semi_structured", structured_cfg: Dict[str, Any] = None, reveal_labels: bool = True, include_reasoning: bool = False) -> str:
    artist = labels_per_dim.get("artist") or "Unknown"
    style = labels_per_dim.get("style") or "Unknown"
    genre = labels_per_dim.get("genre") or "Unknown"
    media = labels_per_dim.get("media") or []
    if isinstance(media, str):
        media_list = [media] if media else []
    else:
        media_list = media
    media_str = ", ".join(media_list) if media_list else "Unknown"

    if not include_reasoning:
        prefix = (
            "You are an art expert. Produce a deterministic, semi-structured English description with ONLY these keys: "
            "artist, style, genre. "
            "Output as key-value pairs (no JSON), one key per line, short and precise. "
            "Avoid any extra commentary or additional keys. "
        )
        hint = f"Target Concepts: artist={artist}; style={style}; genre={genre}. " if reveal_labels else ""
        suffix = "If uncertain, set the value to 'Unknown'."
        return prefix + hint + suffix

    ks_start = (structured_cfg or {}).get("keys_start", "[BEGIN_KEYS]")
    ks_end = (structured_cfg or {}).get("keys_end", "[END_KEYS]")
    rs_start = (structured_cfg or {}).get("reason_start", "[BEGIN_REASON]")
    rs_end = (structured_cfg or {}).get("reason_end", "[END_REASON]")
    guide = (
        "Return exactly two blocks delimited by sentinels. "
        f"First block between {ks_start} and {ks_end} must contain ONLY three lines: 'artist: ...', 'style: ...', 'genre: ...'. "
        f"Second block between {rs_start} and {rs_end} must contain 2–3 short sentences of visual reasoning. "
        "Do not add any text before, between, or after the blocks. "
    )
    hint = f"Target Concepts: artist={artist}; style={style}; genre={genre}. " if reveal_labels else ""
    return guide + hint


def build_target(labels_per_dim: Dict[str, Any], concept_signals: Dict[str, Any] = None, mode: str = "train_semi_structured") -> str:
    artist = labels_per_dim.get("artist") or "Unknown"
    style = labels_per_dim.get("style") or "Unknown"
    genre = labels_per_dim.get("genre") or "Unknown"
    media = labels_per_dim.get("media") or []
    if isinstance(media, str):
        media_list = [media] if media else []
    else:
        media_list = media
    media_str = ", ".join(media_list) if media_list else "Unknown"

    # Deterministic semi-structured target for supervision
    lines = [
        f"artist: {artist}",
        f"style: {style}",
        f"genre: {genre}",
    ]
    return "\n".join(lines)
