from typing import Dict, Any


def build_prompt(labels_per_dim: Dict[str, Any], concept_signals: Dict[str, Any] = None, mode: str = "train_semi_structured", structured_cfg: Dict[str, Any] = None) -> str:
    artist = labels_per_dim.get("artist") or "Unknown"
    style = labels_per_dim.get("style") or "Unknown"
    genre = labels_per_dim.get("genre") or "Unknown"
    media = labels_per_dim.get("media") or []
    if isinstance(media, str):
        media_list = [media] if media else []
    else:
        media_list = media
    media_str = ", ".join(media_list) if media_list else "Unknown"

    return (
        "You are an art expert. Produce a deterministic, semi-structured English description covering the following keys: "
        "artist, style, genre, medium, brushwork, color_palette, composition. "
        "Output as key-value pairs (no JSON), one key per line, with short, precise phrases. "
        "Use exactly these keys and avoid additional commentary. "
        f"Target Concepts: artist={artist}; style={style}; genre={genre}; medium={media_str}. "
        "If any field is unknown, set it to 'Unknown'."
    )


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
        f"medium: {media_str}",
        "brushwork: thick impasto; energetic strokes",  # canonical cues for Post-Impressionism-like
        "color_palette: vivid contrasts; saturated primaries",
        "composition: rhythmic forms; balanced asymmetry",
    ]
    return "\n".join(lines)
