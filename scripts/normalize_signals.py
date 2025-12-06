import argparse
import json
import math
from pathlib import Path


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize_list(vals, mode: str, temperature: float, clip: bool = True):
    out = []
    if clip:
        vals = [max(-1.0, min(1.0, float(v))) for v in vals]
    if mode == "zscore":
        n = max(1, len(vals))
        mean = sum(vals) / n
        var = sum((v - mean) * (v - mean) for v in vals) / n
        std = math.sqrt(var)
        for v in vals:
            z = (v - mean) / (std + 1e-6)
            p = _sigmoid(z)
            p = max(0.0, min(1.0, p))
            if temperature != 1.0:
                p = p ** (1.0 / max(1e-6, temperature))
            out.append(float(max(0.0, min(1.0, p))))
        return out
    if mode == "zero_one":
        for v in vals:
            s2 = 0.5 * (v + 1.0)
            s2 = max(0.0, min(1.0, s2))
            if temperature != 1.0:
                s2 = s2 ** (1.0 / max(1e-6, temperature))
            out.append(float(max(0.0, min(1.0, s2))))
        return out
    if mode == "clamp_zero":
        for v in vals:
            s2 = max(0.0, v)
            s2 = max(0.0, min(1.0, s2))
            if temperature != 1.0:
                s2 = s2 ** (1.0 / max(1e-6, temperature))
            out.append(float(max(0.0, min(1.0, s2))))
        return out
    for v in vals:
        s2 = v
        s2 = max(0.0, min(1.0, s2))
        if temperature != 1.0:
            s2 = s2 ** (1.0 / max(1e-6, temperature))
        out.append(float(max(0.0, min(1.0, s2))))
    return out


def _flatten_pair(arr):
    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
        return arr[0]
    return arr


def normalize_json_file(path: Path, normalize: str, temperature: float, inplace: bool = True) -> Path:
    with path.open("r") as f:
        records = json.load(f)
    dims = ["artist", "style", "genre"]
    for rec in records:
        for dim in dims:
            if dim not in rec:
                continue
            sig_map = rec[dim]
            if not isinstance(sig_map, dict) or len(sig_map) == 0:
                continue
            raw = []
            keys = list(sig_map.keys())
            for k in keys:
                arr = _flatten_pair(sig_map[k])
                if not isinstance(arr, list) or len(arr) < 2:
                    continue
                s = float(arr[1])
                raw.append(s)
            if len(raw) == 0:
                continue
            normed = _normalize_list(raw, mode=normalize, temperature=temperature, clip=True)
            for idx, k in enumerate(keys):
                arr = _flatten_pair(sig_map[k])
                if not isinstance(arr, list) or len(arr) < 2:
                    continue
                s_norm = float(normed[idx])
                d_norm = float(1.0 - s_norm)
                sig_map[k] = [d_norm, s_norm]
    out_path = path if inplace else path.with_name(path.stem + ".normalized" + path.suffix)
    with out_path.open("w") as f:
        json.dump(records, f)
    return out_path


def main():
    p_artist = Path("/Users/chriswu/Documents/25Fall/10-623 GenAI/team_proj/MyVLM_art/artifacts/concept_signals_artist.json")
    p_style = Path("/Users/chriswu/Documents/25Fall/10-623 GenAI/team_proj/MyVLM_art/artifacts/concept_signals_style.json")
    p_genre = Path("/Users/chriswu/Documents/25Fall/10-623 GenAI/team_proj/MyVLM_art/artifacts/concept_signals_genre.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist_json", type=str, default=str(p_artist))
    parser.add_argument("--style_json", type=str, default=str(p_style))
    parser.add_argument("--genre_json", type=str, default=str(p_genre))
    parser.add_argument("--normalize", type=str, default="zscore")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--inplace", type=str, default="true")
    args = parser.parse_args()
    normalize = args.normalize.strip().lower()
    temperature = float(args.temperature)
    inplace = str(args.inplace).lower() in ["1", "true", "yes", "y"]
    paths = []
    for p in [args.artist_json, args.style_json, args.genre_json]:
        if p:
            paths.append(Path(p))
    outs = []
    for p in paths:
        if p.exists():
            outs.append(normalize_json_file(p, normalize=normalize, temperature=temperature, inplace=inplace))
    for o in outs:
        print(str(o))


if __name__ == "__main__":
    main()

