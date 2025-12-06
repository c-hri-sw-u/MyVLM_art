'''
python concept_graph/validation/validate_signals.py \
  --dataset_json data/dataset/wikiart_5artists_dataset.json \
  --signals_artist_json artifacts/concept_signals_artist.json \
  --signals_style_json artifacts/concept_signals_style.json \
  --signals_genre_json artifacts/concept_signals_genre.json \
  --ckpt_artist artifacts/prototypes_artist_trained.pt \
  --ckpt_style artifacts/prototypes_style_trained.pt \
  --ckpt_genre artifacts/prototypes_genre_trained.pt \
  --threshold_mode similarity --threshold 0.75 \
  --backoff_delta 0.05 --topk_per_dim 2 \
  --budget 3 --fairness true \
  --output artifacts/validation_metrics.json
'''


import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch


def _normalize_name(x: str) -> str:
    return x.strip().lower().replace("_", " ").replace("-", " ")


def _load_idx_to_concept(ckpt: Optional[Path], dim: str) -> Optional[List[str]]:
    if ckpt is None:
        return None
    try:
        payload = torch.load(ckpt, map_location="cpu")
        mapping = payload.get("idx_to_concept", {})
        if isinstance(mapping, dict):
            if dim in mapping:
                return [str(s) for s in mapping[dim]]
            if isinstance(mapping, list):
                return [str(s) for s in mapping]
        elif isinstance(mapping, list):
            return [str(s) for s in mapping]
    except Exception:
        return None
    return None


def _read_dataset(dataset_json: Path) -> List[Dict[str, Any]]:
    with dataset_json.open("r") as f:
        return json.load(f)


def _read_signals_json(signals_json: Path, dim: str) -> List[Dict[int, Tuple[float, float]]]:
    with signals_json.open("r") as f:
        items = json.load(f)
    out: List[Dict[int, Tuple[float, float]]] = []
    for rec in items:
        sig = rec.get(dim, {})
        cur: Dict[int, Tuple[float, float]] = {}
        for k_str, vec in sig.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            if isinstance(vec, list) and len(vec) >= 2:
                d = float(vec[0])
                s = float(vec[1])
                cur[k] = (d, s)
        out.append(cur)
    return out


def _select_per_dim(sig: Dict[int, Tuple[float, float]], threshold_mode: str, threshold: float, backoff_delta: float, topk_per_dim: int) -> Tuple[List[Tuple[int, float]], bool, int]:
    activated: List[Tuple[int, float]] = []
    for k, (d, s) in sig.items():
        if threshold_mode == "distance":
            if d <= threshold:
                activated.append((k, 1.0 - d))
        else:
            if s >= threshold:
                activated.append((k, s))
    activated.sort(key=lambda t: t[1], reverse=True)
    used_backoff = False
    activated_pre = len(activated)
    if len(activated) == 0 and backoff_delta > 0.0:
        candidates: List[Tuple[int, float]] = []
        for k, (d, s) in sig.items():
            if threshold_mode == "distance":
                if d <= threshold + backoff_delta:
                    candidates.append((k, 1.0 - d))
            else:
                if s >= max(0.0, threshold - backoff_delta):
                    candidates.append((k, s))
        candidates.sort(key=lambda t: t[1], reverse=True)
        if len(candidates) > 0:
            activated = [candidates[0]]
            used_backoff = True
    if topk_per_dim > 0 and len(activated) > topk_per_dim:
        activated = activated[:topk_per_dim]
    return activated, used_backoff, activated_pre


def _fair_budget_merge(per_dim_sel: Dict[str, List[Tuple[int, float]]], budget: int, fairness: bool, priority: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    if budget <= 0:
        return per_dim_sel
    dims = list(per_dim_sel.keys())
    selected_total = 0
    out: Dict[str, List[Tuple[int, float]]] = {d: [] for d in dims}
    if fairness:
        for d in priority:
            opts = per_dim_sel.get(d, [])
            if len(opts) > 0 and selected_total < budget:
                out[d].append(opts[0])
                selected_total += 1
    remaining: List[Tuple[str, int, float]] = []
    for d in dims:
        start = len(out[d])
        for i in range(start, len(per_dim_sel[d])):
            remaining.append((d, i, per_dim_sel[d][i][1]))
    remaining.sort(key=lambda t: t[2], reverse=True)
    for d, i, score in remaining:
        if selected_total >= budget:
            break
        out[d].append(per_dim_sel[d][i])
        selected_total += 1
    return out


def validate(
    dataset_json: Path,
    signals_artist_json: Path,
    signals_style_json: Path,
    signals_genre_json: Path,
    ckpt_artist: Optional[Path],
    ckpt_style: Optional[Path],
    ckpt_genre: Optional[Path],
    threshold_mode: str,
    threshold: float,
    backoff_delta: float,
    topk_per_dim: int,
    budget: int,
    fairness: bool,
    priority: List[str],
) -> Dict[str, Any]:
    records = _read_dataset(dataset_json)
    sig_artist = _read_signals_json(signals_artist_json, "artist")
    sig_style = _read_signals_json(signals_style_json, "style")
    sig_genre = _read_signals_json(signals_genre_json, "genre")
    n = min(len(records), len(sig_artist), len(sig_style), len(sig_genre))
    idx_to_concept_artist = _load_idx_to_concept(ckpt_artist, "artist")
    idx_to_concept_style = _load_idx_to_concept(ckpt_style, "style")
    idx_to_concept_genre = _load_idx_to_concept(ckpt_genre, "genre")
    dims = ["artist", "style", "genre"]
    activation_counts = {d: 0 for d in dims}
    activation_pre_counts = {d: 0 for d in dims}
    backoff_counts = {d: 0 for d in dims}
    coverage_counts = {d: 0 for d in dims}
    precision_counts = {d: 0 for d in dims}
    precision_total = {d: 0 for d in dims}
    false_injection_counts = {d: 0 for d in dims}
    false_injection_total = {d: 0 for d in dims}
    unknown_injection_counts = {d: 0 for d in dims}
    unknown_injection_total = {d: 0 for d in dims}
    budget_hit = 0
    overflow_precut = 0
    for i in range(n):
        r = records[i]
        gt_concepts = r.get("concepts", {})
        gt_artist = _normalize_name(gt_concepts.get("artist", r.get("artist", "")))
        gt_style = _normalize_name(gt_concepts.get("style", ""))
        gt_genre = _normalize_name(gt_concepts.get("genre", ""))
        per_dim_sel_raw: Dict[str, List[Tuple[int, float]]] = {}
        used_backoff_flags: Dict[str, bool] = {}
        activated_pre_sizes: Dict[str, int] = {}
        a, b, pre = _select_per_dim(sig_artist[i], threshold_mode, threshold, backoff_delta, topk_per_dim)
        per_dim_sel_raw["artist"] = a
        used_backoff_flags["artist"] = b
        activated_pre_sizes["artist"] = pre
        a, b, pre = _select_per_dim(sig_style[i], threshold_mode, threshold, backoff_delta, topk_per_dim)
        per_dim_sel_raw["style"] = a
        used_backoff_flags["style"] = b
        activated_pre_sizes["style"] = pre
        a, b, pre = _select_per_dim(sig_genre[i], threshold_mode, threshold, backoff_delta, topk_per_dim)
        per_dim_sel_raw["genre"] = a
        used_backoff_flags["genre"] = b
        activated_pre_sizes["genre"] = pre
        total_candidates = sum(len(per_dim_sel_raw[d]) for d in dims)
        if budget > 0 and total_candidates > budget:
            overflow_precut += 1
        per_dim_sel = _fair_budget_merge(per_dim_sel_raw, budget, fairness, priority)
        selected_total = sum(len(per_dim_sel[d]) for d in dims)
        if budget > 0 and selected_total == budget:
            budget_hit += 1
        for d in dims:
            if activated_pre_sizes[d] > 0:
                activation_pre_counts[d] += 1
            if used_backoff_flags[d]:
                backoff_counts[d] += 1
            if len(per_dim_sel[d]) > 0:
                activation_counts[d] += 1
                coverage_counts[d] += 1
        def name_of(dim: str, idx: int) -> Optional[str]:
            if dim == "artist" and idx_to_concept_artist and idx < len(idx_to_concept_artist):
                return _normalize_name(idx_to_concept_artist[idx])
            if dim == "style" and idx_to_concept_style and idx < len(idx_to_concept_style):
                return _normalize_name(idx_to_concept_style[idx])
            if dim == "genre" and idx_to_concept_genre and idx < len(idx_to_concept_genre):
                return _normalize_name(idx_to_concept_genre[idx])
            return None
        pairs_gt = {"artist": gt_artist, "style": gt_style, "genre": gt_genre}
        for d in dims:
            if len(per_dim_sel[d]) > 0:
                top_idx = per_dim_sel[d][0][0]
                pred_name = name_of(d, top_idx)
                gt_name = pairs_gt[d]
                if pred_name is not None:
                    if len(gt_name) > 0:
                        precision_total[d] += 1
                        false_injection_total[d] += 1
                        if pred_name == gt_name:
                            precision_counts[d] += 1
                        else:
                            false_injection_counts[d] += 1
                    else:
                        unknown_injection_total[d] += 1
                        unknown_injection_counts[d] += 1
    eps = 1e-9
    result = {
        "samples": n,
        "activation_rate": {d: activation_counts[d] / max(1, n) for d in dims},
        "activation_pre_backoff_rate": {d: activation_pre_counts[d] / max(1, n) for d in dims},
        "backoff_usage_rate": {d: backoff_counts[d] / max(1, n) for d in dims},
        "coverage_rate": {d: coverage_counts[d] / max(1, n) for d in dims},
        "precision_top1": {d: (precision_counts[d] / max(eps, precision_total[d])) for d in dims},
        "false_injection_rate_known_gt": {d: (false_injection_counts[d] / max(eps, false_injection_total[d])) for d in dims},
        "injection_when_unknown_gt": {d: (unknown_injection_counts[d] / max(1, n)) for d in dims},
        "budget_hit_rate": (budget_hit / max(1, n)),
        "budget_overflow_precut_rate": (overflow_precut / max(1, n)),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--signals_artist_json", type=str, required=True)
    parser.add_argument("--signals_style_json", type=str, required=True)
    parser.add_argument("--signals_genre_json", type=str, required=True)
    parser.add_argument("--ckpt_artist", type=str, required=False)
    parser.add_argument("--ckpt_style", type=str, required=False)
    parser.add_argument("--ckpt_genre", type=str, required=False)
    parser.add_argument("--threshold_mode", type=str, default="similarity")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--backoff_delta", type=float, default=0.05)
    parser.add_argument("--topk_per_dim", type=int, default=2)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--fairness", type=str, default="true")
    parser.add_argument("--priority", type=str, default="artist,style,genre")
    parser.add_argument("--output", type=str, default="artifacts/validation_metrics.json")
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    signals_artist_json = Path(args.signals_artist_json)
    signals_style_json = Path(args.signals_style_json)
    signals_genre_json = Path(args.signals_genre_json)
    ckpt_artist = Path(args.ckpt_artist) if args.ckpt_artist else None
    ckpt_style = Path(args.ckpt_style) if args.ckpt_style else None
    ckpt_genre = Path(args.ckpt_genre) if args.ckpt_genre else None
    fairness = str(args.fairness).lower() in ["1", "true", "yes", "y"]
    priority = [s.strip() for s in args.priority.split(",") if s.strip()]

    metrics = validate(
        dataset_json=dataset_json,
        signals_artist_json=signals_artist_json,
        signals_style_json=signals_style_json,
        signals_genre_json=signals_genre_json,
        ckpt_artist=ckpt_artist,
        ckpt_style=ckpt_style,
        ckpt_genre=ckpt_genre,
        threshold_mode=args.threshold_mode.strip().lower(),
        threshold=float(args.threshold),
        backoff_delta=float(args.backoff_delta),
        topk_per_dim=int(args.topk_per_dim),
        budget=int(args.budget),
        fairness=fairness,
        priority=priority,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()