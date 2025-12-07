import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from typing import Dict, Any, List

import pyrallis

from configs.myvlm_art_config import MyVLMArtConfig
from myvlm_art.evaluation.metrics import (
    coverage_score,
    save_results,
    per_class_prf,
    macro_weighted_f1,
    multi_label_all_correct,
    per_concept_breakdown,
    recall_at_k,
    coverage_per_concept,
    best_label_match,
    activation_precision,
    canon_str,
    build_alias_maps,
)


def _load_reasoning(out_dir: Path, override_path: str = "") -> Dict[str, Dict[str, str]]:
    p = Path(override_path) if override_path else (out_dir / "reasoning.json")
    return json.load(p.open()) if p.exists() else {}


def _resolve_dataset_json(images_root: Path, preferred: str = "") -> Path:
    if preferred:
        cand = images_root / preferred
        if cand.exists():
            return cand
    pats = ["*wikiart*5artists*.json", "*dataset*.json", "*test*.json", "*.json"]
    seen = set()
    cands: List[Path] = []
    for pat in pats:
        for p in sorted(images_root.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True):
            if str(p) in seen:
                continue
            seen.add(str(p))
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No dataset JSON found under {images_root}")
    return cands[0]


def _load_dataset_labels(images_root: Path, dataset_json: str = "") -> Dict[str, Dict[str, str]]:
    p = _resolve_dataset_json(images_root, preferred=dataset_json)
    records = json.load(p.open())
    labels: Dict[str, Dict[str, str]] = {}
    for r in records:
        abs = str((images_root / r["image"]).resolve())
        concepts = r.get("concepts", {})
        labels[abs] = {
            "artist": concepts.get("artist", "Unknown"),
            "style": concepts.get("style", "Unknown"),
            "genre": concepts.get("genre", "Unknown"),
        }
    return labels


def _split_modes(reasoning: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    natural: Dict[str, str] = {}
    structured: Dict[str, str] = {}
    for img, entry in reasoning.items():
        if not entry:
            continue
        items = {k: v for k, v in entry.items() if k != "__meta__"}
        for k, v in items.items():
            key_l = (k or "").lower()
            # 结构化提示的更稳健识别：优先依据内容解析，其次依据显式标记
            is_struct_prompt = (
                key_l.startswith("return a json object")
                or key_l.startswith("between <begin_json>")
                or ("<begin_json>" in key_l)
                or ("<end_json>" in key_l)
            )
            # 如果提示明确写了“不使用 JSON/不使用键值行”，则视为自然提示
            is_explicit_natural = ("no json" in key_l) or ("no key-value" in key_l)
            parsed = _try_parse_structured(v)
            if parsed or (is_struct_prompt and not is_explicit_natural):
                structured[img] = v
            else:
                if img not in natural or len(v) > len(natural[img]):
                    natural[img] = v
    return {"natural": natural, "structured": structured}


def _try_parse_structured(text: str, start: str = "<BEGIN_JSON>", end: str = "<END_JSON>") -> Dict[str, Any]:
    def _clean(x: str) -> str:
        x = x.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        x = x.replace("“", '"').replace("”", '"').replace("’", "'")
        return x
    try:
        s = text.find(start)
        e = text.find(end)
        if s != -1:
            if e != -1 and e > s:
                payload = text[s + len(start):e].strip()
            else:
                r = text.rfind("}")
                if r != -1 and r > s:
                    payload = text[s + len(start):r + 1].strip()
                else:
                    payload = ""
        else:
            l = text.find("{")
            r = text.rfind("}")
            payload = text[l:r + 1].strip() if l != -1 and r != -1 and r > l else ""
        if payload:
            payload = _clean(payload)
            return json.loads(payload)
    except Exception:
        try:
            payload = _clean(payload)
            return json.loads(payload)
        except Exception:
            pass
    return {}


def _extract_label_from_text(text: str, candidates: List[str]) -> str:
    # 在自然文本中查找候选概念，优先选择更长的匹配；找不到则回退到最佳近似匹配
    tt = f" {canon_str(text)} "
    best = None
    best_score = -1
    for c in candidates:
        cc = canon_str(c)
        if not cc:
            continue
        if tt.find(f" {cc} ") != -1:
            score = len(cc.split())
            if score > best_score:
                best_score = score
                best = c
    if best is not None:
        return best
    # 回退：用文本整体与候选进行柔性匹配
    return best_label_match(text, candidates)


@pyrallis.wrap()
def main(cfg: MyVLMArtConfig):
    out_dir = cfg.output_root / cfg.concept_name / f"seed_{cfg.seed}" / "reasoning_outputs"
    images_root = Path(cfg.data_root)
    reasoning = _load_reasoning(out_dir, override_path=str(getattr(cfg, "input_json", "")))
    labels = _load_dataset_labels(images_root, dataset_json=str(getattr(cfg, "dataset_json", "")))

    modes = _split_modes(reasoning)
    natural_texts = list(modes["natural"].values())
    structured_texts = list(modes["structured"].values())
    # coverage：自然与结构化都计算
    nat_activated = [[labels.get(img, {}).get("artist", ""), labels.get(img, {}).get("style", ""), labels.get(img, {}).get("genre", "")] for img in modes["natural"].keys()]
    cov_natural = coverage_score(natural_texts, nat_activated) if natural_texts else 0.0
    str_activated = [[labels.get(img, {}).get("artist", ""), labels.get(img, {}).get("style", ""), labels.get(img, {}).get("genre", "")] for img in modes["structured"].keys()]
    cov_structured = coverage_score(structured_texts, str_activated) if structured_texts else 0.0

    # 构建维度候选词表（供自然/结构化统一使用）
    all_artists = sorted({v.get("artist", "Unknown") for v in labels.values()})
    all_styles = sorted({v.get("style", "Unknown") for v in labels.values()})
    all_genres = sorted({v.get("genre", "Unknown") for v in labels.values()})
    # 构建别名映射，提升匹配鲁棒性（如 Post‑Impressionist / van gogh / 复数形式）
    alias_maps = build_alias_maps(all_artists, all_styles, all_genres)

    def _apply_alias(dim: str, text: str, fallback: str) -> str:
        key = canon_str(text)
        # 连字符形式也尝试（已在 build 中加入）
        if key in alias_maps[dim]:
            return alias_maps[dim][key]
        key_dash = key.replace(" ", "-")
        if key_dash in alias_maps[dim]:
            return alias_maps[dim][key_dash]
        return fallback

    # 自然文本：从自由文本中抽取 artist/style/genre 的预测
    preds_nat: List[Dict[str, str]] = []
    labels_nat: List[Dict[str, str]] = []
    for img, text in modes["natural"].items():
        p_artist_raw = _extract_label_from_text(text, all_artists)
        p_style_raw = _extract_label_from_text(text, all_styles)
        p_genre_raw = _extract_label_from_text(text, all_genres)
        preds_nat.append({
            "artist": _apply_alias("artist", p_artist_raw, p_artist_raw),
            "style": _apply_alias("style", p_style_raw, p_style_raw),
            "genre": _apply_alias("genre", p_genre_raw, p_genre_raw),
        })
        labs = labels.get(img, {})
        labels_nat.append({
            "artist": labs.get("artist", "Unknown"),
            "style": labs.get("style", "Unknown"),
            "genre": labs.get("genre", "Unknown"),
        })
    acc_all_nat = multi_label_all_correct(preds_nat, labels_nat) if preds_nat else 0.0

    # 提取自然文本的各维度分类列表
    artists_nat = [l.get("artist", "Unknown") for l in labels_nat]
    preds_artists_nat = [p.get("artist", "Unknown") for p in preds_nat]
    styles_nat = [l.get("style", "Unknown") for l in labels_nat]
    preds_styles_nat = [p.get("style", "Unknown") for p in preds_nat]
    genres_nat = [l.get("genre", "Unknown") for l in labels_nat]
    preds_genres_nat = [p.get("genre", "Unknown") for p in preds_nat]
    per_cls_artist_nat = per_class_prf(preds_artists_nat, artists_nat) if artists_nat and preds_nat else {}
    per_cls_style_nat = per_class_prf(preds_styles_nat, styles_nat) if styles_nat and preds_nat else {}
    per_cls_genre_nat = per_class_prf(preds_genres_nat, genres_nat) if genres_nat and preds_nat else {}
    macro_f1_artist_nat, weighted_f1_artist_nat = macro_weighted_f1(per_cls_artist_nat, artists_nat) if per_cls_artist_nat else (0.0, 0.0)
    macro_f1_style_nat, weighted_f1_style_nat = macro_weighted_f1(per_cls_style_nat, styles_nat) if per_cls_style_nat else (0.0, 0.0)
    macro_f1_genre_nat, weighted_f1_genre_nat = macro_weighted_f1(per_cls_genre_nat, genres_nat) if per_cls_genre_nat else (0.0, 0.0)
    per_concept_artist_nat = per_concept_breakdown(preds_artists_nat, artists_nat) if preds_artists_nat else {}
    per_concept_style_nat = per_concept_breakdown(preds_styles_nat, styles_nat) if preds_styles_nat else {}
    per_concept_genre_nat = per_concept_breakdown(preds_genres_nat, genres_nat) if preds_genres_nat else {}

    # 可选：如果你使用结构化提示并能抽取 artist/style/genre 的预测，这里可以做多维准确率
    # 下面演示：以 GT 作为“预测占位”，你可以替换为真实解析结果
    preds_struct = []
    labels_struct = []
    for img, text in modes["structured"].items():
        parsed = _try_parse_structured(text)
        if parsed:
            pa = parsed.get("artist", "Unknown")
            ps = parsed.get("style", "Unknown")
            pg = parsed.get("genre", "Unknown")
        else:
            pa = _extract_label_from_text(text, all_artists)
            ps = _extract_label_from_text(text, all_styles)
            pg = _extract_label_from_text(text, all_genres)
        preds_struct.append({
            "artist": _apply_alias("artist", pa, pa),
            "style": _apply_alias("style", ps, ps),
            "genre": _apply_alias("genre", pg, pg),
        })
        labs = labels.get(img, {})
        labels_struct.append({
            "artist": labs.get("artist", "Unknown"),
            "style": labs.get("style", "Unknown"),
            "genre": labs.get("genre", "Unknown"),
        })
    acc_all = multi_label_all_correct(preds_struct, labels_struct) if preds_struct else 0.0

    # 示例分类 PRF（需要解析 artist 预测为类名列表）
    # 将结构化预测映射到最接近的词表项，提升变体容忍度
    preds_struct_mapped = []
    for p in preds_struct:
        preds_struct_mapped.append({
            "artist": best_label_match(p.get("artist", "Unknown"), all_artists),
            "style": best_label_match(p.get("style", "Unknown"), all_styles),
            "genre": best_label_match(p.get("genre", "Unknown"), all_genres),
        })
    # 提取各维度列表
    artists = [l.get("artist", "Unknown") for l in labels_struct]
    preds_artists = [p.get("artist", "Unknown") for p in preds_struct_mapped]
    styles = [l.get("style", "Unknown") for l in labels_struct]
    preds_styles = [p.get("style", "Unknown") for p in preds_struct_mapped]
    genres = [l.get("genre", "Unknown") for l in labels_struct]
    preds_genres = [p.get("genre", "Unknown") for p in preds_struct_mapped]
    per_cls_artist = per_class_prf(preds_artists, artists) if artists and preds_struct else {}
    per_cls_style = per_class_prf(preds_styles, styles) if styles and preds_struct else {}
    per_cls_genre = per_class_prf(preds_genres, genres) if genres and preds_struct else {}
    macro_f1_artist, weighted_f1_artist = macro_weighted_f1(per_cls_artist, artists) if per_cls_artist else (0.0, 0.0)
    macro_f1_style, weighted_f1_style = macro_weighted_f1(per_cls_style, styles) if per_cls_style else (0.0, 0.0)
    macro_f1_genre, weighted_f1_genre = macro_weighted_f1(per_cls_genre, genres) if per_cls_genre else (0.0, 0.0)
    per_concept_artist = per_concept_breakdown(preds_artists, artists) if preds_artists else {}
    per_concept_style = per_concept_breakdown(preds_styles, styles) if preds_styles else {}
    per_concept_genre = per_concept_breakdown(preds_genres, genres) if preds_genres else {}
    # coverage per concept（自然与结构化）
    nat_cov_artist = coverage_per_concept(natural_texts, [labels.get(img, {}).get("artist", "") for img in modes["natural"].keys()]) if natural_texts else {}
    nat_cov_style = coverage_per_concept(natural_texts, [labels.get(img, {}).get("style", "") for img in modes["natural"].keys()]) if natural_texts else {}
    nat_cov_genre = coverage_per_concept(natural_texts, [labels.get(img, {}).get("genre", "") for img in modes["natural"].keys()]) if natural_texts else {}
    str_cov_artist = coverage_per_concept(structured_texts, [labels.get(img, {}).get("artist", "") for img in modes["structured"].keys()]) if structured_texts else {}
    str_cov_style = coverage_per_concept(structured_texts, [labels.get(img, {}).get("style", "") for img in modes["structured"].keys()]) if structured_texts else {}
    str_cov_genre = coverage_per_concept(structured_texts, [labels.get(img, {}).get("genre", "") for img in modes["structured"].keys()]) if structured_texts else {}

    results: Dict[str, Any] = {
        "natural": {
            "coverage": cov_natural,
            "coverage_per_concept": {
                "artist": nat_cov_artist,
                "style": nat_cov_style,
                "genre": nat_cov_genre,
            },
            "multi_label_all_correct": acc_all_nat,
            "macro_f1": {
                "artist": macro_f1_artist_nat,
                "style": macro_f1_style_nat,
                "genre": macro_f1_genre_nat,
            },
            "weighted_f1": {
                "artist": weighted_f1_artist_nat,
                "style": weighted_f1_style_nat,
                "genre": weighted_f1_genre_nat,
            },
            "per_concept": {
                "artist": per_concept_artist_nat,
                "style": per_concept_style_nat,
                "genre": per_concept_genre_nat,
            },
        },
        "structured": {
            "coverage": cov_structured,
            "coverage_per_concept": {
                "artist": str_cov_artist,
                "style": str_cov_style,
                "genre": str_cov_genre,
            },
            "multi_label_all_correct": acc_all,
            "macro_f1": {
                "artist": macro_f1_artist,
                "style": macro_f1_style,
                "genre": macro_f1_genre,
            },
            "weighted_f1": {
                "artist": weighted_f1_artist,
                "style": weighted_f1_style,
                "genre": weighted_f1_genre,
            },
            "per_concept": {
                "artist": per_concept_artist,
                "style": per_concept_style,
                "genre": per_concept_genre,
            },
        },
    }
    # 汇总显著图元数据（如果存在）：按维度统计均值/最大值，并记录路径列表
    saliency_paths: Dict[str, List[str]] = {"artist": [], "style": [], "genre": []}
    saliency_stats: Dict[str, Dict[str, float]] = {
        "artist": {"mean": 0.0, "max": 0.0, "count": 0.0},
        "style": {"mean": 0.0, "max": 0.0, "count": 0.0},
        "genre": {"mean": 0.0, "max": 0.0, "count": 0.0},
    }
    for img, entry in reasoning.items():
        meta = entry.get("__meta__", {}) if isinstance(entry, dict) else {}
        sal = meta.get("saliency", {})
        for dim in ["artist", "style", "genre"]:
            info = sal.get(dim, {})
            pth = info.get("path", "")
            st = info.get("stats", {})
            if pth:
                saliency_paths[dim].append(pth)
            m = st.get("mean", None)
            x = st.get("max", None)
            if isinstance(m, (int, float)):
                saliency_stats[dim]["mean"] += float(m)
                saliency_stats[dim]["count"] += 1.0
            if isinstance(x, (int, float)):
                saliency_stats[dim]["max"] = max(saliency_stats[dim]["max"], float(x))
    for dim in ["artist", "style", "genre"]:
        c = saliency_stats[dim]["count"]
        if c > 0:
            saliency_stats[dim]["mean"] = saliency_stats[dim]["mean"] / c
    results["saliency"] = {
        "paths": saliency_paths,
        "stats": {
            "artist": {"mean": saliency_stats["artist"]["mean"], "max": saliency_stats["artist"]["max"], "count": saliency_stats["artist"]["count"]},
            "style": {"mean": saliency_stats["style"]["mean"], "max": saliency_stats["style"]["max"], "count": saliency_stats["style"]["count"]},
            "genre": {"mean": saliency_stats["genre"]["mean"], "max": saliency_stats["genre"]["max"], "count": saliency_stats["genre"]["count"]},
        },
    }
    # 额外：如果存在 LLaVA 绑定的显著图（structured），也汇总其路径与统计
    sal_llava_paths: List[str] = []
    sal_llava_stats = {"mean": 0.0, "max": 0.0, "count": 0.0}
    for img, entry in reasoning.items():
        meta = entry.get("__meta__", {}) if isinstance(entry, dict) else {}
        sal_l = meta.get("saliency_llava", {})
        info = sal_l.get("structured", {})
        pth = info.get("path", "")
        st = info.get("stats", {})
        if pth:
            sal_llava_paths.append(pth)
        m = st.get("mean", None)
        x = st.get("max", None)
        if isinstance(m, (int, float)):
            sal_llava_stats["mean"] += float(m)
            sal_llava_stats["count"] += 1.0
        if isinstance(x, (int, float)):
            sal_llava_stats["max"] = max(sal_llava_stats["max"], float(x))
    if sal_llava_stats["count"] > 0:
        sal_llava_stats["mean"] = sal_llava_stats["mean"] / sal_llava_stats["count"]
    results["saliency_llava"] = {
        "paths": sal_llava_paths,
        "stats": sal_llava_stats,
    }
    # activation metrics (precision, recall@3, recall@5) from runner meta if present
    # build GT per image per dimension
    gt_per_img = {img: [labels.get(img, {}).get("artist", ""), labels.get(img, {}).get("style", ""), labels.get(img, {}).get("genre", "")] for img in reasoning.keys()}
    # extract rankings per dim from __meta__
    rank_artist = {}
    rank_style = {}
    rank_genre = {}
    for img, entry in reasoning.items():
        meta = entry.get("__meta__", {}) if isinstance(entry, dict) else {}
        act = meta.get("activation", {})
        def to_names(pairs):
            # pairs are global idx + score; names unavailable here; keep idx as string for matching if GT is also idx-based
            return [str(i) for (i, _) in pairs]
        rank_artist[img] = to_names(act.get("artist", {}).get("ranked_global", []))
        rank_style[img] = to_names(act.get("style", {}).get("ranked_global", []))
        rank_genre[img] = to_names(act.get("genre", {}).get("ranked_global", []))
    # 提升：将 runner 的排名索引映射为名称后做 recall@k，这里先尝试直接名称匹配（若 runner 未保存名称则此处可能为空）
    def names_from_rank(entry: Dict[str, Any], dim: str) -> List[str]:
        items = entry.get("__meta__", {}).get("activation", {}).get(dim, {}).get("ranked_names", [])
        return items if isinstance(items, list) else []
    rank_artist_names = {img: names_from_rank(entry, "artist") for img, entry in reasoning.items()}
    rank_style_names = {img: names_from_rank(entry, "style") for img, entry in reasoning.items()}
    rank_genre_names = {img: names_from_rank(entry, "genre") for img, entry in reasoning.items()}
    results["activation"] = {
        "artist": {
            "recall@3": recall_at_k(rank_artist_names, {img: [labels.get(img, {}).get("artist", "")] for img in labels.keys()}, 3),
            "recall@5": recall_at_k(rank_artist_names, {img: [labels.get(img, {}).get("artist", "")] for img in labels.keys()}, 5),
            "precision": activation_precision({img: (entry.get("__meta__", {}).get("activation", {}).get("artist", {}).get("selected_names", []) if isinstance(entry, dict) else []) for img, entry in reasoning.items()}, {img: [labels.get(img, {}).get("artist", "")] for img in labels.keys()}),
        },
        "style": {
            "recall@3": recall_at_k(rank_style_names, {img: [labels.get(img, {}).get("style", "")] for img in labels.keys()}, 3),
            "recall@5": recall_at_k(rank_style_names, {img: [labels.get(img, {}).get("style", "")] for img in labels.keys()}, 5),
            "precision": activation_precision({img: (entry.get("__meta__", {}).get("activation", {}).get("style", {}).get("selected_names", []) if isinstance(entry, dict) else []) for img, entry in reasoning.items()}, {img: [labels.get(img, {}).get("style", "")] for img in labels.keys()}),
        },
        "genre": {
            "recall@3": recall_at_k(rank_genre_names, {img: [labels.get(img, {}).get("genre", "")] for img in labels.keys()}, 3),
            "recall@5": recall_at_k(rank_genre_names, {img: [labels.get(img, {}).get("genre", "")] for img in labels.keys()}, 5),
            "precision": activation_precision({img: (entry.get("__meta__", {}).get("activation", {}).get("genre", {}).get("selected_names", []) if isinstance(entry, dict) else []) for img, entry in reasoning.items()}, {img: [labels.get(img, {}).get("genre", "")] for img in labels.keys()}),
        },
    }

    save_results(results, out_dir / "metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
