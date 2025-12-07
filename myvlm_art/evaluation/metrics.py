from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json


def accuracy_topk(preds: List[str], labels: List[str], k: int = 1) -> float:
    correct = 0
    for i, p in enumerate(preds):
        if isinstance(p, list):
            correct += int(labels[i] in p[:k])
        else:
            correct += int(p == labels[i])
    return correct / max(1, len(labels))


def per_class_prf(preds: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    classes = sorted(set(labels))
    cm = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for p, y in zip(preds, labels):
        for c in classes:
            if p == c and y == c:
                cm[c]["tp"] += 1
            elif p == c and y != c:
                cm[c]["fp"] += 1
            elif p != c and y == c:
                cm[c]["fn"] += 1
    out = {}
    for c in classes:
        tp, fp, fn = cm[c]["tp"], cm[c]["fp"], cm[c]["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        out[c] = {"precision": prec, "recall": rec, "f1": f1}
    return out


def per_concept_breakdown(preds: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    classes = sorted(set(labels))
    cm = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for p, y in zip(preds, labels):
        for c in classes:
            if p == c and y == c:
                cm[c]["tp"] += 1
            elif p == c and y != c:
                cm[c]["fp"] += 1
            elif p != c and y == c:
                cm[c]["fn"] += 1
    out = {}
    for c in classes:
        tp, fp, fn = cm[c]["tp"], cm[c]["fp"], cm[c]["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": labels.count(c)}
    return out


def macro_weighted_f1(per_class: Dict[str, Dict[str, float]], labels: List[str]) -> Tuple[float, float]:
    counts = defaultdict(int)
    for y in labels:
        counts[y] += 1
    f1s = []
    weighted = 0.0
    total = len(labels)
    for c, m in per_class.items():
        f1s.append(m["f1"])
        weighted += m["f1"] * counts[c] / max(1, total)
    macro = sum(f1s) / max(1, len(f1s))
    return macro, weighted


def concept_activation_metrics(pred_concepts: Dict[str, List[str]], gt_concepts: Dict[str, List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for key in gt_concepts.keys():
        gt = set(gt_concepts[key])
        pd = set(pred_concepts.get(key, []))
        tp += len(gt & pd)
        fp += len(pd - gt)
        fn += len(gt - pd)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def recall_at_k(rankings: Dict[str, List[str]], gt: Dict[str, List[str]], k: int) -> float:
    hits, total = 0, 0
    for img, gt_list in gt.items():
        ranked = rankings.get(img, [])[:k]
        total += len(gt_list)
        hits += sum(1 for g in gt_list if g in ranked)
    return hits / max(1, total)


def activation_precision(selected: Dict[str, List[str]], gt: Dict[str, List[str]]) -> float:
    hits, total = 0, 0
    for img, sel in selected.items():
        gt_list = gt.get(img, [])
        target = gt_list[0] if len(gt_list) > 0 else None
        if target is None:
            continue
        for s in sel:
            total += 1
            if s == target:
                hits += 1
    return hits / max(1, total)


def canon_str(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def best_label_match(pred: str, candidates: List[str]) -> str:
    p = set(canon_str(pred).split())
    best = None
    best_score = -1.0
    for c in candidates:
        cc = set(canon_str(c).split())
        if len(cc) == 0:
            continue
        inter = len(p & cc)
        union = len(p | cc) if len(p | cc) > 0 else 1
        jacc = inter / union
        contain = 1.0 if (p <= cc or cc <= p) else 0.0
        score = max(jacc, contain)
        if score > best_score:
            best_score = score
            best = c
    return best if best is not None else pred


def multi_label_all_correct(preds: List[Dict[str, str]], labels: List[Dict[str, str]]) -> float:
    correct = 0
    for p, y in zip(preds, labels):
        correct += int(all(p.get(k) == y.get(k) for k in y.keys()))
    return correct / max(1, len(labels))


def coverage_score(generated_texts: List[str], activated_concepts: List[List[str]]) -> float:
    import re
    def canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    scores = []
    for text, concepts in zip(generated_texts, activated_concepts):
        t = canon(text)
        covered = 0
        for c in concepts:
            cc = canon(c)
            if not cc:
                continue
            if (" " + t + " ").find(" " + cc + " ") != -1:
                covered += 1
        scores.append(covered / max(1, len(concepts)))
    return sum(scores) / max(1, len(scores))


def coverage_per_concept(texts: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    import re
    def canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    stats: Dict[str, Tuple[int, int]] = {}
    for t, lab in zip(texts, labels):
        key = lab or ""
        if key not in stats:
            stats[key] = (0, 0)
        hit = 0
        tt = canon(t)
        kk = canon(key)
        if kk and (" " + tt + " ").find(" " + kk + " ") != -1:
            hit = 1
        c_hit, c_tot = stats[key]
        stats[key] = (c_hit + hit, c_tot + 1)
    out: Dict[str, Dict[str, float]] = {}
    for k, (h, tot) in stats.items():
        out[k] = {"coverage": (h / max(1, tot)), "support": float(tot)}
    return out


def save_results(results: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    return results


def strip_diacritics(s: str) -> str:
    import unicodedata
    import re
    if s is None:
        return ""
    nfkd = unicodedata.normalize('NFD', s)
    no_accents = "".join(ch for ch in nfkd if unicodedata.category(ch) != 'Mn')
    return re.sub(r"\s+", " ", no_accents).strip()


def _mutate_ism_to_ist(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t.endswith("ism"):
            out.append(t[:-3] + "ist")
        else:
            out.append(t)
    return out


def build_alias_map_for_artists(candidates: List[str]) -> Dict[str, str]:
    alias_to_many: Dict[str, set] = {}
    for c in candidates:
        s = strip_diacritics(c)
        cs = canon_str(s)
        tokens = [t for t in cs.split() if t]
        variants = set()
        variants.add(cs)
        variants.add(cs.replace(" ", "-"))
        if tokens:
            last = tokens[-1]
            variants.add(last)
            variants.add(last.replace(" ", "-"))
        for v in variants:
            if not v:
                continue
            alias_to_many.setdefault(v, set()).add(c)
    alias_unique: Dict[str, str] = {}
    for a, setc in alias_to_many.items():
        if len(setc) == 1:
            alias_unique[a] = list(setc)[0]
    return alias_unique


def build_alias_map_for_styles(candidates: List[str]) -> Dict[str, str]:
    alias_to_many: Dict[str, set] = {}
    for c in candidates:
        s = strip_diacritics(c)
        cs = canon_str(s)
        tokens = [t for t in cs.split() if t]
        variants = set()
        variants.add(cs)
        variants.add(cs.replace(" ", "-"))
        mt = _mutate_ism_to_ist(tokens)
        if mt != tokens:
            v1 = " ".join(mt)
            variants.add(v1)
            variants.add(v1.replace(" ", "-"))
        for v in variants:
            if not v:
                continue
            alias_to_many.setdefault(v, set()).add(c)
    alias_unique: Dict[str, str] = {}
    for a, setc in alias_to_many.items():
        if len(setc) == 1:
            alias_unique[a] = list(setc)[0]
    return alias_unique


def _pluralize_last(tokens: List[str]) -> List[str]:
    out = tokens[:]
    if len(out) == 0:
        return out
    last = out[-1]
    if last.endswith("s"):
        return out
    out[-1] = last + "s"
    return out


def _singularize_last(tokens: List[str]) -> List[str]:
    out = tokens[:]
    if len(out) == 0:
        return out
    last = out[-1]
    if last.endswith("s") and not last.endswith("ss"):
        out[-1] = last[:-1]
    return out


def build_alias_map_for_genres(candidates: List[str]) -> Dict[str, str]:
    alias_to_many: Dict[str, set] = {}
    for c in candidates:
        s = strip_diacritics(c)
        cs = canon_str(s)
        tokens = [t for t in cs.split() if t]
        variants = set()
        variants.add(cs)
        variants.add(cs.replace(" ", "-"))
        pv = " ".join(_pluralize_last(tokens))
        sv = " ".join(_singularize_last(tokens))
        variants.add(pv)
        variants.add(pv.replace(" ", "-"))
        variants.add(sv)
        variants.add(sv.replace(" ", "-"))
        for v in variants:
            if not v:
                continue
            alias_to_many.setdefault(v, set()).add(c)
    alias_unique: Dict[str, str] = {}
    for a, setc in alias_to_many.items():
        if len(setc) == 1:
            alias_unique[a] = list(setc)[0]
    return alias_unique


def build_alias_maps(artists: List[str], styles: List[str], genres: List[str]) -> Dict[str, Dict[str, str]]:
    return {
        "artist": build_alias_map_for_artists(artists),
        "style": build_alias_map_for_styles(styles),
        "genre": build_alias_map_for_genres(genres),
    }
