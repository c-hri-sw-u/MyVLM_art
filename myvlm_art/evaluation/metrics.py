"""
简要介绍 (ZH):
  评估指标集合。包含艺术家识别准确率、分艺术家 Precision/Recall/F1、概念激活指标、
  解释性评估（覆盖度、连贯性、扎根性、流畅度），以及多维度联合指标（多标签正确率）。

Overview (EN):
  Evaluation metrics for artist identification and explainability. Includes accuracy, per-artist PRF,
  concept activation metrics, and reasoning coverage/coherence/grounding/fluency, plus multi-label correctness.

TODOs (详细):
  1) 分类指标：top‑1/top‑k、macro/weighted F1，分艺术家表格
  2) 概念激活：每维度 Precision/Recall/F1、多标签一致性（4 维同时正确）
  3) 解释评估：
     - 概念覆盖分：生成解释是否覆盖所有激活概念
     - 连贯性/扎根性/流畅度：人评接口 + 规则基评分（可选）
  4) 汇总：将评估结果写入 JSON/CSV，供对比与绘图
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def accuracy_topk(preds: List[str], labels: List[str], k: int = 1) -> float:
    """计算 top‑k 准确率；preds 为按置信度排序的候选列表或单一预测。"""
    correct = 0
    for i, p in enumerate(preds):
        if isinstance(p, list):
            correct += int(labels[i] in p[:k])
        else:
            correct += int(p == labels[i])
    return correct / max(1, len(labels))


def per_class_prf(preds: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    """按类别统计 Precision/Recall/F1（宏观）。"""
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


def macro_weighted_f1(per_class: Dict[str, Dict[str, float]], labels: List[str]) -> Tuple[float, float]:
    """计算 macro 与 weighted F1。"""
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
    """概念激活的 Precision/Recall/F1（按维度或总体），输入为每样本的概念列表。"""
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


def multi_label_all_correct(preds: List[Dict[str, str]], labels: List[Dict[str, str]]) -> float:
    """多维度联合准确率：四维标签全部正确的比例。"""
    correct = 0
    for p, y in zip(preds, labels):
        correct += int(all(p.get(k) == y.get(k) for k in y.keys()))
    return correct / max(1, len(labels))


def coverage_score(generated_texts: List[str], activated_concepts: List[List[str]]) -> float:
    """概念覆盖分：文本是否至少提到每个激活概念的平均比例（简单规则基）。"""
    import re
    scores = []
    for text, concepts in zip(generated_texts, activated_concepts):
        t = text.lower()
        covered = 0
        for c in concepts:
            c_norm = re.escape(c.lower())
            if re.search(r"\b" + c_norm + r"\b", t):
                covered += 1
        scores.append(covered / max(1, len(concepts)))
    return sum(scores) / max(1, len(scores))


def summarize(results: Dict) -> Dict:
    """汇总并返回可保存的评估字典。"""
    return results
