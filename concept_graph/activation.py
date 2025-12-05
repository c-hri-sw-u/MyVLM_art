"""
简要介绍 (ZH):
  概念激活流水线。基于原型相似度 s_c 与阈值 τ，筛选各维度被激活的概念，
  并聚合为概念网络（节点为概念，边为维度内/跨维度关系），供推理与可视化使用。

Overview (EN):
  Concept activation pipeline. Filters activated concepts per dimension using prototype similarity and thresholds,
  then aggregates them into a concept network for downstream reasoning and visualization.

Status:
  - Not implemented.

Remaining Work:
  1) Activation rules: per-dimension threshold τ and/or Top‑K selection; return unified structure.
  2) Concept network: build nodes/edges; intra-dimension adjacency and cross-dimension relations (e.g., artist‑style).
  3) Visualization: return attributes needed by `graph_viz.py` (coordinates/labels/weights).
"""

def activate_concepts(prototype_scores, thresholds, topk=None):
    # TODO: Take per-dimension concept scores/similarities and return activated sets + network structure
    return {"nodes": [], "edges": []}
