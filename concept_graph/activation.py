"""
简要介绍 (ZH):
  概念激活流水线。基于原型相似度 s_c 与阈值 τ，筛选各维度被激活的概念，
  并聚合为概念网络（节点为概念，边为维度内/跨维度关系），供推理与可视化使用。

Overview (EN):
  Concept activation pipeline. Filters activated concepts per dimension using prototype similarity and threshold,
  and aggregates them into a concept network for reasoning and visualization.

TODOs (详细):
  1) 激活规则：按维度阈值 τ 或 Top‑K 选择概念，输出统一结构
  2) 概念网络：构造 nodes/edges；维度内邻接与跨维度关联（如 artist‑style）
  3) 与 graph_viz 对接：返回可视化所需的坐标/标签/权重
"""

def activate_concepts(prototype_scores, thresholds, topk=None):
    # TODO: 输入为每维度概念分数/相似度，输出为激活集合与网络结构
    return {"nodes": [], "edges": []}
