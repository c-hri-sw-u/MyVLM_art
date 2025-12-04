"""
简要介绍 (ZH):
  概念网络可视化。将激活的概念图（nodes/edges）渲染为静态图（PNG/SVG）或交互式视图，
  支持按维度着色、边权重显示、与推理结果联动。

Overview (EN):
  Concept graph visualization. Render activated nodes/edges with dimension-based coloring and edge weights.
  Supports saving static images and connecting to reasoning outputs.

TODOs (详细):
  1) 绘制：使用 networkx + matplotlib 或 graphviz；支持布局与主题
  2) 标注：节点显示概念名与维度，边显示关联类型（同维度/跨维度）与权重
  3) 输出：保存至指定路径，并可返回句柄供上游调用
"""

def draw_graph(concept_network, output_path):
    # TODO: 根据 concept_network 渲染图并保存
    pass
