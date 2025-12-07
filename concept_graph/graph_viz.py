from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIM_COLORS = {
    "artist": "#1f77b4",
    "style": "#2ca02c",
    "genre": "#d62728",
    "other": "#7f7f7f",
}


def draw_graph(concept_network: Dict[str, Any], output_path: Path) -> None:
    nodes: List[Dict[str, Any]] = concept_network.get("nodes", [])
    edges: List[Dict[str, Any]] = concept_network.get("edges", [])
    xs, ys, cs, labels = [], [], [], []
    for n in nodes:
        x, y = float(n.get("x", 0.0)), float(n.get("y", 0.0))
        dim = n.get("dim", "other")
        label = n.get("label", str(n.get("id", "")))
        xs.append(x)
        ys.append(y)
        cs.append(DIM_COLORS.get(dim, DIM_COLORS["other"]))
        labels.append(label)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(xs, ys, c=cs, s=80, edgecolors="black")
    for i, l in enumerate(labels):
        plt.text(xs[i] + 0.01, ys[i] + 0.01, l, fontsize=8)
    for e in edges:
        i, j = int(e.get("src", 0)), int(e.get("dst", 0))
        w = float(e.get("weight", 1.0))
        x1, y1 = float(nodes[i].get("x", 0.0)), float(nodes[i].get("y", 0.0))
        x2, y2 = float(nodes[j].get("x", 0.0)), float(nodes[j].get("y", 0.0))
        plt.plot([x1, x2], [y1, y2], color="#888888", linewidth=max(0.5, w))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
