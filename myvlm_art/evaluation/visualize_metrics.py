import json
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _plot_macro_weighted(metrics: dict, out_dir: Path):
    dims = ["artist", "style", "genre"]
    modes = ["natural", "structured"]
    for key in ["macro_f1", "weighted_f1"]:
        vals = {m: [float(metrics.get(m, {}).get(key, {}).get(d, 0.0)) for d in dims] for m in modes}
        x = list(range(len(dims)))
        w = 0.35
        plt.figure(figsize=(8, 4), dpi=150)
        plt.bar([i - w / 2 for i in x], vals["natural"], width=w, label="natural")
        plt.bar([i + w / 2 for i in x], vals["structured"], width=w, label="structured")
        plt.xticks(x, dims)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"{key}.png"
        plt.savefig(out)
        plt.close()


def _plot_per_concept(metrics: dict, out_dir: Path):
    dims = ["artist", "style", "genre"]
    modes = ["natural", "structured"]
    for m in modes:
        for d in dims:
            data = metrics.get(m, {}).get("per_concept", {}).get(d, {})
            if not isinstance(data, dict) or len(data) == 0:
                continue
            items = []
            for k, v in data.items():
                sup = int(v.get("support", 0))
                f1 = float(v.get("f1", 0.0))
                items.append((k, sup, f1))
            items.sort(key=lambda t: t[1], reverse=True)
            labels = [t[0] for t in items]
            supports = [t[1] for t in items]
            f1s = [t[2] for t in items]
            x = list(range(len(labels)))
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
            axs[0].bar(x, supports)
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(labels, rotation=45, ha="right")
            axs[0].set_title("support")
            axs[1].bar(x, f1s)
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(labels, rotation=45, ha="right")
            axs[1].set_title("f1")
            axs[1].set_ylim(0, 1.0)
            plt.tight_layout()
            out = out_dir / f"per_concept_{d}_{m}.png"
            plt.savefig(out)
            plt.close(fig)


def _plot_coverage_per_concept(metrics: dict, out_dir: Path):
    dims = ["artist", "style", "genre"]
    modes = ["natural", "structured"]
    for m in modes:
        for d in dims:
            data = metrics.get(m, {}).get("coverage_per_concept", {}).get(d, {})
            if not isinstance(data, dict) or len(data) == 0:
                continue
            items = []
            for k, v in data.items():
                sup = int(v.get("support", 0))
                cov = float(v.get("coverage", 0.0))
                items.append((k, sup, cov))
            items.sort(key=lambda t: t[1], reverse=True)
            labels = [t[0] for t in items]
            supports = [t[1] for t in items]
            covs = [t[2] for t in items]
            x = list(range(len(labels)))
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
            axs[0].bar(x, supports)
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(labels, rotation=45, ha="right")
            axs[0].set_title("support")
            axs[1].bar(x, covs)
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(labels, rotation=45, ha="right")
            axs[1].set_title("coverage")
            axs[1].set_ylim(0, 1.0)
            plt.tight_layout()
            out = out_dir / f"coverage_per_concept_{d}_{m}.png"
            plt.savefig(out)
            plt.close(fig)


def _plot_activation(metrics: dict, out_dir: Path):
    dims = ["artist", "style", "genre"]
    for d in dims:
        data = metrics.get("activation", {}).get(d, {})
        if not isinstance(data, dict) or len(data) == 0:
            continue
        labels = ["precision", "recall@3", "recall@5"]
        vals = [float(data.get(k, 0.0)) for k in labels]
        x = list(range(len(labels)))
        plt.figure(figsize=(6, 4), dpi=150)
        plt.bar(x, vals)
        plt.xticks(x, labels)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        out = out_dir / f"activation_{d}.png"
        plt.savefig(out)
        plt.close()


def _write_html(out_dir: Path):
    parts = []
    def img(name):
        p = out_dir / name
        if p.exists():
            parts.append(f"<h3>{name}</h3><img src='{name}' style='max-width:100%;'>")
    parts.append("<h2>Macro/Weighted F1</h2>")
    img("macro_f1.png")
    img("weighted_f1.png")
    parts.append("<h2>Per-Concept</h2>")
    for m in ["natural", "structured"]:
        for d in ["artist", "style", "genre"]:
            img(f"per_concept_{d}_{m}.png")
    parts.append("<h2>Coverage Per Concept</h2>")
    for m in ["natural", "structured"]:
        for d in ["artist", "style", "genre"]:
            img(f"coverage_per_concept_{d}_{m}.png")
    parts.append("<h2>Activation Metrics</h2>")
    for d in ["artist", "style", "genre"]:
        img(f"activation_{d}.png")
    html = """<html><head><meta charset='utf-8'><title>metrics visualization</title></head><body>""" + "\n".join(parts) + "</body></html>"
    with (out_dir / "metrics_viz.html").open("w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    mp = Path(args.metrics_path)
    metrics = _read_json(mp)
    out_dir = Path(args.out_dir) if args.out_dir else mp.parent / "viz"
    _ensure_dir(out_dir)
    _plot_macro_weighted(metrics, out_dir)
    _plot_per_concept(metrics, out_dir)
    _plot_coverage_per_concept(metrics, out_dir)
    _plot_activation(metrics, out_dir)
    _write_html(out_dir)


if __name__ == "__main__":
    main()

