import json
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def _read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _set_style():
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
    })


def _monet_palette():
    return ["#6FA9D2", "#C7B5D9", "#F3C6C3", "#A9D9C1", "#F2D9A7", "#8EC28A", "#D9A7A0"]


def _colors_series(n: int):
    base = _monet_palette()
    if n <= len(base):
        return base[:n]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def _dual_colors():
    return ["#6FA9D2", "#C7B5D9"]


def _annotate(ax, rects, vals):
    for i, r in enumerate(rects):
        v = vals[i] if i < len(vals) else r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)


def _set_ylim(ax, series_vals, pad=0.15, min_top=1.0):
    mx = 0.0
    for arr in series_vals:
        if isinstance(arr, (list, tuple)) and len(arr) > 0:
            try:
                mx = max(mx, max(arr))
            except Exception:
                pass
    top = max(min_top, mx * (1.0 + pad)) if mx > 0 else min_top
    ax.set_ylim(0, top)


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#' + ''.join(f'{int(max(0, min(1, c)) * 255):02x}' for c in rgb)


def _color_variants(base_hex: str, n: int):
    import colorsys
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n <= 1:
        return [base_hex]
    offs = []
    step = 0.15 if n <= 3 else max(0.08, 0.35 / max(1, n - 1))
    start = -step * ((n - 1) / 2.0)
    for i in range(n):
        offs.append(start + i * step)
    out = []
    for o in offs:
        l2 = max(0.15, min(0.85, l + o))
        s2 = max(0.5, min(1.0, s + (0.1 if o < 0 else -0.05)))
        rr, gg, bb = colorsys.hls_to_rgb(h, l2, s2)
        out.append(_rgb_to_hex((rr, gg, bb)))
    return out


def _color_support():
    return "#F2D9A7"


def _color_f1():
    return "#8EC28A"


def _color_coverage():
    return "#A9D9C1"


def _activation_color_map():
    recall_base = "#F3C6C3"
    rec = _near_variants(recall_base, 2, dl=0.01, ds=0.0)
    return {
        "precision": "#D9A7A0",
        "recall@3": rec[0],
        "recall@5": rec[1],
    }


def _near_variants(base_hex: str, n: int, dl: float = 0.02, ds: float = 0.0):
    import colorsys
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n <= 1:
        return [base_hex]
    offs = []
    start = -dl * ((n - 1) / 2.0)
    for i in range(n):
        offs.append(start + i * dl)
    out = []
    for o in offs:
        l2 = max(0.0, min(1.0, l + o))
        s2 = max(0.0, min(1.0, s + ds))
        rr, gg, bb = colorsys.hls_to_rgb(h, l2, s2)
        out.append(_rgb_to_hex((rr, gg, bb)))
    return out


def _plot_activation_single(metrics: dict, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    keys = ["precision", "recall@3", "recall@5"]
    cmap = _activation_color_map()
    cols = [cmap[k] for k in keys]
    for d in dims:
        data = metrics.get("activation", {}).get(d, {})
        if not isinstance(data, dict) or len(data) == 0:
            continue
        vals = [float(data.get(k, 0.0)) for k in keys]
        x = list(range(len(keys)))
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
        r = ax.bar(x, vals, color=cols, width=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        _set_ylim(ax, [vals])
        ax.set_title(d)
        ax.set_ylabel("Activation metric value")
        _annotate(ax, r, vals)
        plt.tight_layout()
        out = out_dir / f"activation_{d}.png"
        fig.savefig(out)
        plt.close(fig)


def _plot_macro_weighted(metrics: dict, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    modes = ["natural", "structured"]
    for key in ["macro_f1", "weighted_f1"]:
        vals = {m: [float(metrics.get(m, {}).get(key, {}).get(d, 0.0)) for d in dims] for m in modes}
        x = list(range(len(dims)))
        w = 0.35
        plt.figure(figsize=(8, 4), dpi=150)
        colors = _dual_colors()
        r1 = plt.bar([i - w / 2 for i in x], vals["natural"], width=w, label="natural", color=colors[0])
        r2 = plt.bar([i + w / 2 for i in x], vals["structured"], width=w, label="structured", color=colors[1])
        plt.xticks(x, dims)
        plt.ylabel("F1")
        _set_ylim(plt.gca(), [vals["natural"], vals["structured"]])
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"{key}.png"
        _annotate(plt.gca(), r1, vals["natural"]) 
        _annotate(plt.gca(), r2, vals["structured"]) 
        plt.savefig(out)
        plt.close()


def _plot_per_concept(metrics: dict, out_dir: Path):
    _set_style()
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
            r_sup = axs[0].bar(x, supports, color=_color_support())
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(labels, rotation=45, ha="right")
            axs[0].set_title("support")
            r_f1 = axs[1].bar(x, f1s, color=_color_f1())
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(labels, rotation=45, ha="right")
            axs[1].set_title("f1")
            axs[1].set_ylabel("F1")
            _set_ylim(axs[1], [f1s])
            plt.tight_layout()
            out = out_dir / f"per_concept_{d}_{m}.png"
            _annotate(axs[0], r_sup, supports)
            _annotate(axs[1], r_f1, f1s)
            plt.savefig(out)
            plt.close(fig)


def _plot_coverage_per_concept(metrics: dict, out_dir: Path):
    _set_style()
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
            r_sup = axs[0].bar(x, supports, color=_color_support())
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(labels, rotation=45, ha="right")
            axs[0].set_title("support")
            r_cov = axs[1].bar(x, covs, color=_color_coverage())
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(labels, rotation=45, ha="right")
            axs[1].set_title("coverage")
            axs[1].set_ylabel("Coverage")
            _set_ylim(axs[1], [covs])
            plt.tight_layout()
            out = out_dir / f"coverage_per_concept_{d}_{m}.png"
            _annotate(axs[0], r_sup, supports)
            _annotate(axs[1], r_cov, covs)
            plt.savefig(out)
            plt.close(fig)


def _plot_activation(metrics: dict, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    keys = ["precision", "recall@3", "recall@5"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    for j, d in enumerate(dims):
        data = metrics.get("activation", {}).get(d, {})
        if not isinstance(data, dict) or len(data) == 0:
            continue
        vals = [float(data.get(k, 0.0)) for k in keys]
        x = list(range(len(keys)))
        cmap = _activation_color_map()
        cols = [cmap[k] for k in keys]
        r = axs[j].bar(x, vals, color=cols, width=0.35)
        axs[j].set_xticks(x)
        axs[j].set_xticklabels(keys)
        _set_ylim(axs[j], [vals])
        axs[j].set_title(d)
        axs[j].set_ylabel("Activation metric value")
        _annotate(axs[j], r, vals)
    plt.tight_layout()
    out = out_dir / "activation_triptych.png"
    fig.savefig(out)
    plt.close(fig)


def _plot_macro_weighted_compare_combined(metrics_list: list, labels: list, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    for key in ["macro_f1", "weighted_f1"]:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
        for j, mode in enumerate(["natural", "structured"]):
            series = []
            for met in metrics_list:
                series.append([float(met.get(mode, {}).get(key, {}).get(d, 0.0)) for d in dims])
            x = list(range(len(dims)))
            n = len(series)
            w = 0.8 / max(1, n)
            base = _dual_colors()[j]
            cols = _color_variants(base, n)
            for i, vals in enumerate(series):
                offs = [-0.4 + i * w + w / 2 + xi for xi in x]
                r = axs[j].bar(offs, vals, width=w, label=labels[i], color=cols[i])
                _annotate(axs[j], r, vals)
            axs[j].set_xticks(x)
            axs[j].set_xticklabels(dims)
            axs[j].set_ylabel("F1")
            _set_ylim(axs[j], series)
            axs[j].set_title(f"F1 (macro/weighted): {mode}")
        fig.legend(labels, loc="upper right")
        fig.tight_layout()
        out = out_dir / f"{key}_compare.png"
        fig.savefig(out)
        plt.close(fig)


def _plot_multi_label_all_correct_compare_combined(metrics_list: list, labels: list, out_dir: Path):
    _set_style()
    modes = ["natural", "structured"]
    vals = {m: [float(met.get(m, {}).get("multi_label_all_correct", 0.0)) for met in metrics_list] for m in modes}
    x = list(range(len(modes)))
    n = len(labels)
    w = 0.8 / max(1, n)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    nat_cols = _color_variants(_dual_colors()[0], n)
    str_cols = _color_variants(_dual_colors()[1], n)
    for i in range(n):
        bar_vals = [vals["natural"][i], vals["structured"][i]]
        offs = [-0.4 + i * w + w / 2 + xi for xi in x]
        bar_colors = [nat_cols[i], str_cols[i]]
        r = ax.bar(offs, bar_vals, width=w, label=labels[i], color=bar_colors)
        _annotate(ax, r, bar_vals)
    ax.set_xticks(list(range(len(modes))))
    ax.set_xticklabels(modes)
    _set_ylim(ax, [vals[m] for m in modes])
    ax.set_ylabel("Exact-match rate")
    ax.legend(title="Model")
    plt.tight_layout()
    out = out_dir / "multi_label_all_correct_compare.png"
    plt.savefig(out)
    plt.close(fig)


def _plot_per_concept_compare(metrics_list: list, labels: list, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    for d in dims:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        drew_any = False
        for j, mode in enumerate(["natural", "structured"]):
            all_keys = set()
            for met in metrics_list:
                data = met.get(mode, {}).get("per_concept", {}).get(d, {})
                all_keys.update(list(data.keys()))
            if not all_keys:
                continue
            drew_any = True
            keys = sorted(list(all_keys))
            x = list(range(len(keys)))
            n = len(metrics_list)
            w = 0.8 / max(1, n)
            base = _dual_colors()[j]
            cols = _color_variants(base, n)
            series_vals = []
            for i, met in enumerate(metrics_list):
                data = met.get(mode, {}).get("per_concept", {}).get(d, {})
                vals = [float(data.get(k, {}).get("f1", 0.0)) for k in keys]
                offs = [-0.4 + i * w + w / 2 + xi for xi in x]
                r = axs[j].bar(offs, vals, width=w, label=labels[i], color=cols[i])
                _annotate(axs[j], r, vals)
                series_vals.append(vals)
            axs[j].set_xticks(x)
            axs[j].set_xticklabels(keys, rotation=45, ha="right")
            _set_ylim(axs[j], series_vals)
            axs[j].set_ylabel("F1")
            axs[j].set_title(f"F1 per concept — {mode}")
        if drew_any:
            fig.legend(labels, loc="upper right")
            fig.tight_layout()
            out = out_dir / f"per_concept_compare_{d}.png"
            fig.savefig(out)
        plt.close(fig)


def _plot_coverage_per_concept_compare(metrics_list: list, labels: list, out_dir: Path):
    _set_style()
    dims = ["artist", "style", "genre"]
    for d in dims:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        drew_any = False
        for j, mode in enumerate(["natural", "structured"]):
            all_keys = set()
            for met in metrics_list:
                data = met.get(mode, {}).get("coverage_per_concept", {}).get(d, {})
                all_keys.update(list(data.keys()))
            if not all_keys:
                continue
            drew_any = True
            keys = sorted(list(all_keys))
            x = list(range(len(keys)))
            n = len(metrics_list)
            w = 0.8 / max(1, n)
            base = _dual_colors()[j]
            cols = _color_variants(base, n)
            series_vals = []
            for i, met in enumerate(metrics_list):
                data = met.get(mode, {}).get("coverage_per_concept", {}).get(d, {})
                vals = [float(data.get(k, {}).get("coverage", 0.0)) for k in keys]
                offs = [-0.4 + i * w + w / 2 + xi for xi in x]
                r = axs[j].bar(offs, vals, width=w, label=labels[i], color=cols[i])
                _annotate(axs[j], r, vals)
                series_vals.append(vals)
            axs[j].set_xticks(x)
            axs[j].set_xticklabels(keys, rotation=45, ha="right")
            _set_ylim(axs[j], series_vals)
            axs[j].set_ylabel("Coverage")
            axs[j].set_title(f"Coverage per concept — {mode}")
        if drew_any:
            fig.legend(labels, loc="upper right")
            fig.tight_layout()
            out = out_dir / f"coverage_per_concept_compare_{d}.png"
            fig.savefig(out)
        plt.close(fig)


def _write_html(out_dir: Path):
    parts = []
    def img(name):
        p = out_dir / name
        if p.exists():
            parts.append(f"<h3>{name}</h3><img src='{name}' style='max-width:100%;'>")
    parts.append("<h2>Macro/Weighted F1</h2>")
    img("macro_f1.png")
    img("weighted_f1.png")
    img("macro_f1_compare.png")
    img("weighted_f1_compare.png")
    parts.append("<h2>Multi-Label All Correct</h2>")
    img("multi_label_all_correct_compare.png")
    parts.append("<h2>Per-Concept</h2>")
    for m in ["natural", "structured"]:
        for d in ["artist", "style", "genre"]:
            img(f"per_concept_{d}_{m}.png")
    for d in ["artist", "style", "genre"]:
        img(f"per_concept_compare_{d}.png")
    parts.append("<h2>Coverage Per Concept</h2>")
    for m in ["natural", "structured"]:
        for d in ["artist", "style", "genre"]:
            img(f"coverage_per_concept_{d}_{m}.png")
    for d in ["artist", "style", "genre"]:
        img(f"coverage_per_concept_compare_{d}.png")
    parts.append("<h2>Activation Metrics</h2>")
    img("activation_triptych.png")
    for d in ["artist", "style", "genre"]:
        img(f"activation_{d}.png")
    html = """<html><head><meta charset='utf-8'><title>metrics visualization</title></head><body>""" + "\n".join(parts) + "</body></html>"
    with (out_dir / "metrics_viz.html").open("w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, default="")
    parser.add_argument("--metrics_paths", type=str, nargs="*", default=[])
    parser.add_argument("--labels", type=str, nargs="*", default=[])
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    files = []
    if args.metrics_paths:
        files = args.metrics_paths
    elif args.metrics_path:
        files = [args.metrics_path]
    else:
        raise SystemExit("must provide --metrics_path or --metrics_paths")
    metrics_list = []
    paths = [Path(f) for f in files]
    for p in paths:
        metrics_list.append(_read_json(p))
    if args.labels and len(args.labels) == len(metrics_list):
        labels = args.labels
    else:
        labels = [paths[i].stem for i in range(len(paths))]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = paths[0].parent / ("viz_compare" if len(paths) > 1 else "viz")
    _ensure_dir(out_dir)
    if len(metrics_list) == 1:
        metrics = metrics_list[0]
        _plot_macro_weighted(metrics, out_dir)
        _plot_per_concept(metrics, out_dir)
        _plot_coverage_per_concept(metrics, out_dir)
        _plot_activation(metrics, out_dir)
        _plot_activation_single(metrics, out_dir)
    else:
        _plot_macro_weighted_compare_combined(metrics_list, labels, out_dir)
        _plot_multi_label_all_correct_compare_combined(metrics_list, labels, out_dir)
        _plot_per_concept_compare(metrics_list, labels, out_dir)
        _plot_coverage_per_concept_compare(metrics_list, labels, out_dir)
    _write_html(out_dir)


if __name__ == "__main__":
    main()
