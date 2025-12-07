from pathlib import Path
import argparse
import json
import random

def main():
    p = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent
    p.add_argument("--dataset_json", type=str, default=str(project_root / "data" / "dataset" / "wikiart_5artists_dataset.json"))
    p.add_argument("--images_root", type=str, default=str(project_root / "data" / "dataset"))
    p.add_argument("--output", type=str, default=str(project_root / "data" / "dataset" / "wikiart_5artists_dataset_min5.json"))
    p.add_argument("--min_per_concept", type=int, default=5)
    p.add_argument("--exclude_unknown", type=str, default="false")
    p.add_argument("--verify_images", type=str, default="true")
    p.add_argument("--shuffle", type=str, default="true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_artist", type=int, default=0)
    p.add_argument("--max_per_style", type=int, default=0)
    p.add_argument("--max_per_genre", type=int, default=0)
    args = p.parse_args()

    dataset_path = Path(args.dataset_json)
    images_root = Path(args.images_root)
    out_path = Path(args.output)
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not images_root.is_absolute():
        images_root = project_root / images_root
    if not out_path.is_absolute():
        out_path = project_root / out_path
    exclude_unknown = str(args.exclude_unknown).lower() in ["1", "true", "yes", "y"]
    verify_images = str(args.verify_images).lower() in ["1", "true", "yes", "y"]
    shuffle = str(args.shuffle).lower() in ["1", "true", "yes", "y"]
    random.seed(int(args.seed))

    with dataset_path.open("r") as f:
        recs = json.load(f)

    cnt_a, cnt_s, cnt_g = {}, {}, {}
    for r in recs:
        c = r.get("concepts", {})
        a = c.get("artist", r.get("artist", ""))
        s = c.get("style", "")
        g = c.get("genre", "")
        cnt_a[a] = cnt_a.get(a, 0) + 1
        cnt_s[s] = cnt_s.get(s, 0) + 1
        cnt_g[g] = cnt_g.get(g, 0) + 1

    def ok(label: str, dim: str) -> bool:
        if dim == "artist":
            n = cnt_a.get(label, 0)
            unk = label.lower() in ["unknown", "unknown artist"]
        elif dim == "style":
            n = cnt_s.get(label, 0)
            unk = label.lower() in ["unknown", "unknown style"]
        else:
            n = cnt_g.get(label, 0)
            unk = label.lower() in ["unknown", "unknown genre"]
        if exclude_unknown and unk:
            return False
        return n >= int(args.min_per_concept)

    max_a = int(args.max_per_artist or 0)
    max_s = int(args.max_per_style or 0)
    max_g = int(args.max_per_genre or 0)
    if max_a > 0 and max_a < args.min_per_concept:
        max_a = int(args.min_per_concept)
    if max_s > 0 and max_s < args.min_per_concept:
        max_s = int(args.min_per_concept)
    if max_g > 0 and max_g < args.min_per_concept:
        max_g = int(args.min_per_concept)

    idxs = list(range(len(recs)))
    if shuffle:
        random.shuffle(idxs)

    kept = []
    kept_a, kept_s, kept_g = {}, {}, {}
    for i in idxs:
        r = recs[i]
        c = r.get("concepts", {})
        a = c.get("artist", r.get("artist", ""))
        s = c.get("style", "")
        g = c.get("genre", "")
        if not (ok(a, "artist") and ok(s, "style") and ok(g, "genre")):
            continue
        img_rel = r.get("image", "")
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = images_root / img_path
        if verify_images and not img_path.exists():
            continue
        if max_a > 0 and kept_a.get(a, 0) >= max_a:
            continue
        if max_s > 0 and kept_s.get(s, 0) >= max_s:
            continue
        if max_g > 0 and kept_g.get(g, 0) >= max_g:
            continue
        kept.append(r)
        kept_a[a] = kept_a.get(a, 0) + 1
        kept_s[s] = kept_s.get(s, 0) + 1
        kept_g[g] = kept_g.get(g, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
