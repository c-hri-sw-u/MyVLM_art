#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path


def gen_title(image_path: str) -> str:
    stem = Path(image_path).stem
    parts = [p for p in stem.split("_") if p]
    return " ".join(s.capitalize() for s in parts)


def _strip_dataset_prefix(path_str: str) -> str:
    try:
        p = Path(path_str)
        parts = p.parts
        if len(parts) > 0 and parts[0] == "dataset":
            return str(Path(*parts[1:]))
        return path_str
    except Exception:
        return path_str


def transform_record(rec: dict, strip_dataset_prefix: bool = False) -> dict:
    if "title" not in rec or rec["title"] is None:
        rec["title"] = gen_title(rec.get("image", ""))
    if "date" not in rec:
        rec["date"] = ""
    if strip_dataset_prefix:
        img = rec.get("image", "")
        if isinstance(img, str):
            rec["image"] = _strip_dataset_prefix(img)
    concepts = rec.get("concepts", {})
    media = concepts.get("media", None)
    if media == "" or media is None:
        concepts["media"] = []
    if "art_movement" not in concepts:
        concepts["art_movement"] = concepts.get("style", "")
    rec["concepts"] = concepts
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/ubuntu/MyVLM_art/data/dataset_minority/wikiart_5artists_dataset.json",
        help="Path to input dataset JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path; if not set, modifies input in place",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--strip_dataset_prefix_csv", action="store_true")
    parser.add_argument("--prune_csv", action="store_true")
    parser.add_argument("--backup", action="store_true", help="Write a .bak backup of original JSON")
    parser.add_argument("--strip_dataset_prefix", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r") as f:
        data = json.load(f)

    new_data = [transform_record(r, strip_dataset_prefix=args.strip_dataset_prefix) for r in data]

    if args.backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        with backup_path.open("w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    out_path = Path(args.output) if args.output else input_path
    with out_path.open("w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(new_data)} records -> {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [row for row in reader]
        if args.prune_csv:
            allowed = set()
            for r in new_data:
                rel = r.get("image", "")
                if rel:
                    allowed.add(_strip_dataset_prefix(rel))
            filtered = []
            for row in rows:
                rel = row.get("image_path", "")
                canon = _strip_dataset_prefix(rel) if args.strip_dataset_prefix_csv else rel
                if canon in allowed:
                    if args.strip_dataset_prefix_csv:
                        row["image_path"] = canon
                    filtered.append(row)
            rows = filtered
        if args.backup:
            bcsv = csv_path.with_suffix(csv_path.suffix + ".bak")
            with bcsv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows:
                    w.writerow(row)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Processed {len(rows)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
