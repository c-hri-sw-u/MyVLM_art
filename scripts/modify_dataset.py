#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def gen_title(image_path: str) -> str:
    stem = Path(image_path).stem
    parts = [p for p in stem.split("_") if p]
    return " ".join(s.capitalize() for s in parts)


def transform_record(rec: dict) -> dict:
    if "title" not in rec or rec["title"] is None:
        rec["title"] = gen_title(rec.get("image", ""))
    if "date" not in rec:
        rec["date"] = ""
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
    parser.add_argument("--backup", action="store_true", help="Write a .bak backup of original JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r") as f:
        data = json.load(f)

    new_data = [transform_record(r) for r in data]

    if args.backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        with backup_path.open("w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    out_path = Path(args.output) if args.output else input_path
    with out_path.open("w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(new_data)} records -> {out_path}")


if __name__ == "__main__":
    main()

