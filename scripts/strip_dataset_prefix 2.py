import csv
from pathlib import Path


def strip_prefix(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r") as fin, out_path.open("w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header is not None:
            writer.writerow(header)
            image_idx = header.index("image_path") if "image_path" in header else 0
        else:
            image_idx = 0
        for row in reader:
            if len(row) == 0:
                continue
            if image_idx < len(row):
                row[image_idx] = row[image_idx].replace("dataset/", "")
            writer.writerow(row)


def main():
    root = Path("artifacts")
    files = [
        root / "concept_signals_artist.csv",
        root / "concept_signals_style.csv",
        root / "concept_signals_genre.csv",
    ]
    for f in files:
        if f.exists():
            out = f.with_name(f.stem + "_stripped.csv")
            strip_prefix(f, out)
            print(f"Processed {f} -> {out}")
        else:
            print(f"Skip {f} (not found)")


if __name__ == "__main__":
    main()
