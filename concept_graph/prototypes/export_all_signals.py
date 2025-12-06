import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_export(dim: str, dataset_json: Path, images_root: Optional[Path], ckpt: Path, out_dir: Path, normalize: str, temperature: float, csv: bool, topk: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "export_signals.py"
    json_out = out_dir / f"concept_signals_{dim}.json"
    cmd_json = [
        sys.executable,
        str(script),
        "--dataset_json",
        str(dataset_json),
        "--ckpt",
        str(ckpt),
        "--dimensions",
        dim,
        "--output",
        str(json_out),
        "--format",
        "json",
        "--normalize",
        normalize,
        "--temperature",
        str(temperature),
    ]
    if images_root is not None:
        cmd_json.extend(["--images_root", str(images_root)])
    subprocess.run(cmd_json, check=True)
    if csv:
        csv_out = out_dir / f"concept_signals_{dim}.csv"
        cmd_csv = [
            sys.executable,
            str(script),
            "--dataset_json",
            str(dataset_json),
            "--ckpt",
            str(ckpt),
            "--dimensions",
            dim,
            "--output",
            str(csv_out),
            "--format",
            "csv",
            "--normalize",
            normalize,
            "--temperature",
            str(temperature),
            "--topk",
            str(topk),
        ]
        if images_root is not None:
            cmd_csv.extend(["--images_root", str(images_root)])
        subprocess.run(cmd_csv, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=False)
    parser.add_argument("--ckpt", type=str, required=False)
    parser.add_argument("--ckpt_artist", type=str, required=False)
    parser.add_argument("--ckpt_style", type=str, required=False)
    parser.add_argument("--ckpt_genre", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--normalize", type=str, default="zscore")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--csv", type=str, default="false")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    images_root = Path(args.images_root) if args.images_root else None
    out_dir = Path(args.output_dir)
    normalize = args.normalize.strip().lower()
    temperature = float(args.temperature)
    do_csv = str(args.csv).lower() in ["1", "true", "yes", "y"]
    topk = max(1, int(args.topk))

    ckpt_artist = Path(args.ckpt_artist) if args.ckpt_artist else (Path(args.ckpt) if args.ckpt else None)
    ckpt_style = Path(args.ckpt_style) if args.ckpt_style else (Path(args.ckpt) if args.ckpt else None)
    ckpt_genre = Path(args.ckpt_genre) if args.ckpt_genre else (Path(args.ckpt) if args.ckpt else None)

    if ckpt_artist is None or ckpt_style is None or ckpt_genre is None:
        raise ValueError("Provide --ckpt (shared) or all of --ckpt_artist/--ckpt_style/--ckpt_genre")

    run_export(
        "artist",
        dataset_json,
        images_root,
        ckpt_artist,
        out_dir,
        normalize,
        temperature,
        do_csv,
        topk,
    )
    run_export(
        "style",
        dataset_json,
        images_root,
        ckpt_style,
        out_dir,
        normalize,
        temperature,
        do_csv,
        topk,
    )
    run_export(
        "genre",
        dataset_json,
        images_root,
        ckpt_genre,
        out_dir,
        normalize,
        temperature,
        do_csv,
        topk,
    )


if __name__ == "__main__":
    main()
