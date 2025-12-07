import json
import sys
from pathlib import Path


def main():
    in_path = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else "dataset_test_more3/"
    key = sys.argv[3] if len(sys.argv) > 3 else "image"
    with in_path.open("r") as f:
        data = json.load(f)
    changed = 0
    for rec in data:
        if isinstance(rec, dict) and key in rec and isinstance(rec[key], str):
            s = rec[key]
            if s.startswith(prefix):
                rec[key] = s[len(prefix):]
                changed += 1
    with in_path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"updated {changed}")


if __name__ == "__main__":
    main()
