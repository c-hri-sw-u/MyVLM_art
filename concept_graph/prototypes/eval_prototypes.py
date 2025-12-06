from pathlib import Path
import json
import sys
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from concept_graph.prototypes.prototype_head import PrototypeHead

ckpt = Path("/home/ubuntu/MyVLM_art/artifacts/prototypes_genre_trained.pt")
head = PrototypeHead(device="cuda")
head.load_prototypes(ckpt)

import json
with open(str(PROJECT_ROOT / "data" / "dataset" / "wikiart_5artists_dataset.json"), "r") as f:
    records = json.load(f)
images_root = PROJECT_ROOT / "data" / "dataset"

dim = "genre"
proto = head.prototypes[dim].to(head.device)  # [C, D]
correct, total = 0, 0
for r in records:
    label = r.get("concepts", {}).get(dim, None)
    if label is None or label not in head.concept_to_idx[dim]:
        continue
    p = images_root / r["image"]
    if not p.exists():
        continue
    with Image.open(p) as img:
        image = img.convert("RGB")
    x = head.preprocess(image).unsqueeze(0).to(head.device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=head.precision=="fp16"):
            feat = head.model.encode_image(x)
        feat = F.normalize(feat.float(), dim=-1)  # [1, D]
    sims = feat @ proto.T  # [1, C]
    pred_idx = int(sims.argmax(dim=1).item())
    gt_idx = head.concept_to_idx[dim][label]
    correct += int(pred_idx == gt_idx)
    total += 1
acc = correct / max(total, 1)
print("Top-1 accuracy:", acc, "total:", total)