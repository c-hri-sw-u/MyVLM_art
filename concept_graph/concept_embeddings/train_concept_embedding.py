from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
from myvlm.common import VLMType, PersonalizationTask, VLM_TO_LAYER
from configs.myvlm_art_config import MyVLMArtConfig
from vlms.llava_wrapper import LLaVAWrapper
from myvlm.myllava import MyLLaVA
from concept_graph.concept_embeddings.trainer import MultiTokenEmbeddingTrainer

cfg = MyVLMArtConfig(
    concept_name="dataset",
    concept_identifier="painting",
    vlm_type=VLMType.LLAVA,
    personalization_task=PersonalizationTask.CAPTIONING,
    output_root=Path("./outputs"),
    data_root=Path("./data"),
    optimization_steps=100,
    learning_rate=1.0,
    batch_size=1,
    reg_lambda=0.0075,
    device='cuda',
    torch_dtype=torch.float16,
    threshold=0.75,
    max_tokens_per_concept=6,
    max_concepts_per_sample=3,
    backoff_delta=0.05,
    val_subset_n=3,
    max_reason_tokens=64,
    grad_accum_steps=8,
    attn_reg_interval=8,
)

vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
myvlm = MyLLaVA(vlm, layer=VLM_TO_LAYER[cfg.vlm_type], concept_name=cfg.concept_name, cfg=cfg)
trainer = MultiTokenEmbeddingTrainer(cfg=cfg, myvlm=myvlm, dataset_builder=None)
checkpoints = trainer.train()
from datetime import datetime
from zoneinfo import ZoneInfo
out_dir = cfg.output_root / cfg.concept_name / f"seed_{cfg.seed}"
out_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now(ZoneInfo('America/New_York')).strftime("%Y%m%d_%H%M%S")
torch.save(checkpoints, out_dir / f"checkpoints_{cfg.concept_name}_seed_{cfg.seed}_{ts}.pt")
