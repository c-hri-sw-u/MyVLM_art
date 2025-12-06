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
    batch_size=4,
    reg_lambda=0.0075,
    device='cuda',
    torch_dtype=torch.bfloat16,
    threshold=0.75,
    max_tokens_per_concept=4,
    max_concepts_per_sample=3,
    backoff_delta=0.05,
    val_subset_n=5,
    max_reason_tokens=64,
)

vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
myvlm = MyLLaVA(vlm, layer=VLM_TO_LAYER[cfg.vlm_type], concept_name=cfg.concept_name, cfg=cfg)
trainer = MultiTokenEmbeddingTrainer(cfg=cfg, myvlm=myvlm, dataset_builder=None)
checkpoints = trainer.train()
