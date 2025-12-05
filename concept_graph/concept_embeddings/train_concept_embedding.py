from pathlib import Path
import torch
from myvlm.common import VLMType, ConceptType, PersonalizationTask, VLM_TO_LAYER
from configs.train_config import EmbeddingTrainingConfig
from vlms.llava_wrapper import LLaVAWrapper
from myvlm.myllava import MyLLaVA
from concept_graph.concept_embeddings.trainer import MultiTokenEmbeddingTrainer

cfg = EmbeddingTrainingConfig(
    concept_name="wikiart_5artists",
    concept_identifier="artist",
    concept_type=ConceptType.OBJECT,
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
)

vlm = LLaVAWrapper(device=cfg.device, torch_dtype=cfg.torch_dtype)
myvlm = MyLLaVA(vlm, layer=VLM_TO_LAYER[cfg.vlm_type], concept_name=cfg.concept_name, cfg=cfg)
trainer = MultiTokenEmbeddingTrainer(cfg=cfg, myvlm=myvlm, dataset_builder=None)
checkpoints = trainer.train()