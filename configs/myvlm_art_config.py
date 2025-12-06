from dataclasses import dataclass
from pathlib import Path

import torch

from myvlm.common import VLMType, PersonalizationTask


@dataclass
class MyVLMArtConfig:
    # 项目/概念基本信息
    concept_name: str
    concept_identifier: str
    vlm_type: VLMType
    personalization_task: PersonalizationTask

    # 路径配置
    output_root: Path = Path('./outputs')
    data_root: Path = Path('./data')

    # 训练超参数
    optimization_steps: int = 100
    learning_rate: float = 1.0
    batch_size: int = 4
    reg_lambda: float = 0.0075
    device: str = 'cuda'
    torch_dtype: torch.dtype = torch.bfloat16

    # 概念注入与 gating
    threshold: float = 0.75                  # 按相似度 s 判定激活：s ≥ threshold
    backoff_delta: float = 0.05              # 无激活时的回退容差：s ≥ threshold − delta 的 Top‑1
    max_tokens_per_concept: int = 4          # 每个概念最多注入的 token 数
    max_concepts_per_sample: int = 3         # 单样本全局预算（跨维度）

    # 分布统一与对比度调节（如需）
    normalize_mode: str = 'none'             # none/zero_one/zscore
    temperature: float = 1.0

    # 多维度合并策略（验证/训练期保持一致）
    topk_per_dim: int = 2                    # 每维度最多保留的候选数
    fairness: bool = True                    # 预算合并时的公平保留（各维度先保一）
    priority: str = 'artist,style,genre'     # 维度优先级（公平保留与剩余预算填充时使用）

    # 运行时派生属性
    save_interval: int = 25
    val_interval: int = 25
    seed: int = 42

    def __post_init__(self):
        # 目录派生与存在性校验（保持与当前项目结构兼容）
        self.concept_data_path = self.data_root / self.concept_name
        assert self.concept_data_path.exists(), \
            f"Data path {self.concept_data_path} does not exist!"
        self.output_path = self.output_root / self.concept_name / f'seed_{self.seed}'
        self.output_path.mkdir(exist_ok=True, parents=True)
