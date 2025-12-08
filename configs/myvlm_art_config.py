from dataclasses import dataclass
from pathlib import Path

# 允许在未安装 torch 时运行评估脚本
try:
    import torch
except Exception:
    from typing import Any
    class _TorchStub:
        class dtype:  # 提供占位以满足类型注解
            pass
        bfloat16 = "bfloat16"
    torch = _TorchStub()

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

    # 推理与可解释性
    prompt: str = 'natural'                  # natural / structured / both
    save_saliency: bool = False
    saliency_grid: int = 16
    saliency_source: str = 'prototype'       # prototype / llava_structured
    saliency_sample_n: int = 0
    saliency_sample_mode: str = 'first'
    saliency_only: bool = False
    limit: int = 0                           # 推理时可选限制图片数量（0 表示不限制）
    input_json: str = ''                     # 评估时可指定 reasoning.json 的绝对路径
    dataset_json: str = 'wikiart_5artists_dataset.json'  # 数据集标签 JSON 文件名

    # 运行时派生属性
    save_interval: int = 1
    val_interval: int = 25
    val_subset_n: int = 5
    max_reason_tokens: int = 64
    seed: int = 42
    grad_accum_steps: int = 4
    attn_reg_interval: int = 4

    def __post_init__(self):
        # 目录派生与存在性校验（保持与当前项目结构兼容）
        # 允许 data_root 直接就是数据集根目录（不再强制拼接 dataset）
        # 如果 data_root 已经包含 concept_name 子目录，则按该子目录；否则直接用 data_root
        candidate = self.data_root / self.concept_name
        self.concept_data_path = candidate if candidate.exists() else self.data_root
        assert self.concept_data_path.exists(), \
            f"Data path {self.concept_data_path} does not exist!"
        self.output_path = self.output_root / self.concept_name / f'seed_{self.seed}'
        self.output_path.mkdir(exist_ok=True, parents=True)
