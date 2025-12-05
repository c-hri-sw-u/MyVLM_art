Status Update (English)

Step 1: Prototype Head (Minimal Viable Version)
- Status: Completed
- Evidence:
  - CLIP load and preprocessing in `concept_graph/prototypes/prototype_head.py:44–46`
  - Prototype initialization/build/save in `concept_graph/prototypes/prototype_head.py:53–95`
  - Signal extraction in `concept_graph/prototypes/prototype_head.py:102–129`
  - Prototype training loop (InfoNCE + margin) in `concept_graph/prototypes/prototype_trainer.py:59–157`
  - CLI training script in `concept_graph/prototypes/train_prototypes.py:34–98`
  - Dataset for prototypes in `concept_graph/prototypes/prototype_dataset.py:52–74`
- Next: Optional per-dimension temperature calibration; optional EMA updates of prototypes

Step 2: Precompute Concept Signals and Integration
- Status: Completed
- Evidence:
  - Batch export to JSON/CSV in `concept_graph/prototypes/export_signals.py:91–103` (JSON) and `concept_graph/prototypes/export_signals.py:113–150` (CSV)
  - Dataset injection of precomputed signals in `concept_graph/datasets/concept_graph_dataset.py:63–79`
- Next: Standardize artifacts path naming and JSON schema across runs

Step 3: Concept Embedding Layer and Gating
- Status: Partially Completed
- Evidence:
  - Multi-token injection and gating in `concept_graph/concept_embeddings/multi_embed_layer.py:49–130`
  - mm_projector accepts `concept_signals` in `vlms/llava/model/llava_arch.py:140–146`
- Gaps:
  - Attention regularization hook not yet wired in `MultiTokenConceptLayer`
  - Key/value mapping lifecycle (init/persist/load) needs consolidation
  - Optional Top‑K gating and minimum-token guarantee are not exposed via config

Step 4: LLaVA Encoding and Weighted Losses
- Status: Implemented at dataset/collate level; Trainer not implemented
- Evidence:
  - Stage A/B collate and token weighting in `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:88–144`
  - CSV loader with precomputed keys/reasons in `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:146–193`
- Gap:
  - `MultiTokenEmbeddingTrainer` is a stub in `concept_graph/concept_embeddings/trainer.py:35–44`

Step 5: Phased Training Schedule
- Status: Partially Completed
- Evidence:
  - Stage A/B data flow and weights in `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:88–144`
- Gaps:
  - Training loop, optimizer/scheduler, checkpointing to be implemented in `concept_graph/concept_embeddings/trainer.py`

Step 6: Evaluation and Alignment
- Status: Not Started
- Evidence:
  - `evaluation/metrics.py` referenced in plan but not present
- Next:
  - Implement metrics for Top‑1/Top‑K per dimension; macro F1; coverage and coherence for explanations

原始中文计划如下：
建议步骤

步骤 1：完成原型头最小可用版本

在 prototype_head.py 实现：

加载 CLIP（与项目已有 open-clip-torch 一致），预处理器与特征提取

原型初始化：按概念聚合参考图像求均值并归一化；保存到 artifacts/prototypes_<dim>.pt

extract_signal(image_paths)：计算 cos_sim 并输出 {img_path: {concept_idx: Tensor([1-s, s])}}

训练 prototype_trainer.py：

每维度 InfoNCE：anchor 与同维度 positives、K 个 negatives，优化原型分离度

检查点：每 epoch 保存原型与概念索引映射

参考文件：concept_heads/clip/concept_head_training/datasets.py:54–77 的简单二分类数据集，照此组织 CLIP 预处理

步骤 2：预计算概念信号并接入底座

写一个小脚本（或在 prototype_trainer 里）批量生成 concept_signals 并存成 JSON/NPY，路径如 artifacts/concept_signals.json

在 ConceptGraphDataset 构造时通过 precomputed_signals 注入，避免每次都在线算

步骤 3：完善概念嵌入层与 gating

在 multi_embed_layer.py 实现：

keys/values 初始化：values 形状 [n_concepts, n_tokens, dim]

gating：根据 s_c 映射选择追加的 token 数（如 ceil(K_max * s_c)），Top‑K 或阈值

forward：拼接 token 至被包裹层输出；训练模式下对自注意力加 L2 正则

确认注入位置：vlms/llava/model/llava_arch.py:129–146 已支持 mm_projector(image_features, concept_signals=...)，保证你的 projector/包裹层接收并使用 concept_signals

步骤 4：实现 LLaVA 训练的编码与分权损失

在 llava_concept_graph_dataset.py 基础上补一个 collate_fn：

将 prompt/target_keys/target_reason 编码为 input_ids/labels/attention_mask

阶段 A：仅编码 keys

阶段 B：keys 与 reason 分别编码；在 labels 中标记两个区段，损失按 w_keys/w_reason 加权

训练循环（参考 vlms/llava_wrapper.py 的生成接口与 minigpt4/models/minigpt_base.py:321–396 的生成逻辑）：

前向：encode_images(images, concept_signals=...) → 语言建模损失（CE）

拼接 / 掩码：确保 labels 中非目标部分置为 -100，只在目标位置计算 CE

权重：为目标 token 段加权；可在 loss 前按 mask分区乘以不同权重

步骤 5：阶段化训练调度

阶段 A：使用 from_csv(csv_path, images_root, ..., stage_mode="A", w_keys=1.0, w_reason=0.0)；跑若干 epoch 至稳定

阶段 B：切换 stage_mode="B"，w_reason=0.2，学习理由的语言风格；定期验证三键分类是否稳定

步骤 6：评估与对齐

用 evaluation/metrics.py 对生成的三键进行分类评估：top‑1/top‑k、宏平均 F1、按维度统计

推理模板：结构化推理（JSON）与自然推理两种，在 concept_graph/reasoning/inference_prompt_templates.py 统一风格，并确保 inference 与训练模板差异可控
