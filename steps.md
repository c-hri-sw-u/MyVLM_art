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
  - Dataset injection of precomputed signals在训练侧由 `concept_graph/concept_embeddings/trainer.py:181–191, 207–281` 完成
  - 路径前缀统一（不再强制添加 `dataset/`）：导出脚本已更新，读取相对路径相对于 `dataset_json.parent`（`concept_graph/prototypes/export_signals.py:31–44, 88–100, 149–161`）
  - 如需批量修正既有 CSV 前缀，使用 `scripts/strip_dataset_prefix.py`
 - Next: 确认 artist/style/genre 三维 JSON 与 CSV 路径一致（相对 `data/dataset`）

Step 3: Concept Embedding Layer and Gating
- Status: Completed
- Evidence:
  - Dynamic multi‑token injection与阈值/回退/预算在 `concept_graph/concept_embeddings/multi_embed_layer.py:111–177, 179–191`
  - mm_projector 透传 `concept_signals` 于注入层 `vlms/llava/model/llava_arch.py:140–146` 与 `vlms/llava/model/multimodal_projector/builder.py:45–52`
  - 概念 token 位置在 LLaVA 编码阶段被记录，用于正则 `vlms/llava/model/language_model/llava_llama.py:243–343`
- Notes:
  - 统一阈值语义为“相似度 s ≥ 0.75”，回退容差 `δ=0.05`，每维 `Top‑K=2`，全局预算 `3`，维度优先级 `artist,style,genre`
  - Gating 配置通过 `MyVLMArtConfig` 传入注入层，训练侧设置见 `concept_graph/concept_embeddings/trainer.py:290–302`

Step 4: LLaVA Encoding and Weighted Losses
- Status: Completed
- Evidence:
  - A/B collate 与 token 加权在 `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:110–137`
  - 预计算 keys/reasons 的 CSV 加载在 `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:146–160`
  - 加权 CE 集成在 `vlms/llava/model/language_model/llava_llama.py:123–146`
- 训练循环、正则、checkpointing 在 `concept_graph/concept_embeddings/trainer.py:68–154`
 - 说明：当 `reason` 为空时，权重仍与样本长度正确对齐并 pad（`concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:116–129, 134–137`），语言模型侧用右移一位的权重向量参与加权 CE（`vlms/llava/model/language_model/llava_llama.py:129–137`）

Step 5: Phased Training Schedule
- Status: Completed
- Evidence:
  - 阶段化 A/B 数据流与权重在 `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:110–137`
  - A/B 训练与验证、优化器与调度、checkpointing 在 `concept_graph/concept_embeddings/trainer.py:68–154,156–191,234–259`

Step 6: Evaluation and Alignment
- Status: In Progress
- Evidence:
  - 概念激活/预算与 Top‑1 精度验证脚本 `concept_graph/validation/validate_signals.py:110–210,229–279`
  - 指标函数集合（准确率、宏/加权 F1、覆盖分）在 `myvlm_art/evaluation/metrics.py:1–111`
- Next:
  - 将评估脚本与训练产物联动，输出 per‑dimension PRF、macro/weighted F1、覆盖/连贯汇总 JSON/CSV；在验证集周期性运行

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

写一个小脚本（或在 prototype_trainer 里）批量生成 concept_signals 并存成 JSON，路径如 artifacts/concept_signals_<dim>.json（相对路径以 `van_gogh/...` 为准，不加 `dataset/`）

在 ConceptGraphDataset 构造时通过 precomputed_signals 注入，避免每次都在线算

步骤 3：完善概念嵌入层与 gating

在 multi_embed_layer.py 实现：

keys/values 初始化：values 形状 [n_concepts, n_tokens, dim]

gating：基于相似度阈值（s ≥ 0.75），无激活按 `τ − δ` 回退每维 Top‑1；每维 Top‑K=2；公平预算合并，总预算 3；根据 s_c 映射追加 token 数 `k = ceil(K_max · s_c)`，且 `k ≥ 1`

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

——

Quick Start（ZH）

- 准备数据与信号
  - 确保存在 `data/dataset/wikiart_5artists_dataset.json`
  - 训练或加载 CLIP 原型检查点：`artifacts/prototypes_<dim>.pt`（artist/style/genre）
  - 导出概念信号（不加 `dataset/` 前缀）：
    - `python concept_graph/prototypes/export_all_signals.py --dataset_json data/dataset/wikiart_5artists_dataset.json --ckpt_artist artifacts/prototypes_artist.pt --ckpt_style artifacts/prototypes_style.pt --ckpt_genre artifacts/prototypes_genre.pt --output_dir artifacts --normalize zscore --temperature 0.8 --csv true --topk 3`
  - 若已有 CSV 含 `dataset/` 前缀，运行：
    - `python scripts/strip_dataset_prefix.py`
- 准备训练 CSV（keys/reason 文本）
  - 最小版：`artifacts/synth_targets.csv`，字段包含 `image_path,artist,style,genre,target_keys,target_reason`
- 启动训练（在你的 conda 环境）
  - `conda activate myvlm`
  - `python concept_graph/concept_embeddings/train_concept_embedding.py`
- 训练输出
  - 检查点：`outputs/wikiart_5artists/seed_42/concept_embeddings_llava_captioning.pt`
  - 中途验证输出与日志见 `concept_graph/concept_embeddings/trainer.py:68–154`

Readiness Checklist

- `data/dataset/wikiart_5artists_dataset.json` 存在，图片路径有效
- `artifacts/prototypes_artist.pt/style.pt/genre.pt` 就绪，能成功导出三维信号到 `artifacts/concept_signals_*.json`
- `artifacts/synth_targets.csv` 已按 `van_gogh/...` 路径风格准备
- 你的 conda 环境已安装 `torch/transformers/bitsandbytes/xformers/open-clip` 等依赖

如果以上条件满足，即可开始测试训练。训练期间会按“相似度 0.75 + 回退 0.05 + 每维 Top‑K=2 + 公平预算=3”的统一策略进行注入与门控。

步骤 5：阶段化训练调度

阶段 A：使用 from_csv(csv_path, images_root, ..., stage_mode="A", w_keys=1.0, w_reason=0.0)；跑若干 epoch 至稳定

阶段 B：切换 stage_mode="B"，w_reason=0.2，学习理由的语言风格；定期验证三键分类是否稳定

步骤 6：评估与对齐

用 evaluation/metrics.py 对生成的三键进行分类评估：top‑1/top‑k、宏平均 F1、按维度统计

推理模板：结构化推理（JSON）与自然推理两种，在 concept_graph/reasoning/inference_prompt_templates.py 统一风格，并确保 inference 与训练模板差异可控
