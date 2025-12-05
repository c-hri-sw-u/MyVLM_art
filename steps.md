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