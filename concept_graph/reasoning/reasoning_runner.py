"""
简要介绍 (ZH):
  解释性推理运行器。将激活的概念图转化为 VLM 的输入（image_tensor + prompts），
  并在注入层追加相应概念嵌入，统一执行生成与结果整理（文字解释 + 可选定位/可视化）。

Overview (EN):
  Reasoning runner. Converts activated concept graph into VLM-ready inputs (image tensor + prompts),
  attaches concept embeddings via injection layer, and executes generation to produce textual justifications
  and optional visualizations.

TODOs (详细):
  1) 与 prompt_templates 对接，生成多模板 prompts
  2) 构造 VLM 输入：调用 vlms/* 封装的 preprocess/generate
  3) 概念嵌入注入：调用 inference/inference_utils.py 将当前迭代的 keys/values 写入目标层
  4) 输出打包：文字解释 JSON，必要时保存概念图与定位图
"""

def run_reasoning(vlm_wrapper, activated_concepts, images, concept_embeddings, cfg):
    # TODO: 实现推理主流程，返回 {image_path: {prompt: output_text, ...}, ...}
    return {}
