"""
简要介绍 (ZH):
  解释性推理运行器。将激活的概念图转化为 VLM 的输入（image_tensor + prompts），
  并在注入层追加相应概念嵌入，统一执行生成与结果整理（文字解释 + 可选定位/可视化）。

Overview (EN):
  Reasoning runner. Converts activated concept graph into VLM-ready inputs (image tensor + prompts),
  attaches concept embeddings via injection layer, and executes generation to produce textual justifications
  and optional visualizations.

Status:
  - Not implemented.

Remaining Work:
  1) Connect to prompt templates; support multiple prompt variants.
  2) Build VLM inputs and call preprocess/generate from `vlms/*` wrappers.
  3) Inject concept embeddings via layer utilities; manage keys/values per iteration.
  4) Package outputs into JSON; optionally save concept graph and visualizations.
"""

def run_reasoning(vlm_wrapper, activated_concepts, images, concept_embeddings, cfg):
    # TODO: Implement end-to-end reasoning and return {image_path: {prompt: output_text, ...}, ...}
    return {}
