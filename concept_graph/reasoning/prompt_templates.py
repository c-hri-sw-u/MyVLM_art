"""
简要介绍 (ZH):
  解释性推理的提示词模板库。根据已激活的多粒度概念（艺术家/风格/题材/媒介），
  自动生成用于 LLaVA 的对话式提示词，用于“解释为何是某艺术家”或“描述作品特征”。

Overview (EN):
  Prompt templates for explainable reasoning. Given activated multi-granular concepts, compose conversational
  prompts for LLaVA to produce natural language justifications and descriptive outputs.

TODOs (详细):
  1) 设计多种模板：分类解释、风格描述、题材/媒介结合说明、反事实对比等
  2) 模板填充：将激活概念按维度插入占位符，控制长度与可读性
  3) 与 reasoning_runner.py 对接：提供函数 get_prompts(activated_concepts)
"""

def get_prompts(activated_concepts):
    # TODO: 根据 activated_concepts (字典或列表结构) 生成多种提示词
    return ["Please explain why this painting is attributed to <artist> given its <style>, <genre>, and <media>."]
