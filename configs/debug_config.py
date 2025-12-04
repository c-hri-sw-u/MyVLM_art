# configs/debug_config.py
"""
本地 CPU/MPS 调试配置
使用小模型和少量数据
"""

DEBUG_MODE = True

# 使用小型 CLIP 模型（ViT-B-32 比 ViT-H-14 小很多）
CLIP_MODEL_DEBUG = "ViT-B-32"
CLIP_PRETRAINED_DEBUG = "openai"

# 调试时的数据量
DEBUG_N_SAMPLES = 2
DEBUG_BATCH_SIZE = 1
DEBUG_EPOCHS = 2

# 是否跳过 LLaVA（本地调试时跳过大模型）
SKIP_LLAVA_DEBUG = True