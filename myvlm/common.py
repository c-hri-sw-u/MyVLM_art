import os
import random
from enum import Enum, auto

import numpy as np
import torch

# myvlm/common.py（在文件顶部添加）
def get_device():
    """
    自动检测最佳可用设备
    - CUDA (NVIDIA GPU)
    - MPS (Apple Silicon)  
    - CPU (fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_info():
    """打印设备信息，方便调试"""
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon GPU (MPS)")
    else:
        print("  CPU mode (no GPU acceleration)")
    
    return device

# 全局设备变量（供其他模块使用）
DEVICE = get_device()


class ConceptType(Enum):
    OBJECT = auto()
    PERSON = auto()


class VLMType(str, Enum):
    BLIP2 = 'blip-2'
    LLAVA = 'llava'
    MINIGPT_V2 = 'minigpt-v2'


class PersonalizationTask(str, Enum):
    CAPTIONING = 'captioning'
    VQA = 'vqa'
    REC = 'rec'


class MyVLMLayerMode(Enum):
    TRAIN = auto()
    INFERENCE = auto()


VLM_TO_LAYER = {
    VLMType.BLIP2: "model.vision_model.encoder.layers[38].mlp.fc2",
    VLMType.LLAVA: "model.model.mm_projector.linear2",
    VLMType.MINIGPT_V2: "model.llama_proj"
}

VLM_TO_EMBEDDING_DIM = {
    VLMType.BLIP2: 1408,
    VLMType.LLAVA: 4096,
    VLMType.MINIGPT_V2: 4096
}

VLM_TO_PROMPTS = {
    VLMType.BLIP2: {
        PersonalizationTask.CAPTIONING: [''],
    },
    VLMType.LLAVA: {
        PersonalizationTask.CAPTIONING: ['Please caption this image of {concept}.'],
        PersonalizationTask.VQA: [
            'Where is {concept} in the image?',
            'Where is {concept} positioned in the image?',
            'What is {concept} doing in the image?',
        ],
    },
    VLMType.MINIGPT_V2: {
        PersonalizationTask.CAPTIONING: [
            '[caption] A short image caption of {concept}:',
            '[refer] {concept} in the image'
        ],
    }
}

VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

CLIP_MODEL_NAME = "DFN5B-CLIP-ViT-H-14-384"
MINIGPT_V2_CKPT_PATH = "/path/to/minigptv2_checkpoint.pth"
HF_TOKEN_FOR_LLAMA = 'IF WORKING WITH MINIGPT_V2, ENTER YOUR HF TOKEN FOR DOWNLOAIDNG LLAMA WEIGHTS'


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
