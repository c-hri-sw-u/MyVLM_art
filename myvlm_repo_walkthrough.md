toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/myvlm/myblip2.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/myvlm/myllava.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/myvlm/myminigpt_v2.py
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 2 items
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 2 items
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/myvlm/utils.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/vlms/blip2_wrapper.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/vlms/llava_wrapper.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/concept_embedding_training/datasets/minigpt_v2_dataset.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /home/ubuntu/MyVLM_art/vlms/minigpt_wrapper.py
          
**项目概览**
- 目标：在现有视觉-语言模型（BLIP-2、LLaVA、MiniGPT‑v2）中注入“概念嵌入”，让模型理解并在输出中自然融入用户特定概念（人/物）。
- 两步法：先训练“概念头”识别概念是否出现，再训练“概念嵌入”引导语言模型把概念写进输出。
- 适用场景：个性化图像描述、个性化问答、MiniGPT‑v2的指代定位。
- 快速了解：训练主流程在 `concept_embedding_training/train.py:35–95`；注入机制在 `myvlm/myvlm_layer.py:28–60, 61–80`。

**使用流程**
- 环境准备：`conda env create -f environment/environment.yaml && conda activate myvlm`（详见 `README.md:44–48`）
- 训练概念头（对象类概念）：`python concept_heads/clip/train.py --config_path example_configs/concept_head_training.yaml`
- 训练概念嵌入：`python concept_embedding_training/train.py --config_path example_configs/<vlm>/<config>.yaml`
- 原始模型推理：`python inference/generate_original_captions.py --config_path example_configs/inference/original_vlm_inference.yaml`
- MyVLM推理：`python inference/run_myvlm_inference.py --config_path example_configs/inference/myvlm_inference.yaml`

**目录结构**
- `myvlm/`：概念注入通用逻辑与各VLM适配
- `vlms/`：三种VLM封装与其上游实现（LLaVA、MiniGPT‑v2、BLIP‑2）
- `concept_heads/`：对象用CLIP线性分类器、人物用insightface人脸识别
- `concept_embedding_training/`：数据加载与概念嵌入训练脚本
- `configs/`：训练/推理配置数据类
- `inference/`：生成原始/个性化输出的脚本与工具
- `example_configs/`：可直接运行的配置示例
- `data/`：示例数据集结构与图片资源

**关键文件（按模块）**
- 概念注入核心
  - `myvlm/myvlm.py`：包装指定VLM层为可注入层、训练循环与校验入口（训练循环 `myvlm/myvlm.py:63–106`；层替换 `myvlm/myvlm.py:34–43`）
  - `myvlm/myvlm_layer.py`：注入层，按阈值将“概念嵌入”拼接进层输出；兼容对象/单人/多人三种信号（前向与注入分支 `myvlm/myvlm_layer.py:28–60, 61–80`）
  - `myvlm/common.py`：任务/模型/层位点映射、嵌入维度、默认提示词等（层位点与维度 `myvlm/common.py:31–41`；提示词 `myvlm/common.py:43–61`）
  - `myvlm/utils.py`：按“路径字符串”定位并替换子模块（`myvlm/utils.py:5–20, 29–30`）

- VLM 封装
  - `vlms/vlm_wrapper.py`：统一接口，含预处理与生成的抽象定义（`vlms/vlm_wrapper.py:15–40`）
  - `vlms/blip2_wrapper.py`：BLIP‑2加载与生成（`vlms/blip2_wrapper.py:17–27, 36–41`）
  - `vlms/llava_wrapper.py`：LLaVA加载、对话格式、生成与停止条件（生成 `vlms/llava_wrapper.py:64–79`）
  - `vlms/minigpt_wrapper.py`：MiniGPT‑v2加载、指令拼装、生成与指代框绘制（生成 `vlms/minigpt_wrapper.py:58–66`；框绘制 `vlms/minigpt_wrapper.py:81–102`）
  - 上游实现目录：`vlms/llava/*` 与 `vlms/minigpt4/*` 为第三方模型代码与工具，不需改动，作为依赖使用。

- 概念头（概念存在性检测）
  - `concept_heads/concept_head.py`：统一抽象接口（`concept_heads/concept_head.py:7–15`）
  - `concept_heads/clip/head.py`：对象类概念头，基于 CLIP 特征 + 线性分类器；支持单/多概念（信号提取 `concept_heads/clip/head.py:30–39`）
  - `concept_heads/face_recognition/head.py`：人物类概念头，基于 insightface 人脸嵌入（`concept_heads/face_recognition/head.py:19–29`）
  - 训练对象概念头：`concept_heads/clip/concept_head_training/train.py`（入口 `:7–11`），配置在 `concept_heads/clip/concept_head_training/config.py:5–31`

- 概念嵌入训练
  - `concept_embedding_training/train.py`：主入口，加载VLM与概念头、数据、训练并保存嵌入；随后在验证集上跑推理（主流程 `concept_embedding_training/train.py:35–95`）
  - `concept_embedding_training/data_utils.py`：数据组织、增强、人物多脸过滤、额外VQA数据加载（多脸过滤 `concept_embedding_training/data_utils.py:84–125`）
  - 任务数据集：
    - BLIP‑2：`concept_embedding_training/datasets/blip2_dataset.py`（采样目标、图像预处理 `:33–43, 45–49`）
    - LLaVA：`concept_embedding_training/datasets/llava_dataset.py`（指令编码与标签对齐 `:63–90, 91–125`）
    - MiniGPT‑v2：`concept_embedding_training/datasets/minigpt_v2_dataset.py`（指令池与样本构造 `:34–38, 42–57`）

- 推理与工具
  - `inference/run_myvlm_inference.py`：加载概念头与嵌入，批量迭代执行推理；MiniGPT‑v2支持指代定位图保存（主循环 `inference/run_myvlm_inference.py:67–103`）
  - `inference/inference_utils.py`：将某一迭代的概念嵌入写入目标VLM层（`inference/inference_utils.py:17–28`）
  - `inference/generate_original_captions.py`：按VLM生成原始（非个性化）描述并保存（`inference/generate_original_captions.py:39–62`）
  - `inference/generate_augmented_vqa_data.py`：生成LLaVA的额外个性化VQA数据（见 `README.md:132–147`）

- 配置与示例
  - `configs/train_config.py`：训练配置数据类与路径校验（`configs/train_config.py:9–57, 58–67`）
  - `configs/inference_config.py`：推理配置数据类与路径/阈值/默认提示词（`configs/inference_config.py:11–43, 44–69`）
  - `example_configs/`：各模型与任务的 yaml 示例，可直接运行

**数据与输出**
- 数据组织：每个概念一个目录，含图片、`captions.json` 或 `captions_augmented.json`，LLaVA可选 `additional_llava_vqa_data.json`（见 `README.md:104–127, 132–147`）
- 训练输出：
  - 概念嵌入检查点：`concept_embeddings_<VLM>_<TASK>.pt`（格式见 `README.md:186–203`）
  - 推理结果：`inference_outputs_<VLM>_<TASK>.json`（格式见 `README.md:303–324`）
- 推理时注入：按配置 `iterations` 写入不同迭代嵌入，并按提示词列表生成结果（`inference/run_myvlm_inference.py:77–93`）

**注意事项**
- MiniGPT‑v2需设置 HuggingFace Token 与检查点路径（`myvlm/common.py:65–68`）
- 阈值：对象 `0.5`、人物 `0.675`（推理配置自动设置，`configs/inference_config.py:61–63`）
- 概念名替换：训练时将 `concept_name` 替换成简短标识 `concept_identifier`，保证语言自然（`concept_embedding_training/data_utils.py:49–54`）

如果你希望我把以上内容整理成项目根的简短中文指南文件并加到仓库里，我可以直接创建一个 `PROJECT_GUIDE_CN.md`；或者你也可以让我为你生成更详细的“分目录逐文件表”。