# Concept As Graph (MyVLM‑Art)

Concept As Graph (MyVLM‑Art) teaches open VLMs to recognize and explain abstract art concepts — artist, style, genre — by combining CLIP‑space prototype heads with multi‑token concept embeddings injected into LLaVA. The system gates and injects concept tokens based on prototype similarity with per‑dimension calibration and global budget control, then reasons under natural or structured prompts.

Key ideas:
- Dual representation: CLIP prototype heads provide detection signals; multi‑token concept embeddings steer generation.
- Dynamic gating: threshold/backoff, per‑dimension Top‑K, global budget with fairness and priority.
- Two‑stage training: Stage A (keys) for stable identification; Stage B (keys+reasons) for explanation quality with attention regularization and weighted loss.
- Structured vs. natural prompts: structured outputs stabilize coverage; natural prose improves readability.

Results (WikiArt small subset):
- Natural prompts — macro F1 (artist/style/genre): 0.92/0.55/0.33 vs LLaVA baseline 0.47/0.39/0.09; multi‑label “all correct”: 0.30 vs 0.10.
- Structured prompts — macro F1: 0.92/0.46/0.72 vs baseline 0.34/0.36/0.29; concept coverage 0.65.

## Table of Contents
- Overview and Highlights
- Installation
- Data and Artifacts
- Prototypes and Signals
- Training Concept Embeddings
- Reasoning and Inference
- Evaluation
- Repository Layout
- Acknowledgments

## Installation
Create and activate the environment:
```
conda env create -f environment/environment.yaml
conda activate myvlm
```

## Data and Artifacts
Expected minimal files:
- `data/dataset/wikiart_5artists_dataset.json` — list of image records with `image` relative paths.
- `artifacts/prototypes_artist_trained.pt`, `artifacts/prototypes_style_trained.pt`, `artifacts/prototypes_genre_trained.pt` — trained CLIP prototype checkpoints.
- `artifacts/concept_signals_artist.json`, `..._style.json`, `..._genre.json` — per‑image concept signals exported from prototypes.
- `artifacts/synth_targets.csv` — training CSV with `image_path, artist, style, genre, target_keys, target_reason`.

Dataset curation guidelines are described in `final.md:180-185`.

## Prototypes and Signals
Train prototypes using `PrototypeHead` + `PrototypeTrainer` (write a small script that constructs a dataset of images per concept and calls `PrototypeTrainer.train`). Key classes:
- `concept_graph/prototypes/prototype_head.py:32-74,75-80,81-108` — build/load prototypes, extract per‑image similarity signals.
- `concept_graph/prototypes/prototype_trainer.py:59-157` — InfoNCE + margin with hard‑negative selection; unit‑norm updates.

Export concept signals (JSON or CSV) for downstream training:
```
python concept_graph/prototypes/export_signals.py \
  --dataset_json data/dataset/wikiart_5artists_dataset.json \
  --images_root data/dataset \
  --ckpt artifacts/prototypes_artist_trained.pt \
  --dimensions artist \
  --output artifacts/concept_signals_artist.json \
  --format json --normalize zscore --temperature 1.0
```
Repeat for `style` and `genre` checkpoints. Implementation references: `concept_graph/prototypes/export_signals.py:47-141,143-214`.

## Training Concept Embeddings
We inject multi‑token concept embeddings at the LLaVA language layer and train in two stages.

Config and entry point:
- Config defaults: `configs/myvlm_art_config.py:30-54,59-65` (threshold/backoff/budget; per‑dim Top‑K; fairness; priority; `max_reason_tokens`).
- Train script: `concept_graph/concept_embeddings/train_concept_embedding.py:11-37`.

Run:
```
python concept_graph/concept_embeddings/train_concept_embedding.py
```
Prerequisites:
- Place dataset JSON under `data/dataset`.
- Place concept signals in `artifacts/concept_signals_{artist,style,genre}.json`.
- Provide `artifacts/synth_targets.csv` with labels and optional reasons.

Core components:
- Injection layer: `concept_graph/concept_embeddings/multi_embed_layer.py:9-40,42-54,56-193,195-202` — `MultiTokenConceptLayer` with threshold/backoff/Top‑K and budget fairness; token appending scaled by score.
- Trainer: `concept_graph/concept_embeddings/trainer.py:47-285,286-341,451-484,342-374` — builds staged datasets, uses weighted loss in Stage B, initializes injection layer and sets dimension ranges.
- Dataset: `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:50-95,97-153,155-202` — collate with per‑segment token weights for keys vs reasons (Stage B).

## Reasoning and Inference
Reasoning baseline (pure LLaVA, no injection):
```
python concept_graph/reasoning/run_reasoning_baseline.py --cfg.data_root ./data --cfg.output_root ./outputs --prompt both
```
References: `concept_graph/reasoning/run_reasoning_baseline.py:120-182,184-201`.

Integrated reasoning with concept signals and injected embeddings:
```
python concept_graph/reasoning/run_reasoning.py --cfg.data_root ./data --cfg.output_root ./outputs --prompt both
```
References: `concept_graph/reasoning/run_reasoning.py:71-116` and runner `concept_graph/reasoning/reasoning_runner.py:201-401,169-198`.

Personalized captions (inference integration):
```
python inference/generate_personalized_captions.py --cfg.data_root ./data --cfg.output_root ./outputs
```
References: `inference/generate_personalized_captions.py:48-113,116-159,15-32`.

## Evaluation
Metrics and aggregation utilities:
- `myvlm_art/evaluation/metrics.py:38-70,134-156` — macro/weighted F1, multi‑label all‑correct, coverage.
- `myvlm_art/evaluation/run_metrics.py:275-359` — aggregates natural/structured runs and saliency metadata.

## Repository Layout
- Concept embeddings: `concept_graph/concept_embeddings/*` — layer, dataset, trainer, entry script.
- Reasoning: `concept_graph/reasoning/*` — baseline and integrated reasoning, prompt templates.
- Prototypes: `concept_graph/prototypes/*` — prototype head, trainer, signal export.
- Inference: `inference/*` — ensure injection layer, run captioning.
- VLM wrappers: `vlms/*` — LLaVA wrapper and base interface.
- Configs: `configs/*` — training/inference/art project config.
- Evaluation: `myvlm_art/evaluation/*` — metrics and aggregation.

## Acknowledgments
- Built on top of the MyVLM codebase and LLaVA. See upstream licenses and repos for original implementations.



For personalized captioning using MiniGPT-v2, please follow: 
```bash
python concept_embedding_training/train.py \
--config_path example_configs/minigpt_v2/concept_embedding_training_captioning.yaml
```
You may need to increase the number of iterations for personalized captioning with MiniGPT-v2. 
This will perform inference on both the captioning and referring expression comprehension personalization tasks.


# Inference
<p align="center">
<img src="docs/background.jpg" width="800px"/>  
<br>
VLMs possess <i>generic</i> knowledge, lacking a personal touch. With MyVLM we equip these models with the ability to comprehend user-specific concepts, tailoring the model specifically to <i>you</i>. 
MyVLM allows users to obtain personalized responses where outputs are no longer generic, but focus on communicating information about the target subject to the user.
</p>


## Original VLM Captioning
If you wish to run captioning using the original VLMs, you can do so using the following command: 
```bash
python inference/generate_original_captions.py \
--images_root /path/to/images \
--vlm_type <VLM_TYPE>
```
where `<VLM_TYPE>` is one of `BLIP2`, `LLAVA`, and `MINIGPT_V2`.

You can also run inference using: 
```bash
python inference/generate_original_captions.py \
--config_path example_configs/inference/original_vlm_inference.yaml
```

Please note that this script can likely be extended to run inference on other tasks/prompts by changing the input language 
instruction that is defined in Line 56: 
```python
inputs = vlm_wrapper.preprocess(image_path, prompt=VLM_TYPE_TO_PROMPT[cfg.vlm_type])
```


## MyVLM Inference
After training the concept heads and concept embeddings, you can run inference on a new set of images using: 
```bash
python inference/run_myvlm_inference.py \
--config_path example_configs/inference/myvlm_inference.yaml
```

All parameters are defined in `configs/inference_config.py` and closely follow the parameters defined in the training config detailed above.
The main parameters that should be modified are:
1. `concept_name`: same as above.
2. `concept_identifier`: same as above.
3. `concept_type`: same as above.
4. `vlm_type`: same as above.
5. `personalization_task`: same as above.
6. `image_paths`: either (1) a list of paths we want to run inference on; or (2) the directory containing the images we want to run inference on.
7. `checkpoint_path`: the path to all the trained concept embedding. This should contain a sub-directory for each concept and seed (e.g., `<output_root>/<concept_name>/seed_<seed>`).
8. `concept_head_path`: if working with objects, this should be the directory holding all the concept heads and seeds (e.g., `<concept_head_path>/<concept_name>/seed_<seed>`).
9. `seed`: random seed. This should be the same as used for the concept head and embedding training.
10. `iterations`: which optimization steps to run inference on. If `None`, we will run on all the checkpoints that were saved during the optimization process.
11. `prompts`: a list of strings defining the prompts to use for inference. If `None`, we will use a default list that is defined in `myvlm/common.py` (in `VLM_TO_PROMPTS`).

The output results will be saved to `<checkpoint_path>/<concept_name>/seed_<seed>/inference_outputs/inference_outputs_<VLM_TYPE>_<TASK>.json`, in the following format:
```
{
    "iteration_10": {
        "image_path_1": {
            "prompt1": "caption1",
            "prompt2": "caption2",
            ...
        },
        "image_path_2": {
            "prompt1": "caption1",
            "prompt2": "caption2",
            ...
        },
        ...
    },
    "iteration_20": {
        ...
    },
  ...
}
```


# Acknowledgements 
This code builds on code from the following repositories: 
- [Transformers](https://github.com/huggingface/transformers): we use the `transformers` library for various model architectures, including CLIP and BLIP-2.
- [LLaVA](https://github.com/haotian-liu/LLaVA): the official implementation of LLaVA-1.6.
- [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4): the official implementation of MiniGPT-v2.
- [GRACE](https://github.com/Thartvigsen/GRACE): official implementation of the GRACE model editing technique on which our original MyVLMLayer implementation was based.


# Citation
If you use this code for your research, please cite the following work:
```
@misc{alaluf2024myvlm,
      title={MyVLM: Personalizing VLMs for User-Specific Queries}, 
      author={Yuval Alaluf and Elad Richardson and Sergey Tulyakov and Kfir Aberman and Daniel Cohen-Or},
      year={2024},
      eprint={2403.14599},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
