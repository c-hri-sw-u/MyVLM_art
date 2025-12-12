# Executive Summary

## 1. Introduction

Art analysis has long relied on expert connoisseurs who identify artistic signatures through subtle visual cues. While deep learning offers automation potential, professional adoption requires explainability through interpretable reasoning rather than opaque confidence scores. Vision‑language models (VLMs) naturally address this need, yet face fundamental limitations: training data remains static while the art world evolves continuously, and visual information in paintings often contains ineffable qualities that resist direct verbalization.

We propose a concept‑graph learning framework that unifies multi‑granular artistic understanding by treating concepts as a graph where each painting activates multiple related concepts across different dimensions. Each artistic concept—whether “Van Gogh” (artist), “Post‑Impressionism” (style), or “Landscape” (genre)—is encoded through dual representations: (1) Artistic Concept Heads as learnable prototypes in CLIP space for efficient retrieval, and (2) Artistic Concept Embeddings as learnable tokens for VLM‑based reasoning.

Our framework achieves data efficiency through concept reuse: each training image contributes to learning multiple concepts (artist + style + genre), and concepts naturally connect through co‑occurrence patterns. We validate this approach on 5 artists spanning diverse periods and styles, demonstrating improvements over zero‑shot baselines: macro F1 (artist/style/genre) 0.92/0.55/0.33 vs 0.47/0.39/0.09; multi‑label “all correct” 0.30 vs 0.10. With structured prompts (keys + brief reasoning), macro F1 reaches 0.92/0.46/0.72 and concept coverage 0.65.

---

## 3. Related Work

A short literature survey of 8 or more relevant papers.

### 3.1 AI for Art Analysis

Machine learning and deep learning has been increasingly applied to computational art analysis:

- **Art Authentication**: *Painting Authorship and Forgery Detection* (2022) and *Art Forgery Detection* (2024) provide accurate predictions using CNN and KAN. However, they lack interpretable reasoning about visual features, making them unsuitable for expert validation.

- **Context-aware Multimodal AI** (2024): Analyzed art evolution across five centuries using Stable Diffusion's latent representations, achieving moderate correlation (R²=0.203) between visual embeddings and art historical context. This work validates two key insights:
  1. Artistic concepts exist at multiple granularities beyond artist identity
  2. These concepts can be encoded through learned representations in vision models' feature spaces

- **GalleryGPT** (2024): Exposed fundamental limitations of large multimodal models in art analysis through systematic evaluation. Key findings:
  - Models rely on pre-memorized knowledge rather than perceptual visual reasoning
  - Struggle with formal elements like composition and brushwork
  - Fail when analyzing works by artists absent from training data
  - **Design implication**: Art analysis systems must learn artist-specific visual concepts from reference paintings rather than retrieving generic knowledge

### 3.2 Concept-Based Vision Models

Our framework builds on methods that encode visual concepts into structured representations for interpretable reasoning.

- **Concept Bottleneck Models (CBMs)** (2020): Pioneered using predefined semantic concepts as interpretable intermediate layers, enabling predictions through human-understandable attributes (e.g., "has feathers," "is blue"). 
  - **Limitations**:
    1. Rely on discrete binary activations that lack nuance
    2. Require extensive manual annotation of concept labels for every training image

- **Concept-as-Tree (CaT)** (2025): Addresses data scarcity in personalized VLM training through hierarchical concept decomposition. Structures concepts as trees (e.g., "cat" → dimensions: "appearance", "behavior" → attributes: "gray fur", "sitting"), then generates diverse training samples via diffusion models.
  - **Limitation for art**: Artistic concepts exhibit fundamentally different structure—a painting simultaneously embodies multiple independent dimensions (artist, style, genre, media) without hierarchical dependencies
  - **Our approach**: Concept-graph framework better captures lateral relationships through co-activation patterns rather than tree-structured decomposition

### 3.3 Personalized Vision-Language Models

Recent advances demonstrate that visual concepts can be efficiently learned through continuous embeddings with minimal training examples—a paradigm we adapt for artistic concept learning.

- **MyVLM** (2024): Introduced dual-component architecture for user-specific visual recognition:
  - **Concept Heads**: Detect concept presence via learned prototypes in CLIP space
  - **Concept Embeddings**: 768-dim vectors provide representations for VLM reasoning
  - Built upon frozen BLIP-2 and LLaVA backbones
  - Learns from merely 3-5 reference images per concept
  - **Limitations for our task**:
    - Targets instance-level recognition (e.g., "my cat," "my mug")
    - Handles single concepts per image
    - Our task requires learning abstract, generalizable concepts and jointly reasoning over multiple activated concepts

- **Yo'LLaVA** (2024): Employs multiple learnable tokens (k=16, 4096-dim each) with contrastive learning using hard negatives, training on 5 positive examples plus negative samples.

- **MC-LLaVA** (2024): Enhanced training efficiency through improved data construction.
  - **Limitation**: Absence of explicit "gate control" mechanisms makes adaptation to identification tasks challenging

**Our Design Choice**: We adopt MyVLM's dual-component architecture (heads for detection, embeddings for representation) but extend to multi-concept scenarios where each painting activates concepts across multiple dimensions. This design provides both explicit concept activation signals for classification and rich embeddings for interpretable reasoning.

### 3.4 Backbone Model

**LLaVA** (2023) serves as the backbone for our framework due to its strong visual-linguistic alignment and modular architecture facilitating efficient adaptation. We build upon **LLaVA-1.5-7B** as the base VLM.

---

## 4. Methods

### 4.1 Baseline Approach
We use pure LLaVA (`LLaVA‑1.6‑Vicuna‑7B`) as the baseline with no concept injection or key–value editing. Pipeline:
- Model and decoding: load official weights; use `temperature=0.2`, `top_p=0.7`; `max_new_tokens` follows the config (typically 64/200 depending on the script). Generation conditions on image + prompt only; no concept signals are provided.
- Prompt modes:
  - Natural paragraph: one cohesive English paragraph that naturally mentions artist, style, and genre, with a brief justification (brushwork, palette, composition).
  - Structured JSON: between `<BEGIN_JSON>` and `<END_JSON>`, return only a compact JSON with keys `artist/style/genre/evidence` where `evidence` is 2–3 concise sentences; no text outside the JSON.
- Inference and saving: iterate over images; run the selected prompt mode(s). If `both` is set, run both and merge per‑image outputs. Save all results keyed by image path to `reasoning_baseline.json`.
- Optional saliency: under the structured mode, run occlusion‑based sensitivity analysis (configurable grid) on the generated target, outputting a heatmap and summary stats to assess attention over regions relevant to identification and evidence.

### 4.2 Main Methods
We extend LLaVA with a concept‑graph and a dynamic multi‑token injection layer to jointly handle artist/style/genre and guide explainable reasoning.
- Components and data flow:
  - Concept heads (prototypes): train CLIP‑space prototypes for artist/style/genre and extract per‑image similarity signals, forming a `signals_map` and per‑dimension index ranges `dim_ranges` (three dimensions only; media is excluded).
  - Concept embeddings (multi‑token): each concrete concept owns up to `K_max` trainable tokens that, at train/inference time, are appended to the language layer hidden states according to activation strength, enabling variable‑length, interpretable injection.
- Activation and gating:
  - Threshold + backoff: activate when similarity `s ≥ τ`; if a dimension has no activation but candidates with `s ≥ τ − δ` exist, back off to Top‑1 in that dimension.
  - Per‑dimension Top‑K: truncate activated candidates by score to `Top‑K_per_dim` (default 2).
  - Global budget + fairness: total budget `max_concepts_per_sample` (default 3). Keep one per dimension first using priority `artist,style,genre`, then fill the remaining budget by score to avoid single‑dimension dominance.
  - g→k mapping and appending: for each selected concept, map activation `g∈[0,1]` to token count `k = ceil(g·K_max)`; L2‑normalize and append the top‑k tokens to the language layer output; each concept is appended at most once.
- K‑V mechanism and alignment:
  - `keys` are CLIP/detector vectors used for distance; `values` are the concept embedding tokens; multiple keys may map to the same value via `key_idx_to_value_idx`. `dim_ranges` ensures fair merging across dimensions.
  - The injection site is the LLaVA language layer selected by the VLM mapping; at inference we call `ensure_multi_token_layer` to replace the original layer with the multi‑token layer and load trained `values`.
- Training (two stages):
  - Stage A: semi‑structured supervision of three keys with `w_keys=1.0, w_reason=0.0` to stabilize identification.
  - Stage B: add brief reasons with `w_reason=0.2` and apply attention regularization at a fixed interval to prevent single‑concept token dominance and encourage multi‑concept coordination.
  - Batching and checkpoints: small batches; save embedding checkpoints and intermediate reasoning per iteration for selection and analysis.
- Inference and explainability:
  - Load concept embeddings and prototype signals, perform gating/injection, and generate text; support both natural and structured outputs; optionally produce prototype/LLaVA saliency heatmaps and stats.
  - Aggregate outputs per image and save to `reasoning.json` for evaluation and presentation.
- Key defaults:
  - `threshold τ=0.75`, `backoff δ=0.05`, `K_max=4`, `Top‑K_per_dim=2`, `max_concepts_per_sample=3`, `fairness=true`, `priority=artist,style,genre`
  - Training: `reg_lambda=0.0075` (attention regularization), `attn_reg_interval=4`, `batch_size=4`, `optimization_steps=100`, `grad_accum_steps=4`
  - Generation: `max_reason_tokens≈64` (can be increased for evaluation), prompt modes `natural/structured/both`

- Innovations beyond MyVLM:
  - Concept as graph and data efficiency: treat concepts as a graph across artist/style/genre; reuse samples across dimensions and leverage co‑occurrence to reduce sample complexity and improve reasoning coherence.
  - Dynamic gating: threshold/backoff, per‑dimension Top‑K, and a global budget with fairness to avoid single‑dimension dominance while remaining robust to near‑threshold noise.
  - Multi‑concept embedding: up to `K_max` tokens per concept with g→k variable‑length injection for interpretable control over how strongly each concept is represented.
  - Prototype head: CLIP‑space prototypes optimized with InfoNCE + margin and hard negatives; unit‑norm updates yield stable detection signals for gating.
  - Two‑stage training: Stage A (keys) stabilizes identification; Stage B (reasons + attention regularization) improves justification quality and multi‑concept coordination.

---

## 6. Code Overview

Key code authored/modified by our team, with references:

- Concept‑Graph Core
  - `concept_graph/concept_embeddings/multi_embed_layer.py:9-40,42-54,56-193,195-202` (Authored by us; Motivation: interpretable multi‑concept variable‑length injection with threshold/backoff/budget; Methods/Classes: `MultiTokenConceptLayer.__init__`, `initialize_values`, `set_keys`, `set_key_to_value_mapping`, `set_dim_ranges`, `forward`, `_compute_distances`): Dynamic multi‑token injection, K‑V init/mapping, threshold/backoff, per‑dim Top‑K, global budget + fairness, g→k mapping and token appending.
  - `concept_graph/concept_embeddings/trainer.py:47-285,286-341,451-484,342-344,345-368,369-374` (Authored by us; Motivation: staged A/B training + weighted CE + attention regularization for stable multi‑concept coordination; Methods/Classes: `MultiTokenEmbeddingTrainer.train`, `_build_stage_loaders`, `_ensure_multi_token_layer_init`, `_prompt_builder_a`, `_prompt_builder_b`, `_target_builder`): Dataset assembly for Stage A/B, attention regularization, layer replacement/init and value loading.
  - `concept_graph/concept_embeddings/datasets/llava_concept_graph_dataset.py:50-95,97-153,155-202` (Authored by us; Motivation: encode keys/reason into weighted‑CE training samples; Methods/Classes: `LLaVAConceptGraphDataset.__getitem__`, `collate_fn`, `from_csv`): LLaVA training dataset for concept‑graph (stage A/B), prompts/targets, collate.
  - `concept_graph/concept_embeddings/train_concept_embedding.py:11-37` (Authored by us; Motivation: minimal entry script wiring config/wrapper/trainer/save; Interface: `MyVLMArtConfig → LLaVAWrapper → MyLLaVA → MultiTokenEmbeddingTrainer`): Entry script wiring config, wrapper, trainer, and checkpoint save.
- Reasoning
  - `concept_graph/reasoning/run_reasoning_baseline.py:120-182,184-201` (Authored by us; Motivation: pure LLaVA baseline with saliency; Methods/Classes: `main`, `run_once`, `_llava_occlusion_saliency`, `_overlay_saliency`, `_saliency_stats`): Pure LLaVA baseline loop, prompting and JSON saving.
  - `concept_graph/reasoning/reasoning_runner.py:201-401,169-198,62-76,78-86` (Authored by us; Motivation: load signals/embeddings, inject and reason, export saliency/metadata; Methods/Classes: `run_reasoning`, `_occlusion_saliency`, `_overlay_saliency`, `_saliency_stats`): Load signals + ensure injection layer, prompt and generate with concept signals, prototype/LLaVA saliency.
  - `concept_graph/reasoning/run_reasoning.py:71-116` (Authored by us; Motivation: unified orchestrator selecting prompt mode and checkpoint; Methods/Classes: `main`): Orchestrator selecting prompt mode and checkpoints.
- Inference Integration
  - `inference/generate_personalized_captions.py:48-113,116-159,15-32` (Authored by us; Motivation: ensure injection layer at inference and load trained values; Methods/Classes: `load_concept_signals`, `ensure_multi_token_layer`, `_resolve_dataset_json`): Ensure multi‑token layer at inference, map dim ranges, load trained values and priorities.
  - `inference/inference_utils.py:17-28` (Modified from MyVLM; Motivation: reuse MyVLM interfaces to write `keys/values/mapping` to target layer at inference; Methods/Classes: `load_concept_embeddings`, `set_concept_embeddings`): Set concept embeddings (keys/values/mapping) onto target VLM layer for inference.
  - `inference/run_myvlm_inference.py:33-64,67-103` (Modified from MyVLM; Motivation: run iteration‑wise inference and save results; Methods/Classes: `main`, `run_inference`): Iteration‑wise inference over images/prompts with concept signals.
- Prompt Templates
  - `concept_graph/reasoning/inference_prompt_templates.py:24-35,38-44,47-48` (Authored by us; Motivation: provide natural/structured reasoning templates; Methods/Classes: `get_prompts`, `build_prompt_structured`, `build_prompt_natural`): Natural paragraph and structured JSON prompts for reasoning.
  - `concept_graph/datasets/synthesize_prompt_templates.py:4-37,40-57` (Authored by us; Motivation: training‑time semi‑structured keys/reason templates; Methods/Classes: `build_prompt`, `build_target`): Training‑time semi‑structured keys and optional reasoning targets.
- Prototypes
  - `concept_graph/prototypes/prototype_trainer.py:59-157` (Authored by us; Motivation: optimize prototypes in CLIP space for separability; Methods/Classes: `PrototypeTrainerConfig`, `PrototypeTrainer.train`): InfoNCE + margin with hard‑negative selection; unit‑norm prototype updates.
  - `concept_graph/reasoning/reasoning_runner.py:224-233` (Authored by us; Motivation: load prototypes per dimension and map idx→concept names; Methods/Classes: uses `PrototypeHead.load_prototypes`): Load trained prototype heads per dimension and map idx→concept names.
  - `concept_graph/prototypes/prototype_head.py:32-74,75-80,81-108` (Authored by us; Motivation: build/load prototypes and extract similarity signals; Methods/Classes: `PrototypeHead.build_prototypes`, `load_prototypes`, `extract_signal`): Prototype head in CLIP space; build/load and per‑image signal extraction.
  - `concept_graph/prototypes/export_signals.py:47-141,143-214` (Authored by us; Motivation: batch export per‑image concept signals to JSON/CSV; Methods/Classes: `main`): Export per‑image concept signals JSON/CSV with normalization/temperature options.
- VLM Wrappers
  - `vlms/llava_wrapper.py:64-71,72-87` (Modified from MyVLM; Motivation: expose `preprocess/generate` and support `concept_signals` passthrough; Methods/Classes: `LLaVAWrapper.preprocess`, `LLaVAWrapper.generate`, `prepare_inputs`): Preprocess with stop criteria and generate; accepts optional `concept_signals` in main framework.
  - `vlms/vlm_wrapper.py:15-40` (Modified from MyVLM; Motivation: unify VLM interface and signal preparation; Methods/Classes: `VLMWrapper.preprocess`, `VLMWrapper.generate`, `prepare_concept_signals`): Base VLM wrapper interface for preprocess/generate and signal preparation.
  - `myvlm/myllava.py:18-58,59-69,95-116` (Modified from MyVLM; Motivation: add validation, attention regularization, and custom collate; Methods/Classes: `MyLLaVA.validate`, `_compute_regularization_loss`, `_collate_func`, `_init_datasets`): LLaVA‑specific validation, attention regularization, and collate pipeline.
- Config
  - `configs/myvlm_art_config.py:30-54,59-65` (Authored by us; Motivation: declare threshold/backoff/budget and saliency parameters centrally; Methods/Classes: `MyVLMArtConfig`): Defaults for threshold/backoff, per‑dim Top‑K, fairness/priority, max tokens/budget, saliency, `max_reason_tokens`.
  - `configs/inference_config.py:1-38,62-70` (Modified from MyVLM; Motivation: inference configuration and path derivation; Methods/Classes: `InferenceConfig.__post_init__`, `_verify_concept_embeddings_exist`, `_verify_concept_heads_exist`): Inference config schema, prompt selection, and thresholds.
  - `configs/train_config.py:1-35` (Modified from MyVLM; Motivation: align original training config with project interfaces; Methods/Classes: `EmbeddingTrainingConfig`): Embedding training config for concept name/type, VLM, task, and IO paths.
 - Evaluation
  - `myvlm_art/evaluation/metrics.py:38-70,134-156` (Authored by us; Motivation: measure classification and coverage metrics; Methods/Classes: `per_concept_breakdown`, `macro_weighted_f1`, `multi_label_all_correct`, `coverage_score`): Per‑concept breakdown, macro/weighted F1, multi‑label all‑correct, coverage.
  - `myvlm_art/evaluation/run_metrics.py:275-359` (Authored by us; Motivation: aggregate natural/structured results and saliency metadata; Methods/Classes: aggregation pipeline functions): Aggregate natural/structured metrics, per‑concept breakdown, saliency metadata.

Notes:
- We keep code style consistent with upstream LLaVA; injection occurs at the mapped language layer while prototypes remain in CLIP space.
- Figures in the Appendix reference these sections to illustrate gating and injection outcomes.

---

## 7. Timeline

| Activity | Hours Spent Person 1 | Hours Spent Person 2 |
|----------|-------------|-------------|
| Reading papers/dataset websites | 12h | 18h |
| Reading code documentation (PyTorch, Torchvision, etc.) | 1.5h | 6h |
| Understanding existing implementation | 6h | 6h |
| Compiling/running existing code | 0h | 6h |
| Modifying/writing new code | 24h | 3h |
| Writing experiment scripts | 12h | 6h |
| Running experiments | 16h | 12h |
| Compiling results | 6h | 6h |
| Writing this document | 3h | 6h |
| **Total** | **[total hours]** |

---

## 8. Research Log

### 8.1 Dataset Curation

Initially, we selected 35 images per artist, assuming this would provide sufficient data for training. However, we quickly discovered a critical imbalance problem. While we had enough images per artist, the distribution across other concepts (style and genre) was highly uneven. For example, Picasso's diverse artistic career meant some of his styles had only 1-2 images, while Monet's work was almost exclusively Impressionist. This imbalance would cause the prototype head to perform inconsistently across different concepts, as some concepts had too few samples to learn reliable prototypes.

To address this, we established a filtering rule: each concept must have at least 5 images (following MyVLM's recommendation) and at most 18 images to prevent any single concept from dominating. This required carefully curating the dataset, sometimes reducing the number of images per artist to ensure balanced concept representation. The final dataset contained 86 training samples across 5 artists, with each concept dimension (artist, style, genre) having between 5-18 samples. This balanced distribution ensures that prototype head training and concept embedding learning work consistently across all concepts, rather than being biased toward concepts with more samples.

### 8.2 Prototype Head Development

We developed a PrototypeHead module using CLIP (ViT-H-14-384) to extract concept signals from images. The idea was to compute cosine similarity between image features and pre-trained concept prototypes, generating probability distributions that indicate how well an image matches each concept. The challenge was that prototype training required a separate pipeline before concept embedding training, which added complexity.

We solved this by training prototypes separately for each dimension (artist, style, genre) and saving them to reusable checkpoint files, then integrating prototype computation into the main training pipeline via precomputed concept_signals JSON files.

We also observed substantial heterogeneity across dimensions: raw prototype signals differ because concepts vary in separability. For example, artists are naturally more separable—top‑1 signals often exceed 0.80 and are well separated from top‑2—whereas genre is weaker, with top‑1 around 0.60–0.70 and small gaps to later ranks. A single global threshold therefore fails.
To address this we did two things:
1) Hard negatives in training: for each concept we include negatives drawn from other concepts in the same dimension and from visually confusable neighbors. Losses combine contrastive objectives on prototypes (InfoNCE/triplet margin) with weighted cross‑entropy on concept tokens. Batches maintain a positive:negative ratio, and we mine hard negatives using current prototype similarities to sharpen decision boundaries.
2) Per‑dimension calibration: scores are standardized (z‑score) and temperature‑scaled per dimension to make artist/style/genre signals comparable. Thresholds are selected per dimension via validation ROC/F1, with optional percentile‑based backoff to control the multi‑concept injection budget. This combination reduces score bias and yields stable gating under a unified budget.

### 8.3 Training Target
We initially planned a single‑stage target with one output. Splitting the synthesis target into keys (artist/style/genre labels) and reasons (short textual justifications), we considered training on keys only versus keys + reasons. We later adopted a two‑stage scheme:
Stage A — Keys: train identification of keys only using weighted cross‑entropy, optionally conditioning on prompt scaffolds; reasons are not supervised. This stage builds a stable concept memory and calibrates gating/thresholds.
Stage B — Keys + Reasons: fine‑tune jointly on keys and reasons with multi‑task objectives—classification for keys and next‑token loss for reasons. We apply attention regularization to maintain concept‑token focus and reweight tokens so keys dominate loss while reasons improve explanation quality and coherence.
Dataset assembly mirrors this split: Stage A uses balanced positives/negatives per concept with prototype signals; Stage B augments each sample with synthesized reasons and structured prompts. The staged approach yields clearer identification and more precise justifications than a single‑stage setup.

### 8.4 Initial Training Implementation on Google Colab

When we first tried to run training on Google Colab, we immediately hit environment compatibility issues. The `transformers` library version conflicted with the LLaVA model, causing errors like unexpected `cache_position` arguments and quantization configuration conflicts. We worked around this by implementing runtime monkey patches in Colab cells that remove problematic arguments before they reach the model. This wasn't ideal, but it allowed us to proceed with training.

Memory constraints became a major issue when we tried to increase batch size from 1 to 2. The weighted loss calculation we implemented for Stage B significantly increased memory footprint, causing Out-of-Memory errors. We solved this by keeping batch_size at 1, adding explicit memory management with garbage collection and CUDA cache clearing, and optimizing the weighted loss calculation to minimize intermediate tensor storage.

The most critical bug we discovered was that Stage B's weighted loss wasn't actually being used. We had implemented `token_weights` to give higher importance to keys (artist/style/genre) than reasoning text, but the model was still using the standard unweighted loss. This meant Stage B wasn't learning the intended distinction. We fixed this by manually implementing weighted cross-entropy loss, extracting token_weights before the model forward pass and applying them during loss calculation.

### 8.5 Image Token Alignment Problem - stage B

During Stage B training, we encountered a shape mismatch error when calculating weighted loss. The error message showed that logits had 755 tokens while labels had only 168 tokens. After investigation, we realized that LLaVA's logits include image tokens at the beginning of the sequence, while labels only contain text tokens. Our weighted loss calculation was trying to compare sequences of different lengths. We fixed this by aligning the sequences, slicing the logits from the end to match the labels' length, effectively removing the image token portion before loss calculation.

### 8.6 Training Refinements and Hyperparameter Tuning

We integrated Weights & Biases (wandb) logging to monitor training progress in real-time. This allowed us to observe that Stage A loss converged quickly, reaching stable low values by epoch 3, while Stage B loss remained relatively stable throughout training. This observation led us to align Stage A and Stage B to use the same number of epochs, rather than having Stage B use half the epochs as originally planned.

Through iterative testing, we found that a learning rate of 0.3 worked best for our small dataset. Higher rates (1.0, 0.5) caused instability, while lower rates (0.05) converged too slowly. For regularization, we discovered that 0.01 was appropriate for our 86-sample dataset - lower values risked overfitting, while higher values might prevent learning. We settled on 5 epochs total, which gives Stage A time to converge (by epoch 3) and Stage B time to refine the reasoning generation.

### 8.7 Deviations from Plan
We initially intended to prioritize niche artists/styles/genres. However, the base LLaVA baseline underperformed even on popular concepts, while our synthesized targets were more reliable for those. To isolate the effect of our method and avoid confounds from extreme data scarcity, we pivoted to evaluating on popular concepts first.
For similar reasons, we did not benchmark against frontier proprietary VLMs (e.g., ChatGPT‑5.1, Gemini 3 Pro). These models likely already memorize many art concepts, and their closed training regimes are out of scope. Our goal is to demonstrate teachability relative to an open LLaVA baseline; consistent improvements there are sufficient for the claims of this work. Future work will expand to niche concepts and stronger open baselines.

---

## 9. Conclusion

Our concept–graph framework (dual representations (CLIP prototype heads + multi‑token concept embeddings) with dynamic gating) improves explainability and performance over a pure LLaVA baseline: under natural prompts, macro F1 (artist/style/genre) 0.92/0.55/0.33 vs baseline 0.47/0.39/0.09, with multi‑label “all correct” 0.30 vs 0.10; under structured prompts, macro F1 0.92/0.46/0.72 vs baseline 0.34/0.36/0.29, multi‑label “all correct” 0.24, and concept coverage 0.65. Injecting concept tokens guided by prototype similarity (threshold/backoff, per‑dimension Top‑K, global budget + fairness) yields more consistent identification of artist/style/genre and clearer, concise justifications. Structured prompting complements the approach by stabilizing outputs and increasing concept coverage.

Main findings:
- Beyond personalized objects, the method learns abstract art concepts (artist, style, genre) effectively and produces explanations tied to visual evidence.
- Multi‑concept reasoning is feasible: jointly handling artist/style/genre yields consistent identification and coherent justifications.
- Two‑stage training is feasible: Stage A (keys) stabilizes identification; Stage B (reasons + attention regularization) improves explanation quality and coordination.
- Concept reuse across dimensions improves data efficiency and reasoning coherence compared to single‑task setups.
- Dynamic multi‑token injection (g→k) with fairness prevents single‑dimension dominance and yields more balanced attributions.
- Attention regularization (interval‑based) stabilizes training and mitigates over‑reliance on a few concept tokens.
- Best performance is on artist, while style is weakest; likely due to limited style training data and CLIP’s inherently higher separability on artist (vs. style) observed in prior work.
- Structured prompts slightly outperform natural prompts overall, but this is not a robust rule: (1) the max output token limit can truncate natural outputs; (2) results are from a single sampling; rigorous evaluation should average multiple samples.

Limitations:
- Scope: small WikiArt subset (5 artists) and three dimensions; media is excluded in the current implementation.
- Calibration: per‑dimension score normalization/temperature is not fully tuned, which may affect cross‑dimension comparability.
- Generalization: reliance on CLIP prototypes and limited tokens per concept may constrain transfer to broader art domains.

Future work:
- Scale the dataset (artists/styles/genres) and add hierarchical relations to the concept graph; incorporate media when feasible.
- Calibrate per‑dimension scores (z‑score/temperature) and study threshold/backoff sensitivity under different budgets.
- Strengthen base VLMs and investigate layer‑wise injection sites; unify training for keys + reasons with multi‑task objectives.
- Human‑in‑the‑loop evaluation of explanations (coverage, correctness, specificity) with art experts; expand saliency analyses.
- Systematize ablations on Top‑K/fairness/attention regularization to quantify their contributions and trade‑offs.
