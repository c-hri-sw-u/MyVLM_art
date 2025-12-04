# Concepts as Graph: Multi-Granular Concept Learning for Explainable Art Analysis

**Authors:** Risa Xie & Chris Wu  
**Course:** 10-423/623 Generative AI  
**Date:** November 25, 2025

---

## Project Overview

A concept-graph learning framework for artist identification that provides explainable predictions through multi-granular artistic concepts. The system learns from 10 reference paintings per artist and generates interpretable reasoning through activated concept networks.

---

## Dataset & Task

### Task: Artist Identification
- **Input:** Query painting image
- **Output:** Predicted artist + activated concept graph + natural language reasoning
- **Setting:** Few-shot learning with 10 reference paintings per artist

### Dataset: WikiArt Curated Subset

**Artists (5):**
- Vincent van Gogh (Post-Impressionism, 1853-1890)
- Claude Monet (Impressionism, 1840-1926)
- Pablo Picasso (Cubism, 1881-1973)
- Rembrandt van Rijn (Baroque/Realism, 1606-1669)
- Leonardo da Vinci (Renaissance, 1452-1519)

**Attribute Dimensions (3):**
1. **Genre:** Portrait, Landscape, Still Life, Religious, Cityscape, Abstract, Genre Painting
2. **Style:** Impressionism, Post-Impressionism, Cubism, Baroque, Renaissance, Realism
3. **Media:** Oil on canvas, Watercolor, Drawing, Fresco, Tempera

**Dataset Statistics:**

| Split | Images/Artist | Total Images | Total Concepts |
|-------|---------------|--------------|----------------|
| Train | 10 | 50 | ~18 |
| Val | 5 | 25 | ~18 |
| Test | 20 | 100 | ~18 |

*Total concepts = Artists (5) + Genres (~6) + Styles (~5) + Media (~3)*

**Data Efficiency:** Each training image contributes to learning 4 concepts simultaneously (artist + style + genre + media), yielding 50 images × 4 = 200 concept-image training pairs.

---

## Technical Approach

### Architecture Components

Our framework comprises three main components:

#### 1. Artistic Concept Heads
- Learnable prototypes **p_c ∈ ℝ^512** in CLIP's visual feature space
- **Initialization:** Mean of CLIP features for concept c:
  ```
  p_c = (1/N) Σ CLIP(x_i)
  ```
- **Activation:** Cosine similarity scoring:
  ```
  s_c = cos_sim(CLIP(x_q), p_c)
  ```
- **Threshold:** τ = 0.7 (concepts with s_c > τ are activated)

#### 2. Artistic Concept Embeddings
- Learnable tokens **e_c ∈ ℝ^4096** for VLM reasoning (LLaVA)
- **Initialization:** Random `e_c ~ N(0, 0.01)`, scaled to match vision token norms
- **Usage:** Concatenate activated embeddings: `[image_features, e_c1, e_c2, ...]`
- **Regularization:** Attention regularization to prevent dominance:
  ```
  L_attn = ||softmax(tokens · e_c^T)||²₂
  ```

#### 3. Multi-Concept Inference Pipeline
1. **Concept Activation:** Compute {s_c} for all concepts; activate those exceeding threshold
2. **Concept Network:** Activated concepts form a multi-granular network
3. **VLM Reasoning:** Concatenate activated embeddings, generate explanation via LLaVA

### Training Strategy

**Phase 1: Initialization**
- Initialize concept heads as mean CLIP features per concept
- Initialize concept embeddings randomly with norm scaling

**Phase 2: Joint Optimization**

For each training image x with ground truth labels {c_artist, c_style, c_genre, c_media}:

**Loss Component 1 - Concept Head Learning:**
```
L_head = Σ_dim L_InfoNCE(x, pos_c, neg)
```
Uses contrastive learning with K=3 random negative concepts from the same dimension.

**Loss Component 2 - Concept Embedding Learning:**
```
L_embed = L_CE(generated_reasoning, target_reasoning)
```
Standard language modeling loss.

**Total Loss:**
```
L_total = L_head + α·L_embed + β·L_reg
```

**Hyperparameters:**
- α = 0.5
- β = 0.1
- Learning rates: concept heads (1e-4), embeddings (5e-5)
- Batch size: 8
- Epochs: 50 with early stopping
- Optimizer: AdamW

---

## Implementation Details

### Model Configuration
- **CLIP:** ViT-B/16 (frozen, 512-dim features)
- **VLM:** LLaVA-1.5-7B (frozen except concept embeddings)
- **Concept embeddings:** 4 tokens × 4096 dim per concept
- **Activation threshold:** τ = 0.7

### Training Configuration
- **Optimizer:** AdamW (lr=1e-4 heads, 5e-5 embeddings)
- **Batch size:** 8 images
- **Epochs:** 50 (early stopping on validation)
- **Loss weight:** λ = 0.5

### Computational Resources
- **Platform:** Kaggle P100 GPU (16GB VRAM)
- **Estimated training time:** ~3 hours
- **Inference speed:** ~0.5s per image

---

## Baseline Results

### Zero-shot LLaVA-1.5 Performance

**Overall Metrics:**
- Overall Accuracy: 55.4%
- Macro Avg F1: 0.45
- Weighted Avg F1: 0.54

**Per-Artist Performance:**

| Artist | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Claude Monet | 0.55 | 0.89 | 0.68 |
| Leonardo da Vinci | 1.00 | 0.29 | 0.44 |
| Pablo Picasso | 0.52 | 0.69 | 0.59 |
| Rembrandt | 0.62 | 0.60 | 0.61 |
| Vincent van Gogh | 0.39 | 0.31 | 0.35 |

**Key Observations:**
- High variance across artists
- Leonardo da Vinci: perfect precision (1.00) but very low recall (0.29)
- Claude Monet: high recall (0.89) but moderate precision (0.55)
- Van Gogh shows lowest F1 (0.35) despite being well-represented in pre-training data
- Model relies on memorization rather than learning artist-specific visual patterns

---

## Evaluation Metrics

### Primary Metrics
1. Artist Identification Accuracy
2. Per-Artist Precision/Recall/F1

### Concept Activation Metrics
1. Precision/Recall/F1 per concept dimension
2. Multi-label Accuracy (all 4 concepts correct)

### Explainability Evaluation
1. **Concept Coverage Score:** Whether generated reasoning mentions all activated concepts
2. **Reasoning Coherence:** Human evaluation on completeness, grounding, coherence, and fluency

---

## Planned Experiments

### RQ1 & RQ2: Artist Identification Performance

| Method | Top-1 Acc | Mean F1 |
|--------|-----------|---------|
| Zero-shot LLaVA-1.5 | 0.554 | 0.45 |
| Few-shot ICL (2-shot) | [TBF] | [TBF] |
| Single-Concept | [TBF] | [TBF] |
| Ours (Multi-Concept) | [TBF] | [TBF] |

### RQ3: Concept Activation Performance

| Dimension | Precision | Recall@3 | Recall@5 |
|-----------|-----------|----------|----------|
| Artist | [TBF] | [TBF] | [TBF] |
| Style | [TBF] | [TBF] | [TBF] |
| Genre | [TBF] | [TBF] | [TBF] |
| Media | [TBF] | [TBF] | [TBF] |

### Ablation Study

| Configuration | Top-1 Acc | Avg. Concepts |
|---------------|-----------|---------------|
| Artist only | [TBF] | 1.0 |
| Artist + Style | [TBF] | ~2.5 |
| Artist + Style + Genre | [TBF] | ~3.5 |
| All 4 Dimensions | [TBF] | ~4.2 |

### Activation Threshold Sensitivity

| Threshold τ | Avg. Concepts | Precision | Top-1 Acc |
|-------------|---------------|-----------|-----------|
| 0.5 | [TBF] | [TBF] | [TBF] |
| 0.6 | [TBF] | [TBF] | [TBF] |
| 0.7 (default) | [TBF] | [TBF] | [TBF] |
| 0.8 | [TBF] | [TBF] | [TBF] |
| 0.9 | [TBF] | [TBF] | [TBF] |

---

## Key References

- Lu et al. [2024] - MyVLM: Personalizing VLMs for user-specific queries
- Liu et al. [2024] - Visual instruction tuning (LLaVA)
- Radford et al. [2021] - CLIP: Learning transferable visual models
- Koh et al. [2020] - Concept bottleneck models
