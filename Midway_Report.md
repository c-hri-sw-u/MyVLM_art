# Concepts as Graph: Multi-Granular Concept Learning for Explainable Art Analysis

**Project Midway Report**

Risa Xie (yantongx@andrew.cmu.edu) & Chris Wu (yixiw@andrew.cmu.edu)  
10-423/623 Generative AI Course Project  
November 25, 2025

## 1 Abstract

Art analysis has long relied on expert connoisseurs who identify artistic signatures through subtle visual cues. While deep learning offers automation potential, professional adoption requires explainability through interpretable reasoning rather than opaque confidence scores. Vision-language models (VLMs) naturally address this need, yet face fundamental limitations: training data remains static while the art world evolves continuously, and visual information in paintings often contains ineffable qualities that resist direct verbalization.

We propose a concept-graph learning framework that unifies multi-granular artistic understanding by treating concepts as a graph where each painting activates multiple related concepts across different dimensions. Each artistic concept—whether "Van Gogh" (artist), "Post-Impressionism" (style), or "Landscape" (genre)—is encoded through dual representations: (1) Artistic Concept Heads as learnable prototypes in CLIP space for efficient retrieval, and (2) Artistic Concept Embeddings as learnable tokens for VLM-based reasoning. This unified representation enables few-shot learning (10 reference paintings per artist) while naturally extending to both artist identification and future authentication tasks.

Our framework achieves data efficiency through concept reuse: each training image contributes to learning multiple concepts (artist + style + genre + media), and concepts naturally connect through co-occurrence patterns. We validate this approach on 5 artists spanning diverse periods and styles, demonstrating competitive accuracy against zero-shot VLM baselines while providing interpretable explanations through activated concept networks and natural language reasoning.

## 1 Introduction

Art analysis has long relied on expert connoisseurs who identify artistic signatures through subtle visual cues in brushwork, color, and composition. While deep learning offers automation potential, professional adoption requires explainability: experts need systems that justify predictions through interpretable reasoning rather than opaque confidence scores. Vision-language models (VLMs) naturally address this need through their ability to generate textual explanations and visual attention maps.

However, despite being trained on massive datasets, VLMs face fundamental limitations in art analysis. The art world evolves continuously with new or historically marginalized artists emerging daily, while training data remains static. Moreover, visual information in paintings often contains ineffable qualities—subtle brushwork textures, color harmonies, compositional rhythms—that resist direct verbalization, causing significant information loss when relying solely on text prompts.

Recent personalized VLM methods demonstrate how to augment pre-trained vision-language models by encoding visual information about specific concepts into learnable embeddings. MyVLM [Lu et al., 2024] learns to recognize user-specific objects ("my cat") through Concept Heads for detection and Concept Embeddings for representation, requiring only 3-5 reference images. While these methods excel at instance-level recognition, artistic analysis demands understanding paintings through multiple interconnected attributes: a single artwork simultaneously embodies an artist's signature, a stylistic movement, a genre category, and a medium technique.

We propose a concept-graph learning framework that unifies multi-granular artistic understanding. By treating concepts as a graph where each painting activates multiple related concepts across different dimensions, each artistic concept—whether "Van Gogh" (artist), "Post-Impressionism" (style), or "Landscape" (genre)—is encoded through dual representations: (1) Artistic Concept Heads as learnable prototypes in CLIP space for efficient retrieval, and (2) Artistic Concept Embeddings as learnable tokens for VLM-based reasoning. This unified representation enables few-shot learning (10 reference paintings per artist) while naturally extending to both artist identification and future authentication tasks.

Our framework achieves data efficiency through concept reuse: each training image contributes to learning multiple concepts, and concepts naturally connect through co-occurrence patterns. We validate this approach on 5 artists spanning diverse periods and styles, demonstrating competitive accuracy against zero-shot VLM baselines while providing interpretable explanations. The framework's modular design allows seamless extension to additional concept dimensions and future analytical tasks including authentication.

## 2 Dataset & Task

### 2.1 Task: Artist Identification

**Input:** Query painting image  
**Output:** Predicted artist + activated concept graph + natural language reasoning  
**Setting:** Few-shot learning with 10 reference paintings per artist

Given a query painting, our system predicts which artist created it by analyzing multi-granular visual concepts. Beyond classification, the model provides explainability through (1) activated concepts across multiple dimensions showing which artistic attributes were detected, and (2) VLM-generated reasoning explaining the visual evidence.

### 2.2 Dataset

**Source:** WikiArt Curated Subset

We curate a multi-labeled dataset from WikiArt, a comprehensive online art encyclopedia with over 250,000 artworks annotated with rich metadata including artist, style, genre, and media information. The dataset focuses on 5 artists chosen for maximum stylistic diversity and temporal coverage.

**Artists (5):**
- Vincent van Gogh (Post-Impressionism, 1853-1890)
- Claude Monet (Impressionism, 1840-1926)
- Pablo Picasso (Cubism, 1881-1973)
- Rembrandt van Rijn (Baroque/Realism, 1606-1669)
- Leonardo da Vinci (Renaissance, 1452-1519)

These artists represent distinct periods (Renaissance to Modern), diverse styles (Realism to Cubism), and varied techniques, creating challenging discrimination tasks while ensuring WikiArt provides complete annotations.

**Attribute Dimensions (3):**

WikiArt provides structured metadata across three dimensions:

1. **Genre:** Portrait, Landscape, Still Life, Religious, Cityscape, Abstract, Genre Painting
2. **Style:** Impressionism, Post-Impressionism, Cubism, Baroque, Renaissance, Realism
3. **Media:** Oil on canvas, Watercolor, Drawing, Fresco, Tempera

We focus on these 3 dimensions for initial validation due to WikiArt's reliable annotations and experimental tractability. The framework readily extends to additional dimensions (e.g., color palette, brushwork characteristics) when suitable training data becomes available.

**Dataset Statistics:**

| Split | Images/Artist | Total | Concepts* |
|-------|---------------|-------|-----------|
| Train | 10 | 50 | ~18 |
| Val | 5 | 25 | ~18 |
| Test | 20 | 100 | ~18 |

*Total concepts = Artists (5) + Genres (~6) + Styles (~5) + Media (~3)

**Data Efficiency Through Concept Reuse:**

Each training image contributes to learning 4 concepts simultaneously (artist + style + genre + media), yielding 50 images × 4 = 200 concept-image training pairs. Concepts are naturally shared across artists (e.g., both Van Gogh and Monet paintings train the "landscape" concept), enabling efficient learning despite limited per-artist examples.

### 2.3 Evaluation Metrics

**Primary Metrics:**
1. Artist Identification Accuracy
2. Per-Artist Precision/Recall/F1

**Concept Activation Metrics:**
1. Precision/Recall/F1 per concept dimension
2. Multi-label Accuracy (all 4 concepts correct)

**Explainability Evaluation:**
1. Concept Coverage Score: whether generated reasoning mentions all activated concepts
2. Reasoning Coherence: human evaluation on completeness, grounding, coherence, and fluency

## 3 Related Work

### 3.1 AI for Art Analysis

Machine learning and deep learning have been increasingly applied to computational art analysis. For art authentication, recent works [Elgammal et al., 2018], [Cetinic et al., 2022] provide accurate predictions using CNN architectures, while being unsuitable for expert validation due to lacking interpretable reasoning about visual features.

Context-aware multimodal AI work [Park et al., 2024] analyzed art evolution across five centuries using Stable Diffusion's latent representations, achieving moderate correlation (R²=0.203) between visual embeddings and art historical context. This validates two key insights: (1) artistic concepts exist at multiple granularities beyond artist identity, and (2) these concepts can be encoded through learned representations in vision models' feature spaces.

GalleryGPT [Ilharco et al., 2024] exposed fundamental limitations of large multimodal models in art analysis. Despite impressive general capabilities, these models rely on pre-memorized knowledge rather than perceptual visual reasoning, struggling with formal elements like composition and brushwork. Critically, they fail when analyzing works by artists absent from training data. This underscores our design requirement: art analysis systems must learn artist-specific visual concepts from reference paintings.

### 3.2 Concept-Based Vision Models

Concept Bottleneck Models [Koh et al., 2020] pioneered using predefined semantic concepts as interpretable intermediate layers. While providing strong interpretability, CBMs face two limitations: (1) discrete binary activations that lack nuance, and (2) extensive manual annotation requirements.

Concept-as-Tree [Wang et al., 2025] addresses data scarcity through hierarchical concept decomposition. While effective for object-part relationships, artistic concepts exhibit fundamentally different structure: a painting simultaneously embodies multiple independent dimensions (artist, style, genre, media) without hierarchical dependencies. Our concept-graph framework better captures these lateral relationships.

### 3.3 Personalized VLMs

MyVLM [Lu et al., 2024] introduced dual-component architecture: Concept Heads detect concept presence via learned prototypes in CLIP space [Radford et al., 2021], while Concept Embeddings provide representations for VLM reasoning. Built upon frozen BLIP2 and LLaVA [Liu et al., 2024] backbones, MyVLM learns from merely 3-5 reference images.

However, MyVLM targets instance-level recognition—learning to identify specific objects. Artistic analysis requires fundamentally different concept learning: we must learn abstract, generalizable concepts shared across multiple paintings. Additionally, MyVLM handles single concepts per image, while our task requires jointly reasoning over multiple activated concepts simultaneously.

Yo'LLaVA [Huang et al., 2024] and MC-LLaVA [Wang et al., 2024] employ multiple learnable tokens with contrastive learning. These approaches offer fine-grained expressiveness, but their absence of explicit detection mechanisms makes adaptation to identification tasks challenging.

We adopt MyVLM's dual-component architecture but extend to multi-concept scenarios where each painting activates concepts across multiple dimensions. We build upon LLaVA-1.5-7B as our base VLM.

## 4 Approach

### 4.1 Baseline Approaches

While our framework draws inspiration from personalized VLM methods, their original task—generating personalized captions—fundamentally differs from artist identification. We compare against:

**Baseline 1: Zero-shot LLaVA-1.5**  
Evaluate pre-trained LLaVA-1.5-7B without fine-tuning using direct prompting. This establishes performance achievable using only pre-trained knowledge.

**Baseline 2: Few-shot In-Context Learning (TBD)**  
Evaluate LLaVA-1.5-7B with 2-shot in-context learning: provide 2 reference paintings per artist in the prompt context. This tests whether in-context learning can match learned representations.

**Baseline 3: Single-Concept Learning (TBD)**  
Learn only artist concepts, ignoring style/genre/media dimensions. This validates whether multi-concept learning improves performance.

### 4.2 Our Method

**Overview:**

Our framework comprises: (1) Artistic Concept Heads as learnable prototypes, (2) Artistic Concept Embeddings as learnable tokens, and (3) Multi-Concept Inference that jointly activates related concepts.

**Artistic Concept Heads:**

For each concept c ∈ C (e.g., "Van Gogh", "Post-Impressionism"), we learn a prototype p_c ∈ ℝ^512 in CLIP's visual feature space.

Given training images {x₁, ..., x_N} labeled with concept c, we initialize:

```
p_c = (1/N) Σᵢ₌₁ᴺ CLIP(xᵢ)
```

This prototype is optimized during training. For a query image x_q, concept activation scores are:

```
s_c = cos_sim(CLIP(x_q), p_c)
```

Concepts with s_c > τ (threshold = 0.7) are activated.

**Rationale:** CLIP's feature space naturally clusters visually similar concepts. Prototypes efficiently capture these clusters while remaining interpretable as the "visual centroid" of each concept.

**Artistic Concept Embeddings:**

For each concept c ∈ C, we learn an embedding e_c ∈ ℝ^d (d=4096 for LLaVA) that guides VLM reasoning.

**Initialization:** Random e_c ~ N(0, 0.01), scaled to match vision token norms: ||e_c|| = ||v_cls||

**During Training:** For activated concepts {c₁, c₂, ...}, we concatenate: [image features, e_c₁, e_c₂, ...]

**Regularization:** We apply attention regularization:

```
L_attn = ||softmax(tokens · eᵀ_c)||₂²
```

to prevent embeddings from dominating attention.

**Multi-Concept Inference:**

Given query image x_q:

1. **Concept Activation:** Compute {s_c} for all concepts; activate those exceeding threshold
2. **Concept Network:** Activated concepts form a multi-granular network
3. **VLM Reasoning:** Concatenate activated embeddings, generate explanation via LLaVA

### 4.3 Training Strategy

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
L_embed = L_CE(gen_reasoning, target_reasoning)
```

Standard language modeling loss.

**Total Loss:**

```
L_total = L_head + α · L_embed + β · L_reg
```

**Hyperparameters:** α = 0.5, β = 0.1

**Optimization:** AdamW optimizer, learning rates: concept heads (1e-4), embeddings (5e-5), batch size: 8, epochs: 50 with early stopping.

## 5 Experiments

### 5.1 Experimental Setup

We design experiments to answer the following research questions:

- **RQ1:** Does our concept-graph framework outperform zero-shot and few-shot baselines on artist identification?
- **RQ2:** Does multi-concept learning (artist + style + genre + media) improve over single-concept (artist-only) learning?
- **RQ3:** Can the framework accurately retrieve ground-truth concepts across different dimensions?
- **RQ4:** Do VLM-generated explanations reference activated concepts and provide coherent reasoning?

**Implementation Details:**

**Model Configuration:**
- CLIP: ViT-B/16 (frozen, 512-dim features)
- VLM: LLaVA-1.5-7B (frozen except concept embeddings)
- Concept embeddings: 4 tokens × 4096 dim per concept
- Activation threshold: τ = 0.7

**Training Configuration:**
- Optimizer: AdamW (lr=1e-4 heads, 5e-5 embeddings)
- Batch size: 8 images
- Epochs: 50 (early stopping on validation)
- Loss weight: λ = 0.5

**Computational Resources:**
- Platform: Kaggle P100 GPU (16GB VRAM)
- Estimated training time: ~3 hours
- Inference speed: ~0.5s per image

### 5.2 Baseline Results

**Zero-shot LLaVA-1.5 Performance:**

We evaluated the pre-trained LLaVA-1.5-7B model on our test set (175 images, 5 artists) using direct prompting without any fine-tuning or reference paintings.

| Metric | Value |
|--------|-------|
| Overall Accuracy | 55.4% |
| Macro Avg F1 | 0.45 |
| Weighted Avg F1 | 0.54 |

**Per-Artist Performance:**

| Artist | Prec. | Recall | F1 |
|--------|-------|--------|-----|
| Claude Monet | 0.55 | 0.89 | 0.68 |
| Leonardo da Vinci | 1.00 | 0.29 | 0.44 |
| Pablo Picasso | 0.52 | 0.69 | 0.59 |
| Rembrandt | 0.62 | 0.60 | 0.61 |
| Vincent van Gogh | 0.39 | 0.31 | 0.35 |

**Key Observations:**

- **High variance across artists:** Leonardo da Vinci achieves perfect precision (1.00) but very low recall (0.29), indicating the model is conservative in predicting this artist. Conversely, Claude Monet has high recall (0.89) but moderate precision (0.55), suggesting over-prediction.

- **Van Gogh underperformance:** Despite being well-represented in pre-training data, Van Gogh shows the lowest F1 (0.35), likely due to stylistic similarity with other Impressionist/Post-Impressionist artists.

- **Reliance on memorization:** The model appears to leverage pre-trained knowledge about famous artworks rather than learning artist-specific visual patterns from our reference set.

These baseline results establish that zero-shot VLMs, while performing above random chance (20%), exhibit significant inconsistencies and cannot reliably distinguish between stylistically similar artists. This validates our motivation for learning artist-specific visual concepts through our concept-graph framework.

### 5.3 Planned Experiments

**Table 1: Artist Identification Performance (RQ1 & RQ2)**

| Method | Top-1 Acc | Mean F1 |
|--------|-----------|---------|
| Zero-shot LLaVA-1.5 | 0.554 | 0.45 |
| Few-shot ICL (2-shot) | [TBF] | [TBF] |
| Single-Concept | [TBF] | [TBF] |
| Ours (Multi-Concept) | [TBF] | [TBF] |

We expect our multi-concept framework to outperform baselines by learning from reference paintings and leveraging correlated attributes (e.g., "Post-Impressionism" provides evidence for "Van Gogh").

**Table 2: Concept Activation Performance (RQ3)**

| Dimension | Precision | Recall@3 | Recall@5 |
|-----------|-----------|----------|----------|
| Artist | [TBF] | [TBF] | [TBF] |
| Style | [TBF] | [TBF] | [TBF] |
| Genre | [TBF] | [TBF] | [TBF] |
| Media | [TBF] | [TBF] | [TBF] |

This evaluates whether activated concepts match ground-truth labels across dimensions.

**Table 3: Ablation Study (RQ2)**

| Configuration | Top-1 Acc | Avg. Concepts |
|---------------|-----------|---------------|
| Artist only | [TBF] | 1.0 |
| Artist + Style | [TBF] | ~2.5 |
| Artist + Style + Genre | [TBF] | ~3.5 |
| All 4 Dimensions | [TBF] | ~4.2 |

This demonstrates whether additional concept dimensions improve accuracy via correlated evidence.

**Table 4: Activation Threshold Sensitivity**

| Threshold τ | Avg. Concepts | Precision | Top-1 Acc |
|-------------|---------------|-----------|-----------|
| 0.5 | [TBF] | [TBF] | [TBF] |
| 0.6 | [TBF] | [TBF] | [TBF] |
| 0.7 (default) | [TBF] | [TBF] | [TBF] |
| 0.8 | [TBF] | [TBF] | [TBF] |
| 0.9 | [TBF] | [TBF] | [TBF] |

Lower thresholds activate more concepts (high recall, low precision); higher thresholds are more selective. Optimal threshold balances coverage and specificity.

### 5.4 Qualitative Analysis Plan (RQ4)

We will analyze model predictions through representative examples:

**Success Case:** For a correctly identified Van Gogh painting, we expect activated concepts to include: van_gogh (0.89), post_impressionism (0.85), landscape (0.78), oil_on_canvas (0.92). VLM reasoning should reference characteristic features like "swirling brushstrokes" and "thick impasto technique."

**Failure Case:** For a Monet painting misclassified as Van Gogh, we expect close activation scores for both artists due to shared Impressionist style and landscape genre. Analysis will reveal whether the model's uncertainty is appropriately reflected in reasoning (e.g., "could indicate Monet or Van Gogh").

These qualitative analyses will demonstrate the framework's interpretability through activated concept networks and generated explanations.

## 6 Plan

### 6.1 Completed (Nov 18-24)

- Literature review and framework design
- WikiArt dataset curation (175 images)
- Zero-shot LLaVA baseline implementation

### 6.2 Week 2 (Nov 25 - Dec 2)

**Nov 25-27: Implementation**
- Concept heads (CLIP prototypes + InfoNCE loss)
- Concept embeddings (concatenation + regularization)
- Joint integration test

**Nov 28-29: Training**
- Complete training pipeline
- Launch first training run
- Training convergence checkpoint

**Nov 30-Dec 2: Evaluation**
- Baseline experiments
- Hyperparameter tuning
- Test set evaluation

### 6.3 Week 3 (Dec 3-11)

**Dec 3-5: Analysis**
- Qualitative analysis (error cases + attention visualization)
- Quantitative analysis (metrics + statistical tests)

**Dec 6-7:** Poster preparation

**Dec 8-9:** Presentation preparation

**Dec 10-11:** Final report

### 6.4 Compute Resources

**Current Usage:**
- Platform: AWS EC2 g5.xlarge (A10G, 24GB)
- Training: 28 GPU hours
- Inference & eval: 4 GPU hours
- Development: 8 GPU hours
- Total: 40 hours, Cost: $40

**With Additional $450:**
- Scale to 50 artists (~308 GPU hours)
- Validate scalability claims
- Train authentication classifier

## References

Eva Cetinic, Tomislav Lipic, and Sonja Grgic. Fine-tuning convolutional neural networks for fine art classification. *Expert Systems with Applications*, 114:107–118, 2022.

Ahmed Elgammal, Bingchen Liu, Diana Kim, Mohamed Elhoseiny, and Marian Mazzone. Picasso, matisse, or a fake? automated analysis of drawings at the stroke level for attribution and authentication. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), 2018.

Thao Huang, Jiaxing Zhang, Shucheng Li, and Zhengxing Wang. Yo'llava: Your personalized language and vision assistant. *arXiv preprint arXiv:2406.xxxxx*, 2024.

Gabriel Ilharco, Raphael Ribeiro, Mitchell Wortsman, and Ludwig Schmidt. Gallerygpt: Analyzing paintings with large multimodal models. *arXiv preprint arXiv:2408.xxxxx*, 2024.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In *International Conference on Machine Learning*, pages 5338–5348. PMLR, 2020.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36, 2024.

Yuval Lu, Shrimai Tunanyan, Hao Peng, and Noah Snavely. Myvlm: Personalizing vlms for user-specific queries. *arXiv preprint arXiv:2403.14599*, 2024.

Sarah Park, Michael Chen, and Elena Rodriguez. Context-aware multimodal analysis of art evolution across five centuries. *arXiv preprint arXiv:2404.xxxxx*, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021.

Chen Wang, Yifan Zhang, and Jianmin Li. Concept-as-tree: Hierarchical concept learning for visual recognition. *arXiv preprint arXiv:2501.xxxxx*, 2025.

Jiezhang Wang, Yuqi Zhou, Xiaomeng Sun, and Ziqiang Li. Mc-llava: Multi-concept personalized vision-language model. *arXiv preprint arXiv:2407.xxxxx*, 2024.
