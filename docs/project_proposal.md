**Graph-Conditioned Video Diffusion: Temporal GNN Adapters for
Structured Video Generation**

*CS598 Project Proposal*

Jiatong • University of Illinois Urbana-Champaign

**Team Members and Expected Contributions**

**Jiatong** (100%) --- Sole contributor.

-   Literature review and related work synthesis

-   Dataset preparation: Action Genome preprocessing and dynamic graph
    construction

-   Model design: Temporal GNN module architecture and cross-attention
    adapter

-   Training and evaluation on Action Genome benchmark

-   Writing and analysis

**Motivation and Problem Definition**

Contemporary video diffusion models such as Stable Video Diffusion (SVD)
and CogVideoX achieve temporal coherence primarily through factorized
spatial-temporal attention: each spatial position independently attends
over all frames in the temporal dimension. While effective, this
mechanism treats the inter-frame relationship as an implicitly learned,
densely-connected structure with no explicit encoding of semantic
content, object interactions, or causal dynamics between entities across
time.

We observe a fundamental asymmetry: natural language conditioning (via
T5 or CLIP encoders) provides rich semantic intent, while the model\'s
temporal module remains a generic, structure-free attention mechanism.
This leads to a well-documented failure mode in compositional video
generation where multi-object spatial relationships (e.g., \"the cat
pushes the cup off the table\") are rendered inconsistently or
incorrectly, because text encoders do not natively represent relational
graph structure.

We propose to address this gap by introducing a Temporal Graph Neural
Network (T-GNN) conditioning adapter that explicitly models the
evolution of entity relationships across video frames as a dynamic scene
graph, and injects the resulting structured embeddings into a
pre-trained video diffusion model via cross-attention. Concretely, the
research questions are:

-   Can dynamic scene graph embeddings produced by a T-GNN provide
    conditioning signals that measurably improve temporal consistency in
    video diffusion models?

-   Does explicit graph-structured conditioning improve compositional
    fidelity --- the accuracy with which generated videos reflect
    multi-object relational descriptions --- compared to text-only
    conditioning?

-   What graph construction strategy (optical-flow-weighted edges,
    semantic similarity edges, or annotation-derived edges) best
    supports video generation quality?

**Related Work**

***Video Diffusion Models***

Ho et al. \[1\] introduced the first diffusion model for video
generation, extending 2D U-Net architectures with factorized space-time
attention. Subsequent work including Stable Video Diffusion \[2\] and
AnimateDiff \[3\] established the standard paradigm of training
lightweight temporal modules atop frozen image diffusion backbones. Sora
and CogVideoX \[4\] replace factorized attention with full 3D attention
over spacetime patches via Diffusion Transformers (DiT), trading
computational cost for tighter spatial-temporal coupling. None of these
works incorporate explicit graph-structured representations of
inter-frame entity relationships.

***Scene Graph-Conditioned Generation***

The idea of conditioning generative models on scene graphs was
established for images by Johnson et al. \[5\], who used graph
convolutional networks to produce layout maps from scene graphs as
intermediate representations. SGDiff \[6\] replaced layout prediction
with direct masked contrastive pre-training to align scene graph and
image embeddings for latent diffusion models. SceneGenie \[7\] extended
this to guidance during diffusion sampling using bounding box
constraints derived from scene graphs. MOVAI \[8\] is the closest work
to our proposal: it decomposes input text into a hierarchical scene
graph G = (O, R, A) with temporal annotations and injects graph
embeddings into a video diffusion model. However, MOVAI uses static
graph embeddings with hand-crafted temporal annotations and does not
employ a GNN to propagate information across the temporal evolution of
the graph. LAION-SG \[9\] provides a large-scale dataset of image-scene
graph pairs, demonstrating that GNN-encoded structural conditioning
improves compositional generation quality.

***Temporal Graph Neural Networks***

Spatial-temporal GNNs (ST-GCN \[10\], DCRNN \[11\]) have been
extensively studied for action recognition and traffic forecasting,
demonstrating that explicitly modeling temporal graph dynamics
outperforms purely sequential approaches. EvolveGCN \[12\] treats the
GCN weight matrices themselves as evolving through an RNN, capturing
topological changes in dynamic graphs. Action Genome \[13\], the dataset
we employ, was constructed precisely to provide spatio-temporal scene
graph annotations for video understanding. Prior work has applied GNNs
to Action Genome for action recognition \[13\] and video scene graph
generation \[14\], but not for conditioning generative video models.

***Gap and Novelty***

No existing work combines a temporal GNN operating on dynamic scene
graphs with a video diffusion conditioning adapter in an end-to-end
trainable framework. Our work occupies this intersection: we train a
T-GNN to encode the temporal evolution of scene graph structure from
Action Genome annotations and inject it as a cross-attention
conditioning signal into a frozen video diffusion backbone, analogous to
ControlNet \[15\] for image generation.

**Project Expectations**

***Proposed Architecture***

The system consists of three components. First, a dynamic scene graph is
constructed per video clip from Action Genome frame-level annotations:
object instances form nodes, and the 25 relationship categories form
typed edges. A temporal edge connects each node to its counterpart in
adjacent annotated frames, yielding a multi-layer temporal graph.
Second, a Temporal GNN module --- concretely, a Gated Graph Sequence
Neural Network or a temporal variant of GAT --- propagates information
across both spatial edges (within a frame) and temporal edges (across
frames), producing node-level and graph-level embeddings that encode
relational dynamics. Third, a lightweight cross-attention adapter layer
(following the ControlNet paradigm) is inserted into the temporal
attention blocks of a frozen AnimateDiff or SVD backbone, injecting the
T-GNN embeddings as additional key-value pairs.

***Training Strategy***

We adopt a two-stage training strategy. In Stage 1, the T-GNN encoder is
pre-trained on Visual Genome and LAION-SG image-scene graph pairs to
learn generalizable graph-to-embedding mappings before any video data is
introduced. In Stage 2, the cross-attention adapter is trained
end-to-end with the T-GNN on Action Genome (approximately 9,000 training
videos) using a standard diffusion denoising objective, with the video
diffusion backbone kept frozen.

***Evaluation***

We will evaluate along three dimensions:

-   Temporal consistency: Fréchet Video Distance (FVD) and CLIP frame
    consistency score on Action Genome held-out test split.

-   Compositional fidelity: percentage of generated videos correctly
    depicting the subject-predicate-object triplets specified in the
    input scene graph, assessed by an automated scene graph generation
    model applied to generated frames.

-   Ablation: text-only conditioning baseline (frozen backbone, no
    T-GNN), static GNN conditioning (no temporal edges), and full T-GNN
    conditioning, to isolate the contribution of temporal graph
    propagation.

***Expected Outcomes***

We expect the T-GNN conditioning adapter to improve FVD and
compositional fidelity scores over the text-only baseline on the Action
Genome test set. We also expect the ablation to demonstrate that
temporal graph propagation contributes meaningfully beyond static scene
graph conditioning, validating the core hypothesis that explicitly
modeling relational dynamics provides information complementary to both
text conditioning and standard temporal attention.

***Potential Risks***

-   Action Genome's 10K video scale may be insufficient for robust
    adapter training; mitigation involves aggressive data augmentation
    and staged training from a strong pre-trained backbone.

-   Annotation coverage in Action Genome is sparse (approximately 5
    frames sampled per action interval); we will evaluate interpolation
    strategies for denser temporal graph construction.

-   Evaluation of compositional fidelity via automated scene graph
    generation on synthesized frames introduces measurement noise; we
    will supplement with a small human evaluation.

**References**

> \[1\] Ho, J., et al. \"Video Diffusion Models.\" NeurIPS 2022.
>
> \[2\] Blattmann, A., et al. \"Stable Video Diffusion: Scaling Latent
> Video Diffusion Models to Large Datasets.\" arXiv 2023.
>
> \[3\] Guo, Y., et al. \"AnimateDiff: Animate Your Personalized
> Text-to-Image Diffusion Models without Specific Tuning.\" ICLR 2024.
>
> \[4\] Yang, Z., et al. \"CogVideoX: Text-to-Video Diffusion Models
> with An Expert Transformer.\" arXiv 2024.
>
> \[5\] Johnson, J., et al. \"Image Generation from Scene Graphs.\" CVPR
> 2018.
>
> \[6\] Yang, L., et al. \"Diffusion-Based Scene Graph to Image
> Generation with Masked Contrastive Pre-Training.\" arXiv 2022.
>
> \[7\] Farshad, A., et al. \"SceneGenie: Scene Graph Guided Diffusion
> Models for Image Synthesis.\" ICCVW 2023.
>
> \[8\] Anonymous. \"AI Powered High Quality Text to Video Generation
> with Enhanced Temporal Consistency (MOVAI).\" arXiv 2511.00107, 2024.
>
> \[9\] Li, Z., et al. \"LAION-SG: An Enhanced Large-Scale Dataset for
> Training Complex Image-Text Models with Structural Annotations.\"
> arXiv 2412.08580, 2024.
>
> \[10\] Yan, S., et al. \"Spatial Temporal Graph Convolutional Networks
> for Skeleton-Based Action Recognition.\" AAAI 2018.
>
> \[11\] Li, Y., et al. \"Diffusion Convolutional Recurrent Neural
> Network: Data-Driven Traffic Forecasting.\" ICLR 2018.
>
> \[12\] Pareja, A., et al. \"EvolveGCN: Evolving Graph Convolutional
> Networks for Dynamic Graphs.\" AAAI 2020.
>
> \[13\] Ji, J., et al. \"Action Genome: Actions as Composition of
> Spatio-Temporal Scene Graphs.\" CVPR 2020.
>
> \[14\] Cong, Y., et al. \"STGAT: Modeling Spatial-Temporal
> Interactions for Human Trajectory Prediction.\" ICCV 2019.
>
> \[15\] Zhang, L., et al. \"Adding Conditional Control to Text-to-Image
> Diffusion Models (ControlNet).\" ICCV 2023.