# Graph-Conditioned Image Diffusion via Temporal GNN Adapters

Source code for the CS598 final project: graph-conditioned image
generation with SDXL, using Action Genome's spatio-temporal scene
graphs as conditioning.

## Final pipeline (SDXL)

The submitted system uses these files end-to-end:

| File | Purpose |
|---|---|
| `src/ag_graph_dataset.py` | Build per-video PyG `Data` objects from raw AG annotations (`person_bbox.pkl`, `object_bbox_and_relationship.pkl`). Outputs `dataset/ag/graphs/<video_id>.pt`. |
| `src/dataset_image.py` | `MixedImageDataset` — 70/30 mix of Charades text-only frames and AG graph-conditioned frames, with deterministic train/val carve. Includes `build_prompt_from_graph` (object-aware prompt builder). |
| `src/tgnn_model.py` | `TGNNEncoder` — 3-layer GATv2 over spatial+temporal edges, with target-frame marker and CLIP-initialised class embedding. |
| `src/adapter_sdxl.py` | `SDXLGraphAdapter` and `GraphAttnProcessor` — IP-Adapter–style parallel cross-attention injected into every cross-attn layer of the SDXL UNet, with K/V projectors shared per resolution. |
| `src/train_sdxl.py` | Training loop, validation, early stopping, checkpoint saving. |
| `src/infer_sdxl.py` | Inference: generates `<vid>_graph_cond.png` and `<vid>_baseline.png` pairs using the SAME prompt for fair comparison. |
| `src/eval_sdxl_recall.py` | Object Recall evaluation via Grounding DINO open-vocabulary detection. |
| `src/test_sdxl_dryrun.py` | Shape-checks every step of the pipeline on a single batch (no backward). |

The original Delta SLURM wrappers (`run_*.sh`) are not committed because they hard-code our HPC paths, account names, and partitions. The Python recipe below is portable and should be wrapped in whatever batch system your cluster uses.

## Reproducing the main result

```bash
# 0) Environment (PyTorch + diffusers + PyG + transformers + imageio[pyav])
#    A single A100 40GB or H200 80GB is enough for everything below.

# 1) Pre-build PyG graphs from raw Action Genome annotations (one-shot, ~1 min)
python src/ag_graph_dataset.py \
    --ann_dir dataset/ag/annotations/action_genome_v1.0 \
    --out_dir dataset/ag/graphs

# 2) Shape sanity check on a single batch (~3 min once SDXL is cached)
python src/test_sdxl_dryrun.py

# 3) Train: ~3-4h on A100, early-stops via patience=5 on val MSE
python src/train_sdxl.py \
    --out_dir checkpoints_sdxl --batch_size 1 --graph_prob 0.30

# 4) Generate 20 (graph_cond, baseline) image pairs for the test split
python src/infer_sdxl.py --checkpoint checkpoints_sdxl/best --n 20

# 5) Object Recall via Grounding DINO open-vocabulary detection
python src/eval_sdxl_recall.py \
    --img_dir outputs/sdxl --out outputs/sdxl_recall.json
```

## Headline results (n=20 AG test graphs, GroundingDINO threshold 0.3)

| Variant | Graph-cond | Baseline | Δ |
|---|---:|---:|---:|
| Generic prompt, double zero-init bug (broken) | 0.4883 | 0.4883 | 0.0000 |
| Generic prompt, fixed zero-init | 0.5188 | 0.4883 | +0.0305 |
| Object-aware prompt, fixed zero-init | 0.8215 | 0.8468 | −0.0252 |

Row 2 is the cleanest evidence the adapter learned a non-trivial graph-to-image mapping. Row 3 shows that with a rich prompt, baseline saturates and Object Recall ceases to be a discriminative metric — see report for full discussion.

Final report: `docs/final_report/reportlatex/main.tex` (compiles cleanly on Overleaf with default TeX Live; figures are pre-rendered PNGs in `docs/final_report/reportlatex/figures/`).

## Codebase notes / dead branches

The repo also contains exploratory code for three earlier video-diffusion backbones we tried before settling on SDXL. They are kept for transparency; **none of them are used in the final pipeline**:

- `train_hunyuan.py`, `infer_hunyuan*.py`, `adapter_hunyuan.py`, `test_hunyuan_dryrun.py` — HunyuanVideo MMDiT (13B) attempt; trained but adapter signal was swamped by the frozen backbone (only 5%×5K=~250 graph forward passes vs. 13B params).
- `train_wan.py`, `infer_wan*.py`, `adapter_wan*.py`, `test_wan_dryrun.py` — Wan 2.1 video diffusion (1.3B/14B); produced colour-spot outputs.
- `adapter.py`, `train.py`, `infer_animatediff.py`, `infer_graph.py` — earliest CogVideoX-2b attempts at 256×256 (the resolution turned out to be the killer).
- `dataset_mixed.py` — the 95/5 video version of the mixed dataset; superseded by `dataset_image.py`.

The exploration history section of the report (Section "Exploration history and ongoing work") describes what each attempt taught us and why we ended up at the SDXL design.

## Third-party code & licences

- HuggingFace `diffusers` (Apache-2.0): SDXL pipeline and `Attention` / `AttnProcessor` API.
- HuggingFace `transformers` (Apache-2.0): CLIP text encoders for class-embedding init; Grounding DINO for evaluation.
- PyTorch Geometric (MIT): `GATv2Conv`, `AttentionalAggregation`, `Batch`.
- Action Genome annotations and Charades-v1 videos: academic use as released by their respective authors.

No third-party model weights are redistributed in this repository; all backbone weights are downloaded from the HuggingFace hub at runtime.
