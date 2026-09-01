# Tensor Networks for Classical and Quantum Machine Learning

This repository holds the code for my MSci Theoretical Physics dissertation at UCL
(Department of Physics and Astronomy). The project takes an existing tensor-network
image classifier apart to see what it has actually learned, and then works out the
algebra needed to rebuild it more compactly.

| | |
|---|---|
| **Module** | PHAS0097 — Physics & Astrophysics Project (45 credits) |
| **Year** | 2025/26 (final year) |
| **Supervisor** | Prof. Andrew G. Green |
| **Mark** | 74 / 100 (first class) |

## Abstract

> The deterministic tensor network classifier of Wright et al. achieves modest
> classification accuracy on MNIST. This falls below that of standard machine learning
> models such as convolutional neural networks, which achieve above 99% accuracy on the
> same dataset. This report investigates whether the classifier can offer transparency
> into its decision-making process, and whether understanding its internal structure can
> lead to a more efficient construction.
>
> This classifier is built deterministically using Quantics data encoding. This allows
> the classifier to organise hierarchically, from global features to pixel-level details.
> Feature extraction reveals that fine-scale 4 × 4 geometric primitives (motifs) show high
> cross-class overlap, exceeding 0.99 for the majority of classes. Replacing the fine-scale
> tensors of all ten classes with a single shared motif drops accuracy from 80.9% to
> 80.4%, confirming a redundancy in the stored data.
>
> The study also sets up an algebraic framework for manipulating Quantics encoded motifs.
> It is shown that rotating and scaling a motif are cheap operations that do not increase
> the complexity of the representation. Shifting a motif by a single pixel is far more
> complex. For this manipulation an MPO with bond dimension two is required, regardless of
> grid size. Furthermore, when multiple instances of the same motif appear at different
> positions, the identical parts of their representations can be stored only once. This
> achieves lossless compression.
>
> Together, these two results point toward a more efficient and interpretable classifier,
> where a single shared motif and a set of geometric placement rules are enough to
> reconstruct the full classifier.

## Navigating the repository

There are two notebooks, one per half of the project — **Part 1**, the feature
extraction and compressibility study, and **Part 2**, the algebraic framework for
manipulating Quantics-encoded motifs — each with its own helper module. Suggested
reading order is `classifier.ipynb` first, then
`motif_encoding/motif_encoding_main.ipynb`.

### `classifier.ipynb` — Part 1

Runs the whole first half of the project, top to bottom:

1. **Building a class prototype.** Encodes a batch of training images as MPS and sums them
   into one MPO per class, with the block construction spelled out site by site.
2. **Compressing it ("Train MPO").** Sweeps back and forth truncating the MPO to a target
   bond dimension. The markdown cells record the accuracy of each saved model (for
   example ~83% at bond dimension 256, ~73% at bond dimension 10).
3. **Loading a saved model and testing it** against a held-out cluster of 10 classes ×
   100 digits.
4. **Extracting masks.** Turns the lowest-level tensors back into a per-class image and
   compares it, via `compute_overlap_for_mps`, to the plain pixel-average of that class.
5. **Weighted motifs.** A right-to-left sweep (`plot_top_motifs`) pulls out the fine-scale
   4 × 4 shapes that contribute most to each class operator.
6. **Motif overlap.** `plot_motif_overlaps` computes the inner product between motif *i* of
   class *c* and motif *j* of class *c'*, producing the cross-class overlap matrices that
   show the fine scale is shared.
7. **Modifying and truncating the classifier.** Replaces correlated motifs across classes
   with a shared one, compares maximum bond dimensions 20/30/40, and finally
   (`evaluate_config`) runs a brute-force grid search over per-site bond-dimension
   configurations to find how far the operator can be squeezed before accuracy falls. The
   last section contrasts truncating the MPS with truncating the MPO.

### `my_functions.py` — helpers for Part 1

Grouped by what they do:

- **Encoding and MPS construction:** `image_to_hierarchical` (Quantics layout),
  `dense_to_mps_L_canonical` and `split_L_canonical` (dense array → left-canonical MPS),
  `batch_mps_cluster`, plus small utilities `celg2` and `n_hadamard`.
- **Classification:** `classify_state` (the image–operator contraction),
  `evaluate_accuracy`, `sweep_single_class`, `eval_class_vs_ensemble`.
- **Reading tensors back as images:** `generalized_reconstruction`, `left_to_right_sweep`.
- **Ensembles:** `construct_ensemble`, `plot_ensemble_performance`.
- **Compression:** `truncate_mpo_via_gauge`, `count_params`, `eval_config_full_mpo`,
  `grid_search_mpo_config`.

### `motif_encoding/motif_encoding_main.ipynb` — Part 2

Split into two investigations plus a translation section:

- **Investigation 1 – rotation operators.** Applies rotation operators to an image-encoded
  MPS and shows how the effect depends on which site they act on (on a "positioning" site
  a rotation behaves as a transposition; on a shape site it rotates the shape). Covers
  90° and 180° rotations, and rotations on larger grids and multiple sites at once.
- **Investigation 2 – building ensembles by superposition.** Places several shapes on one
  grid by adding their MPS, then compresses the sum by merging the branches they share.
  Worked examples include a fractal (used as a figure in the dissertation), two shapes at
  different scales or orientations, and the three-shape case where the simplified version
  demonstrates the lossless suffix compression.
- **Translation.** Half-cell and single-pixel shifts of a small shape, including the case
  where a shape straddles a quadrant boundary and has to be encoded as several pieces.

### `motif_encoding/help_functions_motifs.py` — helpers for Part 2

`create_grid` (place a shape in a zero-padded 2ⁿ grid), `plot_save_grid`,
`create_superposition_mps` (add MPS together), `plot_motif_reconstruction`.
`help_functions_motifs.py` in the repository root is an identical copy so that
`classifier.ipynb` can import the same helpers.

### Data and figure files

- `mnist_tensors_*.npz` — pre-computed class MPO tensors at several bond dimensions and
  qubit counts, so the notebooks can load a classifier instead of rebuilding it.
- `mnist_clusters/*.npz` — MNIST train/test images already encoded and batched as MPS
  clusters.
- `mpo_baseline_stats_dim_*.npz` — the plain pixel-average baseline for each class.
- `plots/` and `motif_encoding/plots/` — figures reproduced in the dissertation.

## Attribution

- The classifier construction follows **Wright et al., arXiv:2205.09768**.
- Parts of `my_functions.py` are adapted from code by **Ivan Shalashilin** (UCL), who also
  helped me understand the classifier codebase, as acknowledged in the dissertation.
- MNIST is the standard handwritten-digit dataset of LeCun et al.
