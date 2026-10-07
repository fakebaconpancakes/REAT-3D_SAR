# REAT-3D-SAR

REAT-3D-SAR is a skeleton-based human-action-recognition project built around
the NTU-RGB+D datasets. It combines:

- skeleton preprocessing and deterministic train/validation/test preparation;
- physics-inspired feature engineering;
- a spatial graph convolutional network (GCN);
- a temporal Transformer-style aggregation layer;
- four independently trained representation pipelines;
- late-fusion ensemble evaluation;
- differential XAI heatmaps for important joints and frames;
- fidelity and semantic explanation validation; and
- an optional browser dashboard for inspecting generated results.

This document is the complete operating guide. Follow the sections in order
for a new experiment. All commands are intended to be run from the repository
root.

---

## 1. How the project fits together

The complete workflow is:

```text
Raw NTU skeleton files
        |
        v
01_data_splitter.py       Create train/validation splits
        |
        v
02_val_test_split.py      Move part of validation into test
        |
        v
03_convert_to_binary.py   Convert .skeleton files to .pt tensors
        |
        v
train60.py / train120.py  Train one or more representation pipelines
        |
        v
saved_weights/<dataset>/<run-id>/<pipeline>/
        |
        +--> evaluation/evaluate.py
        +--> evaluation/ablation_evaluator.py
        +--> evaluation/tune_ensemble.py
        |
        v
xai_validation/xai_explainer.py
        |
        +--> proving_engine_fidelity.py
        +--> proving_engine_semantic.py
        |
        v
results/<dataset>/<run-id>/<ensemble>/
        |
        v
tools/publish_results.py   Optional: prepare files for the web dashboard
        |
        v
web_app/
```

The scripts do not share a single command-line configuration. The dataset name,
run ID, and ensemble must be kept consistent manually in the relevant files.
The most important rule is:

> A model checkpoint, evaluation script, XAI script, and result directory must
> all refer to the same dataset and run ID.

---

## 2. Model architecture

REAT-3D-SAR is a two-stage spatio-temporal classifier. It first learns
relationships between joints inside each frame, then learns how those
frame-level representations evolve across time and across the two possible
people in a sequence.

```text
Input skeleton sequence
(B, M=2, T=100, V=25, C=9)
              |
              v
Feature engineering
root centering + height normalization
joint coordinates + bone vectors + velocity
              |
              v
Pipeline channel selection
JBV / joints / bones / velocity
              |
              v
Reshape bodies into independent samples
(B*M, T, V, C_pipeline)
              |
              v
Spatial GCN
biological skeleton adjacency
+ learnable edge scaling
              |
              v
Append one global node per frame
(B*M, T, 26, 128)
              |
              v
Temporal_Brain_Layer
  1. masked spatial Flex Attention per frame
  2. global-node extraction
  3. body and time embeddings
  4. video token + local temporal attention
  5. ghost-body masking
              |
              v
Video representation
(B, 128)
              |
              v
Linear classifier
(B, number_of_classes)
              |
              v
Class logits, probabilities, prediction
```

### 2.1 Input representation and feature engineering

The binary dataset stores raw Kinect coordinates with shape
`(T, 2, 25, 3)`: time, body slot, joint, and `(x, y, z)` coordinate. The
dataset loader standardizes every sample to 100 frames by zero-padding shorter
sequences and truncating longer sequences.

`utils\dataset.py` then constructs a canonical nine-channel tensor:

```text
Joints   = root-centered, height-normalized coordinates       [0:3]
Bones    = joint[v] - joint[parent[v]]                        [3:6]
Velocity = joint[t+1] - joint[t]                              [6:9]
JBV      = concatenate(Joints, Bones, Velocity)                [0:9]
```

The transformations are applied in this order:

1. Joint 0 (SpineBase) is subtracted from every joint so the skeleton is
   expressed relative to its own root rather than the room or camera.
2. Coordinates are divided by the SpineBase-to-SpineShoulder distance, with a
   small clamp to avoid division by zero. This reduces sensitivity to subject
   scale.
3. Bone vectors are computed from the Kinect kinematic parent tree.
4. Velocity is computed as the frame-to-frame difference. The final padded or
   initial frame has zero velocity.
5. The tensor is permuted to `(M, T, V, C)` before it is returned.

The loader always returns all nine channels. The selected training or
evaluation stream is chosen later by `select_pipeline_input()` so that every
pipeline uses exactly the same preprocessing.

### 2.2 The four representation pipelines

Each pipeline trains an independent GCN, temporal brain, and classifier:

| Pipeline | Channels | Meaning | Checkpoint directory |
|---|---:|---|---|
| `jbv` | 9 | Joint coordinates + bones + velocity | `jbv` |
| `joints` | 3 | Root-centered joint coordinates | `pure_joints` |
| `bones` | 3 | Relative kinematic bone vectors | `bones` |
| `velocity` | 3 | Motion between consecutive frames | `pure_velocity` |

For a 9-channel batch with shape `(B, M, T, V, 9)`, selecting one stream
produces `(B, M, T, V, 3)` for a three-channel stream or preserves all nine
channels for JBV. The model then folds the body dimension into the batch:

```text
(B, M, T, V, C) -> (B*M, T, V, C)
```

This lets the spatial encoder process each body independently while the
temporal brain later reconnects body streams at the video level.

### 2.3 Spatial graph convolution

`models\spatial_gcn.py` creates a 25-joint biological adjacency matrix from the
Kinect skeleton connections. Self-loops are added and the matrix is
symmetrically normalized:

```text
A_hat = D^(-1/2) (A + I) D^(-1/2)
```

For each frame, the GCN first projects the selected input channels into 128
features. It then performs graph message passing with:

```text
scaled_adjacency = A_hat * edge_scale
out = scaled_adjacency message passing over projected joints
out = LayerNorm(out + projected_input)
out = ReLU + dropout
```

`edge_scale` is learnable. The fixed biological graph supplies the initial
human structure, while the learned scaling lets the model strengthen or
weaken individual joint-to-joint relationships. The output has shape
`(B*M, T, 25, 128)`.

### 2.4 Temporal brain and global nodes

`models\temporal_brain.py` adds a trainable global node to every frame. This
26th token summarizes the frame after spatial processing. The temporal brain
has two distinct attention stages:

1. **Spatial attention within each frame.** The 25 physical joints and the
   global node interact under an anatomical block mask.
2. **Temporal attention across frames and bodies.** The extracted global node
   for every body and frame is combined into a sequence and summarized by a
   trainable video token.

Body embeddings identify body slot 0 versus body slot 1. Learned temporal
position embeddings identify the frame index. The video token is placed at the
front of the temporal sequence and becomes the final `(B, 128)` video
representation after the temporal block.

### 2.5 Flex Attention: what it does and why it reduces attention work

The spatial stage uses PyTorch's `flex_attention` with a compiled block mask.
The 25 joints are assigned to anatomical rooms, and the mask allows:

```text
joint -> joint       only when both joints share an anatomical room
joint -> global      always
global -> joint      always
global -> global     always
```

The global node therefore acts as the cross-room communication route. A hand
does not directly attend to every unrelated joint in the same frame, but it
can communicate with the global node and with joints in its own room.

Dense self-attention calculates an interaction for every query/key pair,
which has quadratic cost in the token count (`O(S^2)`). The Flex Attention
block mask expresses the permitted pattern directly instead of constructing a
full unconstrained attention pattern. This reduces the number of meaningful
joint interactions from an all-to-all pattern to local anatomical interactions
plus global-node connections, lowering attention memory and potentially
reducing compute when the compiled backend exploits the block sparsity.

The qualification “potentially” matters: the mask guarantees restricted
attention semantics, but the actual wall-clock improvement depends on the
PyTorch version, compiler, GPU, and kernel backend. The implementation
compiles `flex_attention` once and caches the 26-by-26 block mask after the
first forward pass, avoiding repeated mask construction.

The temporal stage is separate: it currently uses
`torch.nn.MultiheadAttention` with an additive sliding-window mask of
`window_size = 5`. A temporal token can attend locally within five frames,
while the video token can attend to the whole sequence. This limits noisy
long-range frame-to-frame interactions, but it is not the same implementation
as the spatial Flex Attention stage.

### 2.6 Handling one or two people

Every sample has two body slots. When a slot contains no meaningful motion,
the dataset creates a `body_mask` entry for that slot. The temporal brain
expands this mask across all frames and passes it as
`key_padding_mask` to temporal attention. The video token remains unmasked, so
the model can summarize the real body without treating zero padding as a
second person.

---

## 3. Training procedure

`train60.py` and `train120.py` share the same training design; they differ in
the dataset entry point and class count. A run trains one pipeline at a time.
The `--pipeline` argument selects the representation:

```powershell
python train60.py --pipeline jbv
python train60.py --pipeline joints
python train60.py --pipeline bones
python train60.py --pipeline velocity
```

For every batch, training performs:

```text
load engineered tensor and body mask
        |
select pipeline channels
        |
run Spatial_GCN_Layer
        |
append frame-level global node
        |
run Temporal_Brain_Layer
        |
run 128 -> class-count linear classifier
        |
CrossEntropyLoss against the action label
        |
AdamW backpropagation and update
```

The default configuration is 100 epochs, batch size 64, learning rate
`0.001`, and weight decay `1e-4`. The optimizer is AdamW. The learning-rate
schedule uses 10 epochs of linear warmup followed by cosine decay to
`1e-6`. Warmup prevents the randomly initialized attention and classifier
layers from receiving the full learning rate immediately; cosine decay then
reduces the step size as the model converges.

After each training epoch, the model is evaluated on the validation split.
The best validation-accuracy checkpoint is saved as:

```text
best_gcn.pth
best_transformer.pth
best_classifier.pth
metadata.json
```

Every ten epochs, an additional numbered checkpoint is written. Training and
validation metrics are logged to Weights & Biases when W&B is enabled. The
test split is not used by the training loop.

---

## 4. Late-fusion ensemble and evaluation

The independent pipeline checkpoints can be combined without retraining them.
Each expert produces a class-probability vector:

```text
p_fused = sum(stream_weight[i] * p_i)
prediction = argmax(p_fused)
```

The ensemble weights are configured in the evaluation and XAI scripts. The
supported experiment labels are `pure`, `2-stream`, `3-stream`, and
`4-stream`. A stream with weight zero is not loaded, which saves GPU memory
and avoids unnecessary inference.

For example, the X-SUB four-stream configuration is:

```text
JBV       0.30
Bones     0.20
Joints    0.20
Velocity  0.30
```

The important distinction is:

- **pipeline training:** one representation, one checkpoint family, one
  classifier;
- **ensemble evaluation:** several already-trained experts, weighted
  probability fusion;
- **XAI fusion:** the same stream weights fuse both class probabilities and
  the stream-specific differential heatmaps.

The standard evaluation flow is:

```text
trained checkpoints
        |
        +--> individual expert inference
        |
        +--> softmax probabilities
                    |
                    v
             weighted late fusion
                    |
                    v
              top-1 prediction
```

Validation-set ensemble tuning belongs in `evaluation\tune_ensemble.py`.
Final test evaluation should use the selected weights only after tuning is
complete.

---

## 5. XAI and explanation validation pipeline

The explanation workflow uses the same forward pass as ensemble evaluation,
but requests attention maps from the temporal brain:

```text
test skeleton
    |
    v
each active expert
    |
    +--> spatial joint attention
    +--> temporal video-token attention
    |
    v
spatial attention x temporal attention
    |
    v
differential fold-change filtering
    |
    v
weighted stream heatmap fusion
    |
    v
global normalization to [0, 1]
    |
    +--> .npy heatmap
    +--> animated GIF
    +--> peak-frame PNG
    +--> explanation_manifest.json
```

For XAI mode, the temporal brain returns:

- temporal importance for each body and frame, taken from the video token's
  attention to temporal tokens;
- spatial importance for each body, frame, and joint, computed from the
  global-node query against physical-joint keys; and
- their product, which creates the fused spatio-temporal importance matrix.

`utils\xai_extractor.py` then applies differential filtering. For each body,
it identifies the highest-entropy frame as a resting baseline and keeps
positive fold changes relative to that baseline. This emphasizes where
attention becomes more concentrated during the action rather than simply
showing uniformly high attention. The active stream heatmaps are then combined
using the same ensemble weights used for class probabilities and normalized
globally per video.

The validation engines consume the generated `.npy` heatmaps:

| Validator | Question answered | Main output |
|---|---|---|
| Fidelity | Does removing highly attributed joints reduce true-class confidence? | Deletion/insertion curves and summary JSON |
| Semantic Pointing Game | Does the peak joint fall in an action-relevant anatomical region? | Hit rate and anatomical attribution details |

The dashboard publisher converts NPY heatmaps to browser-readable JSON and
copies the explanation, fidelity, and semantic metadata into
`web_app/public/results/`.

---

## 6. Repository architecture

```text
REAT-3D_SAR/
├── preprocessing/
│   ├── 01_data_splitter.py       raw skeleton -> train/validation files
│   ├── 02_val_test_split.py      validation -> test split
│   └── 03_convert_to_binary.py   .skeleton -> .pt tensors
├── utils/
│   ├── dataset.py                loading and nine-channel engineering
│   ├── pipeline_config.py        datasets, streams, paths, channel slicing
│   ├── action_labels.py          NTU action names
│   └── xai_extractor.py          differential heatmap filtering
├── models/
│   ├── spatial_gcn.py            biological graph message passing
│   └── temporal_brain.py         Flex spatial attention + temporal attention
├── train60.py / train120.py      one-stream training entry points
├── evaluation/
│   ├── evaluate.py               fixed two-stream test evaluation
│   ├── ablation_evaluator.py     stream and combination comparisons
│   └── tune_ensemble.py          validation-set fusion-weight search
├── xai_validation/
│   ├── xai_explainer.py          ensemble predictions and heatmaps
│   ├── proving_engine_fidelity.py deletion/insertion validation
│   └── proving_engine_semantic.py pointing-game validation
├── tools/
│   └── publish_results.py         source results -> dashboard assets
├── saved_weights/                 trained expert checkpoints
├── results/                       source XAI and validation bundles
└── web_app/                       optional result dashboard
```

---

## 7. Requirements

### 7.1 Hardware

Training and XAI generation are designed for a CUDA-capable GPU. The code
automatically selects CUDA when PyTorch detects it and otherwise uses the CPU.
CPU execution is useful for smoke tests, but full NTU-RGB+D training and
explanation generation can be very slow and may require reducing batch size or
the number of processed samples.

The models use up to two bodies, 100 frames, and 25 joints per sample:

```text
(bodies=2, frames=100, joints=25, channels=9)
```

### 7.2 Python environment

The repository includes `requirements.txt`. PyTorch is intentionally not
listed there because the correct PyTorch package depends on the CUDA version.
For the existing Windows setup, activate the project environment first:

```powershell
conda activate fyp_env
```

If PowerShell does not recognize `conda activate`, initialize Conda for
PowerShell once and restart the terminal:

```powershell
conda init powershell
```

Create a new environment instead if required:

```powershell
conda create -n fyp_env python=3.11 -y
conda activate fyp_env
```

Install PyTorch using the command appropriate for the installed CUDA runtime.
For example, the dependency notes in this repository use CUDA 12.4:

```powershell
python -m pip install torch==2.5.1 torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import numpy, pandas, matplotlib, wandb; print('Python dependencies OK')"
```

The main Python dependencies are:

| Package | Used for |
|---|---|
| PyTorch | Model training, inference, and tensor processing |
| NumPy | Heatmaps and numeric processing |
| pandas | Evaluation and result tables |
| matplotlib | XAI GIF and PNG generation |
| Pillow | Image/GIF support |
| tqdm | Progress bars |
| Weights & Biases | Training and evaluation logging |
| thop | Complexity profiling |
| plotly | Profiling/visualization utilities |

### 7.3 Weights & Biases

Training and some evaluation scripts call `wandb.init()`. Log in before
training if runs should be recorded remotely:

```powershell
wandb login
```

If W&B is not desired, configure it according to your local policy before
running the scripts. Do not commit API keys or other credentials to the
repository.

### 7.4 Web dashboard dependencies

The optional dashboard is under `web_app/`. Its dependencies are managed
separately with npm:

```powershell
cd web_app
npm install
cd ..
```

`web_app/node_modules/` is not needed for the Python workflow.

---

## 8. Dataset preparation

The project recognizes these dataset names:

| Dataset name | Classes | Intended benchmark |
|---|---:|---|
| `xview` | 60 | NTU-RGB+D 60 cross-view |
| `xsub` | 60 | NTU-RGB+D 60 cross-subject |
| `xsub120` | 120 | NTU-RGB+D 120 cross-subject |
| `xset120` | 120 | NTU-RGB+D 120 cross-setup |

The shared configuration is in `utils\pipeline_config.py`. It defines the
dataset roots, class counts, pipeline names, checkpoint names, and result
paths.

### 8.1 Expected raw NTU layout

For the NTU-RGB+D 120 splitter, place the raw skeleton files in:

```text
data/
└── nturgbd_skeletons_s018_to_s032/
    ├── S001C001P001R001A001.skeleton
    ├── ...
```

Also keep the missing/corrupted sample list at the repository root:

```text
NTU_RGBD120_samples_with_missing_skeletons.txt
```

The splitter skips files listed in this file. If the file is absent, the
splitter continues but prints a warning and does not exclude corrupted names.

### 8.2 NTU-RGB+D 120: create train and validation data

Open `preprocessing\01_data_splitter.py` and set:

```python
BENCHMARK_MODE = "xsub120"
RAW_DATA_DIR = "data/nturgbd_skeletons_s018_to_s032"
MISSING_SKELETONS_TXT = "NTU_RGBD120_samples_with_missing_skeletons.txt"
```

Use `xsub120` for the official NTU-120 cross-subject rule or `xset120` for
the cross-setup rule. The script creates:

```text
data/<dataset>/
├── train_skeletons/
└── val_skeletons/
```

Run it from the repository root:

```powershell
python preprocessing\01_data_splitter.py
```

The script copies the raw `.skeleton` files. It does not convert them to
PyTorch tensors.

### 8.3 Create the test split

Open `preprocessing\02_val_test_split.py` and set both values to the same
dataset:

```python
BENCHMARK_MODE = "xsub120"
DATASET_DIRECTORY = BENCHMARK_MODE
```

Run:

```powershell
python preprocessing\02_val_test_split.py
```

This uses `random.seed(42)`, shuffles the files currently in
`val_skeletons`, and moves half of them into `test_skeletons`. If matching
`.pt` files already exist, they are moved with their `.skeleton` files.

Important:

- This script **moves** files; it does not copy them.
- Do not run it twice on the same prepared directory unless the validation
  files have been restored.
- The resulting test split is the repository's deterministic random split,
  not an official NTU-60 test protocol.

The resulting layout is:

```text
data/xsub120/
├── train_skeletons/
├── val_skeletons/
└── test_skeletons/
```

### 8.4 Convert skeleton text files to binary tensors

Open `preprocessing\03_convert_to_binary.py` and make sure the directory list
contains the datasets you want to prepare:

```python
directories_to_process = [
    "data/xsub120/train_skeletons",
    "data/xsub120/val_skeletons",
    "data/xsub120/test_skeletons",
]
```

The converter:

1. creates `raw_text` and `binary_pt` directories;
2. moves loose `.skeleton` files into `raw_text`;
3. parses up to two bodies, 25 joints, and three coordinates per frame; and
4. writes matching `.pt` tensors into `binary_pt`.

Run:

```powershell
python preprocessing\03_convert_to_binary.py
```

The converter skips an output `.pt` file when it already exists, so rerunning
it is safe for already converted files. A prepared split should look like:

```text
data/xsub120/train_skeletons/
├── raw_text/
│   ├── S001C001P001R001A001.skeleton
│   └── ...
└── binary_pt/
    ├── S001C001P001R001A001.pt
    └── ...
```

### 8.5 Preparing NTU-RGB+D 60

`preprocessing\01_data_splitter.py` is intentionally an NTU-RGB+D 120
splitter. Do not substitute NTU-60 subject lists into it.

For `xsub` or `xview`, prepare the directories using the official NTU-60
split definitions and place the files in:

```text
data/xsub/train_skeletons/
data/xsub/val_skeletons/
data/xsub/test_skeletons/
```

or:

```text
data/xview/train_skeletons/
data/xview/val_skeletons/
data/xview/test_skeletons/
```

Then update `BENCHMARK_MODE` in `preprocessing\02_val_test_split.py` if that
script is being used for the validation/test division, and add the matching
directories to `directories_to_process` in
`preprocessing\03_convert_to_binary.py`.

Before training, verify that every selected split contains a `binary_pt`
directory with `.pt` files.

---

## 9. Feature engineering and input pipelines

`utils\dataset.py` loads each `.pt` tensor and creates a canonical nine-channel
representation:

| Channel range | Representation | Pipeline |
|---|---|---|
| `0:3` | Root-centered and height-normalized joint coordinates | `joints` |
| `3:6` | Bone vectors relative to the kinematic parent | `bones` |
| `6:9` | Frame-to-frame velocity | `velocity` |
| `0:9` | All three representations concatenated | `jbv` |

Each sample is padded or truncated to 100 frames. Skeletons are root-centered
using joint 0 and normalized using the spine height. The model receives two
body slots even when only one person is present; a body mask identifies empty
slots.

The four supported pipeline names are:

```text
jbv       9 input channels
joints    3 input channels
bones     3 input channels
velocity  3 input channels
```

Do not edit the dataset return statement to select a pipeline. Training and
inference use `select_pipeline_input()` to slice the canonical nine-channel
tensor consistently.

---

## 10. Configure and train models

There are separate training entry points:

- `train60.py` for `xsub` or `xview`;
- `train120.py` for `xsub120` or `xset120`.

Before running either script, edit the constants near the top:

```python
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

`DATASET_NAME` must match a key in `utils\pipeline_config.py`. `RUN_ID` is
part of the checkpoint path, so use a new value when you want to preserve an
older experiment.

The training hyperparameters are also defined in the script:

```python
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
```

The default training model is a spatial GCN followed by the temporal brain
and a linear classifier. The script checks that both the training and
validation `binary_pt` directories exist before starting.

### 10.1 Train one pipeline

For NTU-RGB+D 60:

```powershell
python train60.py --pipeline jbv
python train60.py --pipeline joints
python train60.py --pipeline bones
python train60.py --pipeline velocity
```

For NTU-RGB+D 120:

```powershell
python train120.py --pipeline jbv
python train120.py --pipeline joints
python train120.py --pipeline bones
python train120.py --pipeline velocity
```

Train only the streams needed for the planned experiment. For example, a
four-stream XAI experiment requires all four checkpoints, while a pure JBV
experiment requires only the JBV checkpoint.

### 10.2 Checkpoint layout

Checkpoints are written under:

```text
saved_weights/<dataset>/<run-id>/<checkpoint-name>/
```

The pipeline-to-directory mapping is:

| Pipeline | Checkpoint directory | Required files |
|---|---|---|
| `jbv` | `jbv` | `best_gcn.pth`, `best_transformer.pth`, `best_classifier.pth`, `metadata.json` |
| `joints` | `pure_joints` | same files |
| `bones` | `bones` | same files |
| `velocity` | `pure_velocity` | same files |

For example:

```text
saved_weights/xsub120/run17/
├── jbv/
│   ├── best_gcn.pth
│   ├── best_transformer.pth
│   ├── best_classifier.pth
│   └── metadata.json
├── pure_joints/
├── bones/
└── pure_velocity/
```

The `metadata.json` file records the dataset, run ID, pipeline, channel count,
frame limit, and class count. Keep checkpoints from different datasets and
runs in separate directories.

---

## 11. Evaluate trained models

All evaluation scripts use configuration constants rather than command-line
arguments. Edit `DATASET_NAME` and `RUN_ID` at the top of the script before
running it. Run evaluation from the repository root.

### 11.1 Fixed two-stream evaluation

`evaluation\evaluate.py` evaluates the JBV and bones experts with fixed fusion
weights. It currently uses:

```python
FUSION_WEIGHTS = {
    "JBV": 0.5,
    "B": 0.5,
}
```

Configure:

```python
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

Then run:

```powershell
python evaluation\evaluate.py
```

This loads the test split, computes probabilities from both experts, performs
weighted late fusion, and reports top-1 accuracy.

### 11.2 Four-stream ablation evaluation

`evaluation\ablation_evaluator.py` loads all four experts:

- JBV;
- bones;
- joints; and
- velocity.

Configure `DATASET_NAME` and `RUN_ID`, then run:

```powershell
python evaluation\ablation_evaluator.py
```

This script is intended for comparing individual streams and combinations.
It requires all four checkpoint directories for the selected dataset and run.

### 11.3 Tune ensemble weights

`evaluation\tune_ensemble.py` uses the validation split to search for fusion
weights. This is the correct place to tune weights because the test split
should remain unseen until final evaluation.

Configure `DATASET_NAME` and `RUN_ID`, verify that the corresponding JBV and
comparison checkpoint directories exist, then run:

```powershell
python evaluation\tune_ensemble.py
```

Do not use the test split for tuning. After selecting weights, record the
choice and use the untouched test split for final reporting.

### 11.4 Complexity profiling

The optional `complexity_profiler.py` reports analytic model complexity:

```powershell
python complexity_profiler.py
```

Use it after the model and pipeline configuration are stable. It is separate
from accuracy evaluation and does not create the XAI result bundle.

---

## 12. Configure ensemble XAI experiments

The XAI scripts use the same late-fusion expert definitions as the evaluation
workflow. The important configuration values are near the top of each file:

```python
CURRENT_DATASET = "XSUB120"
CURRENT_ENSEMBLE = "4-stream"
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

`CURRENT_DATASET` selects the uppercase key in `THESIS_CONFIGS`; `DATASET_NAME`
selects the lowercase dataset path. They must describe the same dataset:

| `CURRENT_DATASET` | `DATASET_NAME` |
|---|---|
| `X-VIEW` | `xview` |
| `X-SUB` | `xsub` |
| `XSET120` | `xset120` |
| `XSUB120` | `xsub120` |

Supported ensemble names are:

```text
pure
2-stream
3-stream
4-stream
```

The configured weights determine which checkpoint directories are loaded. For
example, a four-stream run normally requires JBV, bones, joints, and velocity
checkpoints.

The result directory is:

```text
results/<dataset>/<run-id>/<ensemble>/
```

Do not mix a result directory from one dataset with checkpoints from another.

---

## 13. Generate XAI explanations

Run the explainer first:

```powershell
python -m xai_validation.xai_explainer --start 0 --end 100
```

The range is zero-based and uses the sorted test dataset file list:

- `--start 0` starts at the first test sample;
- `--end 100` stops before index 100;
- omitting `--end` processes through the end of the dataset.

Examples:

```powershell
# Process the first 25 test samples
python -m xai_validation.xai_explainer --start 0 --end 25

# Continue with samples 25 through 49
python -m xai_validation.xai_explainer --start 25 --end 50

# Process the complete test split
python -m xai_validation.xai_explainer
```

The explainer is resumable for samples whose NPY, GIF, and PNG outputs already
exist. It writes:

```text
results/<dataset>/<run-id>/<ensemble>/
├── xai_npy/
│   └── <sample>_fused.npy
├── xai_gifs/
│   └── <sample>_fused.gif
├── xai_frames/
│   └── <sample>_fused_peak.png
└── explanation_manifest.json
```

The NPY heatmap represents joint importance over:

```text
(bodies=2, frames=100, joints=25)
```

The manifest records sample IDs and relative paths to the generated assets.
The explainer also records the predicted action, true action, and confidence
used by the dashboard.

The XAI generation order is mandatory:

1. Train the required expert pipelines.
2. Configure dataset, run ID, and ensemble in `xai_explainer.py`.
3. Run the explainer for the desired test range.
4. Run fidelity and semantic validation against those generated NPY files.

---

## 14. Validate the explanations

Both proving engines consume the NPY heatmaps created by the explainer. They
skip samples for which an NPY heatmap is not present.

### 14.1 Fidelity validation

Configure `DATASET_NAME`, `RUN_ID`, and `CURRENT_ENSEMBLE` in
`xai_validation\proving_engine_fidelity.py`, then run:

```powershell
python -m xai_validation.proving_engine_fidelity
```

The fidelity procedure:

1. loads the selected ensemble;
2. computes the confidence for the ground-truth class;
3. ranks joints using the generated heatmap;
4. masks or inserts the most important joints for several `K` values; and
5. writes detailed and summary fidelity results.

Output:

```text
results/<dataset>/<run-id>/<ensemble>/fidelity/
├── fidelity_details.csv
└── fidelity_results.json
```

### 14.2 Semantic validation

Configure the matching dataset, run ID, and ensemble in
`xai_validation\proving_engine_semantic.py`, then run:

```powershell
python -m xai_validation.proving_engine_semantic
```

The semantic engine applies the Pointing Game: it finds the most important
joint in the heatmap and checks whether it belongs to the expected anatomical
group for the action label.

Output:

```text
results/<dataset>/<run-id>/<ensemble>/semantic/
├── semantic_details.csv
└── semantic_results.json
```

The semantic engine contains explicit anatomical target mappings for all 120
NTU-RGB+D action indices. The classifier uses zero-based labels internally:
A1 maps to index 0 and A120 maps to index 119. The shared action-label list is
also used by the XAI manifest and semantic CSV output.

The semantic confidence formulas, interpretation, and dashboard data contract
are documented in `README_SEMANTIC_CONFIDENCE.md`.

### 14.3 Complete result bundle

A fully processed result bundle normally contains:

```text
results/<dataset>/<run-id>/<ensemble>/
├── explanation_manifest.json
├── xai_npy/
├── xai_gifs/
├── xai_frames/
├── fidelity/
│   ├── fidelity_details.csv
│   └── fidelity_results.json
└── semantic/
    ├── semantic_details.csv
    └── semantic_results.json
```

The Python validation results remain under `results/`. They are not
automatically copied into the dashboard directory.

---

## 15. View results in the web dashboard

The dashboard is optional. You do not need it for training, evaluation, or
XAI validation.

### 15.1 Publish a result bundle for the dashboard

The dashboard reads browser-ready files from:

```text
web_app/public/results/
```

The Python result bundle uses NPY heatmaps, but browsers cannot load those
directly. `tools\publish_results.py` converts each NPY heatmap to compact JSON,
copies the GIF/PNG and summary JSON assets, and updates `catalog.json`.

Run it only when you want to view a new or changed bundle:

```powershell
python tools\publish_results.py `
  --dataset xsub `
  --run-id weights_xsub-run17 `
  --ensemble 4-stream
```

The source directory must be:

```text
results/<dataset>/<run-id>/<ensemble>/
```

and must contain:

```text
explanation_manifest.json
fidelity/fidelity_results.json
semantic/semantic_results.json
```

The publisher also expects the referenced GIF, PNG, and NPY files to exist.
It writes converted heatmaps to:

```text
web_app/public/results/<dataset>/<run-id>/<ensemble>/xai_heatmaps/
```

If the dashboard is not part of the experiment, this step can be skipped.

### 15.2 Serve the dashboard locally

From the repository root:

```powershell
python -m http.server 8765 --directory web_app
```

Open:

```text
http://localhost:8765
```

Do not open `web_app\index.html` directly from the filesystem. Browsers
typically block the JSON and media requests when using a `file://` URL.

The dashboard supports:

- dataset, run, and ensemble selection;
- sample selection;
- predicted and true action display;
- confidence display;
- 25-joint skeleton visualization;
- temporal heatmap inspection;
- GIF and peak-frame PNG viewing;
- fidelity summary metrics; and
- semantic Pointing Game summary metrics.

### 15.3 Build or preview with Vite

For a Vite development server:

```powershell
cd web_app
npm run dev
```

For a production build:

```powershell
npm run build
npm run preview
```

The Vercel project root should be set to `web_app/`. The static result files
must be present under `web_app/public/results/` before deployment.

---

## 16. Recommended complete command sequence

The following is a practical NTU-RGB+D 120 cross-subject sequence. Adjust
dataset names, run IDs, and ensemble settings for the experiment.

### Step 1: activate and install

```powershell
conda activate fyp_env
python -m pip install -r requirements.txt
```

Install the CUDA-specific PyTorch package before the requirements file if it
is not already installed.

### Step 2: prepare the dataset

Edit `preprocessing\01_data_splitter.py`:

```python
BENCHMARK_MODE = "xsub120"
```

Then run:

```powershell
python preprocessing\01_data_splitter.py
python preprocessing\02_val_test_split.py
python preprocessing\03_convert_to_binary.py
```

### Step 3: configure training

Edit `train120.py`:

```python
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

Train the required streams:

```powershell
python train120.py --pipeline jbv
python train120.py --pipeline joints
python train120.py --pipeline bones
python train120.py --pipeline velocity
```

### Step 4: evaluate

Edit `evaluation\evaluate.py` to use the same dataset and run:

```python
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

Run the fixed two-stream evaluation:

```powershell
python evaluation\evaluate.py
```

Run the four-stream ablation evaluation if all four checkpoints are available:

```powershell
python evaluation\ablation_evaluator.py
```

### Step 5: configure and generate XAI

In all three XAI scripts, use matching settings. For example:

```python
CURRENT_DATASET = "XSUB120"
CURRENT_ENSEMBLE = "4-stream"
DATASET_NAME = "xsub120"
RUN_ID = "run17"
```

Generate explanations:

```powershell
python -m xai_validation.xai_explainer --start 0 --end 100
```

### Step 6: validate XAI

```powershell
python -m xai_validation.proving_engine_fidelity
python -m xai_validation.proving_engine_semantic
```

### Step 7: optionally publish and view

If the generated result directory is `results/xsub120/run17/4-stream/`:

```powershell
python tools\publish_results.py `
  --dataset xsub120 `
  --run-id run17 `
  --ensemble 4-stream

python -m http.server 8765 --directory web_app
```

Open `http://localhost:8765`.

---

## 17. Troubleshooting

### `ModuleNotFoundError`

Activate the intended Conda environment and install dependencies:

```powershell
conda activate fyp_env
python -m pip install -r requirements.txt
```

Confirm that `python` and `pip` belong to the same environment:

```powershell
where.exe python
python -m pip --version
```

### `torch.cuda.is_available()` is false

The code falls back to CPU. Check that the installed PyTorch build matches
the available CUDA setup:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If CUDA is required, reinstall PyTorch using the correct official index URL
for the machine's CUDA version.

### `Missing binary dataset split`

Training requires:

```text
data/<dataset>/train_skeletons/binary_pt/
data/<dataset>/val_skeletons/binary_pt/
```

Run the conversion script and ensure the selected directories are present in
`directories_to_process`.

### `FileNotFoundError` for a checkpoint

Confirm that the script's `DATASET_NAME` and `RUN_ID` match the checkpoint
directory and that the pipeline uses the correct checkpoint name:

```text
saved_weights/<dataset>/<run-id>/jbv/
saved_weights/<dataset>/<run-id>/pure_joints/
saved_weights/<dataset>/<run-id>/bones/
saved_weights/<dataset>/<run-id>/pure_velocity/
```

### XAI produces no validation rows

The proving engines only process samples with a corresponding:

```text
results/<dataset>/<run-id>/<ensemble>/xai_npy/<sample>_fused.npy
```

Run the explainer first and ensure that its dataset, run ID, and ensemble
match the proving engine settings.

### Dashboard reports missing JSON files

Run `tools\publish_results.py` for the exact source bundle and confirm that
the required files exist under `web_app\public\results\`. Serve `web_app` over
HTTP rather than opening `index.html` directly.

### Accidental data loss from the validation/test splitter

`02_val_test_split.py` moves files from validation to test. If it was run too
early or with the wrong dataset, restore the files from the original dataset
copy and repeat the preparation steps. Do not repeatedly run the script on
the same directory.

---

## 18. Important experiment hygiene

- Keep one `RUN_ID` for one reproducible training configuration.
- Use the same `DATASET_NAME` in training, evaluation, XAI, and publishing.
- Train only on `train_skeletons`; use validation for model selection or
  ensemble tuning; reserve test data for final evaluation and explanation
  validation.
- Do not tune ensemble weights on the test set.
- Keep raw data, generated checkpoints, and result bundles organized by
  dataset and run ID.
- Record any changes to batch size, frame count, split policy, learning rate,
  or ensemble weights alongside the experiment.
- Do not commit dataset archives, credentials, W&B API keys, or unnecessary
  generated artifacts.
- If a result is intended for the dashboard, publish it only after the
  Python result bundle is complete and internally consistent.

---

## 19. Repository map

```text
REAT-3D_SAR/
├── data/                         Prepared datasets
├── saved_weights/                Trained model checkpoints
├── results/                      Python evaluation and XAI outputs
├── models/
│   ├── spatial_gcn.py            Spatial graph convolution layer
│   └── temporal_brain.py         Temporal aggregation layer
├── preprocessing/
│   ├── 01_data_splitter.py       NTU-RGB+D 120 train/validation splitter
│   ├── 02_val_test_split.py      Deterministic validation/test mover
│   └── 03_convert_to_binary.py   Skeleton-to-PT conversion
├── utils/
│   ├── dataset.py                Loading and feature engineering
│   ├── pipeline_config.py        Shared datasets, pipelines, and paths
│   └── xai_extractor.py          Differential XAI implementation
├── evaluation/
│   ├── evaluate.py               Fixed two-stream test evaluation
│   ├── ablation_evaluator.py     Four-stream ablation evaluation
│   └── tune_ensemble.py          Validation-set fusion tuning
├── xai_validation/
│   ├── xai_explainer.py          Heatmaps, GIFs, PNGs, manifest
│   ├── proving_engine_fidelity.py Fidelity validation
│   └── proving_engine_semantic.py Pointing Game validation
├── tools/
│   └── publish_results.py        Optional dashboard result publisher
├── train60.py                    NTU-RGB+D 60 training entry point
├── train120.py                   NTU-RGB+D 120 training entry point
├── complexity_profiler.py        Model complexity report
└── web_app/                      Optional Vite/static dashboard
```
