# REAT-3D_SAR

## Preprocessing Guide

Preprocessing is a three-step workflow. Run the scripts from the repository
root and complete them in this order:

```powershell
python preprocessing\01_data_splitter.py
python preprocessing\02_val_test_split.py
python preprocessing\03_convert_to_binary.py
```

The resulting layout for each dataset should be:

```text
data/<dataset>/
├── train_skeletons/
│   ├── raw_text/
│   └── binary_pt/
├── val_skeletons/
│   ├── raw_text/
│   └── binary_pt/
└── test_skeletons/
    ├── raw_text/
    └── binary_pt/
```

### Step 1: Split the NTU-RGB+D 120 source data

`preprocessing/01_data_splitter.py` is intentionally an NTU-RGB+D 120
splitter. It supports:

| `BENCHMARK_MODE` | Split rule | Output directory |
|---|---|---|
| `xsub120` | Official NTU-120 cross-subject training subjects | `data/xsub120/` |
| `xset120` | Even setups for training and odd setups for validation | `data/xset120/` |

Before running it, a coder may change only the dataset-specific configuration
near the top of the file:

```python
BENCHMARK_MODE = "xsub120"
RAW_DATA_DIR = "data/nturgbd_skeletons_s018_to_s032"
MISSING_SKELETONS_TXT = "NTU_RGBD120_samples_with_missing_skeletons.txt"
```

Change `BENCHMARK_MODE` to select `xsub120` or `xset120`. Change
`RAW_DATA_DIR` if the downloaded NTU-120 skeleton files are stored elsewhere.
Change `MISSING_SKELETONS_TXT` if the official missing-skeleton list has a
different location.

Do not change the `training_subjects` list, filename parsing, or routing logic
unless you are intentionally replacing the official NTU-120 split rules.
This script does not support NTU-60. To prepare `xsub` or `xview`, use a
separate NTU-60 splitter with the official NTU-60 cross-subject or cross-view
split definitions; do not substitute NTU-60 lists into this NTU-120 script.

### Step 2: Create the validation/test split

`preprocessing/02_val_test_split.py` takes the generated validation directory
and moves half of its raw skeleton files into `test_skeletons/`. The split is
deterministic because it uses `random.seed(42)`, and matching `.pt` files are
moved when they already exist.

Change only these settings:

```python
BENCHMARK_MODE = "xsub120"
DATASET_DIRECTORY = BENCHMARK_MODE
```

Set both values to the dataset directory being prepared:
`xsub120`, `xset120`, `xsub`, or `xview`. Do not run this script twice on the
same prepared directory unless you have restored the validation files, because
it moves files rather than copying them.

For a publication-quality NTU-60 experiment, replace this random split with
the official NTU-60 test protocol or a documented split policy. The script's
random split is intended for the current preprocessing workflow and should not
be presented as an official benchmark split without justification.

### Step 3: Convert skeleton text files to binary tensors

`preprocessing/03_convert_to_binary.py` organizes raw `.skeleton` files into
`raw_text/` and creates matching PyTorch tensors under `binary_pt/`.
The parser expects standard NTU skeleton files and stores up to two bodies,
25 joints, and three coordinates per frame.

Before running it, update only the `directories_to_process` list so it
contains the train, validation, and test directories that actually exist:

```python
directories_to_process = [
    "data/xsub120/train_skeletons",
    "data/xsub120/val_skeletons",
    "data/xsub120/test_skeletons",
    "data/xset120/train_skeletons",
    "data/xset120/val_skeletons",
    "data/xset120/test_skeletons",
]
```

Add the corresponding `data/xsub/...` or `data/xview/...` directories when
preparing NTU-60 data. Do not change `parse_single_skeleton()` or the output
names unless the source file format changes. Existing `.pt` files are skipped,
so rerunning the converter is safe for already converted files.

Preprocessing must be completed before training or evaluation. Training reads
`train_skeletons/binary_pt/` and `val_skeletons/binary_pt/`; evaluation reads
the matching `test_skeletons/binary_pt/` directory.

## Training Guide

Training is controlled by the `--pipeline` option. Do not edit the return
statement in `utils/dataset.py` or change the GCN input width manually. The
dataset always produces the canonical nine-channel tensor, and the selected
pipeline determines which channels are passed to the model.

Run training commands from the repository root:

```powershell
python train60.py --pipeline jbv
python train60.py --pipeline joints
python train60.py --pipeline bones
python train60.py --pipeline velocity

python train120.py --pipeline jbv
python train120.py --pipeline joints
python train120.py --pipeline bones
python train120.py --pipeline velocity
```

The available pipelines are:

| Pipeline | Input channels | Representation |
|---|---:|---|
| `jbv` | 9 | Relative joints + bones + velocity |
| `joints` | 3 | Relative joint coordinates |
| `bones` | 3 | Bone vectors |
| `velocity` | 3 | Temporal velocity |

The project uses four independent dataset configurations. Every dataset has
its own `train_skeletons/`, `val_skeletons/`, and `test_skeletons/` folders:

| Dataset | Classes | Training script | Dataset root |
|---|---:|---|---|
| `xview` | 60 | `train60.py` | `data/xview/` |
| `xsub` | 60 | `train60.py` | `data/xsub/` |
| `xsub120` | 120 | `train120.py` | `data/xsub120/` |
| `xset120` | 120 | `train120.py` | `data/xset120/` |

The training scripts currently select their dataset through the
`DATASET_NAME` constant near the top of the file. Set it to the intended
dataset before running that script; do not confuse the dataset name with the
pipeline name. For example, set `DATASET_NAME = "xsub"` in `train60.py` to
train the 60-class cross-subject dataset, or set
`DATASET_NAME = "xset120"` in `train120.py` to train the 120-class
cross-setup dataset. Each selected dataset must contain both
`train_skeletons/binary_pt/` and `val_skeletons/binary_pt/`.

The training split and validation split are used only for model training and
model selection. The corresponding `test_skeletons/` split is reserved for
the evaluation workflows below, where trained JBV, joints, bones, and
velocity pipelines are combined and tested. Test-only data is not sufficient
for training.

The run identifier is configured inside each training script through
`RUN_ID` (currently `run17`). Checkpoints are written automatically to:

```text
saved_weights/<dataset>/<run_id>/<pipeline>/
```

For example, an NTU-RGB+D 120 JBV run writes to
`saved_weights/xsub120/run17/jbv/`. The directory contains the best model
weights and a `metadata.json` file describing the dataset, pipeline, channel
count, frame limit, and class count. Checkpoints for the four datasets are
kept separate by the dataset component of this path.

Training uses CUDA when it is available and otherwise falls back to CPU.
However, full training is intended for the configured GPU environment and may
be impractical or unsupported on a CPU-only machine. Before starting a run,
install the project dependencies, prepare both training and validation splits,
and configure Weights & Biases if logging is enabled by the environment.

## XAI Explanation and Validation

The XAI workflow is split into an explanation generator and two independent
validation engines:

```powershell
python -m xai_validation.xai_explainer --start 0 --end 100
python -m xai_validation.proving_engine_fidelity
python -m xai_validation.proving_engine_semantic
```

Before running these scripts, set the matching `DATASET_NAME`, `RUN_ID`, and
ensemble configuration in each script. The explainer must run first because
the validation engines consume its fused NPY heatmaps.

Results are organized by dataset, run, and ensemble:

```text
results/<dataset>/<run_id>/<ensemble>/
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

The explainer produces NPY heatmaps, GIF animations, static peak-frame PNGs,
and an asset manifest. The fidelity and semantic engines produce detailed CSV
records plus JSON summaries intended for later frontend or Vercel use. The
JSON files use relative asset paths and normalized numeric metrics; they do
not contain absolute local filesystem paths.

## Evaluation Workflows

The repository provides two evaluation paths. Choose the path that matches the experiment you want to report.

### Individual two-stream pipeline

Use this path when evaluating one selected two-stream ensemble, such as the Kinematic (`JBV`) stream combined with one structural stream:

1. Run `tune_ensemble.py` using the validation set to search for the best fusion coefficient (`beta`).
2. Apply the reported beta to the fusion weights in `evaluate.py`. Do not tune the coefficient on the test set.
3. Run `evaluate.py` using the untouched test set to report the final accuracy and metrics.

`tune_ensemble.py` is therefore the validation-time tuning stage, while `evaluate.py` is the final test-time reporting stage. The current `evaluate.py` contains a fixed fusion ratio, so the selected beta must be transferred to that script before evaluation.

### Complete multi-stream ablation

Use `ablation_evaluator.py` when the goal is to evaluate every supported stream combination rather than one selected ensemble. It loads the four trained streams:

- `JBV` — 9-channel Joints, Bones, and Velocity stream
- `B` — Bones stream
- `J` — Joints stream
- `V` — Velocity stream

The ablation evaluator searches fusion weights itself and reports the best result for every 2-stream, 3-stream, and 4-stream combination. Do not run `tune_ensemble.py` and `evaluate.py` first for this experiment; `ablation_evaluator.py` is the self-contained workflow for the complete combination study.

| Experiment goal | Workflow |
|---|---|
| One selected two-stream ensemble | `tune_ensemble.py` -> transfer beta to `evaluate.py` -> `evaluate.py` |
| Every supported stream combination | `ablation_evaluator.py` |

The validation set must be used for selecting fusion weights, and the test set must remain isolated for final reporting.

Before running an evaluation workflow, set its `DATASET_NAME` and `RUN_ID`
constants to the same dataset and run used during training. The evaluator
must load checkpoints from all pipelines for that dataset, and it must read
the matching `test_skeletons/` split. Do not combine checkpoints or test data
from different dataset configurations.

## Running Evaluation Scripts

Evaluation workflows are kept in the `evaluation/` package:

```text
evaluation/
├── tune_ensemble.py
├── evaluate.py
└── ablation_evaluator.py
```

Always run these commands from the repository root, not after changing into the `evaluation/` directory:

```powershell
python -m evaluation.tune_ensemble
python -m evaluation.evaluate
python -m evaluation.ablation_evaluator
```

Use module execution rather than running the files by path. The evaluation scripts use absolute imports from `models` and `utils`, and they use repository-root-relative paths for `data/` and `saved_weights/`. Running them as modules from the repository root keeps those imports and paths resolvable without modifying the existing import statements.

Do not use:

```powershell
cd evaluation
python evaluate.py
```

That invocation can remove the repository root from the import search path and cause `models` or `utils` import failures.
