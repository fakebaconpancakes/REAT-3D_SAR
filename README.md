# REAT-3D_SAR

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
