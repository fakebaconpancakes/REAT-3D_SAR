# Rebuilding benchmark splits

The corrected experiments use separate `data/<protocol>_official_v2/` roots;
the previous split directories and checkpoints are left in place. The splitter
never overwrites an existing output. Each protocol needs the complete original
raw skeleton directory and the matching official missing-skeleton list.

## Official assignments and internal validation

- NTU RGB+D 60 `xsub`: official training subjects are
  `1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38`.
- NTU RGB+D 60 `xview`: cameras 2 and 3 are official training; camera 1 is
  official evaluation.
- NTU RGB+D 120 `xsub120`: use the 53 official training subjects encoded in
  `01_data_splitter.py`.
- NTU RGB+D 120 `xset120`: even setup IDs are official training; odd setup IDs
  are official evaluation.

For internal validation, seed 42 selects whole subjects within official
training for `xsub`, `xsub120`, and `xview`, and whole setups for `xset120`.
This prevents group overlap between train and validation. The resulting ratio
can differ from 90/10 because groups are indivisible; the actual ratio is
written to `split_summary.txt`. Official evaluation samples are never used for
model selection or ensemble tuning.

The splitter checks filenames, file readability/structure, sample uniqueness,
group leakage, exact train/validation coverage of the usable official training
partition, exact test coverage of the usable official evaluation partition,
and copied-file counts. Missing skeletons are excluded using the release lists;
unexpected malformed skeletons stop the build rather than being silently
discarded. `split_manifest.csv` records the assignment of each usable sample.

## Commands

Put the complete raw files at the default locations below, or pass `--raw-dir`:

```powershell
python preprocessing/01_data_splitter.py --protocol xsub
python preprocessing/01_data_splitter.py --protocol xview
python preprocessing/01_data_splitter.py --protocol xsub120
python preprocessing/01_data_splitter.py --protocol xset120
python preprocessing/03_convert_to_binary.py
```

Defaults are `data/nturgbd_skeletons` for NTU 60 and
`data/nturgbd_skeletons_s018_to_s032` for NTU 120. If the full raw data is
elsewhere, pass its directory with `--raw-dir`. Do not point the converter at
the old dataset roots. The legacy `02_val_test_split.py` is now a no-op.

After splitting and conversion, `utils/pipeline_config.py` points the training,
validation, test, tuning, and XAI loaders at the new `*_official_v2` roots.
Training and downstream scripts use run ID `run18_official_v2`, preserving
`run17` checkpoints and results. Train the models from scratch before running
ensemble tuning, final test evaluation, or final XAI evaluation.

## Assumptions

The filename fields follow the official NTU convention `SsssCcccPpppRrrrAaaa`.
The standard NTU RGB+D 60 camera convention is camera 1 for evaluation and
cameras 2/3 for training. The 120 cross-setup convention is even setups for
training and odd setups for evaluation. Counts in the generated summary are
derived from the actual complete raw directory after the published missing
skeleton exclusions, so an incomplete download will not be mistaken for the
expected full benchmark: the raw source itself must be complete.
