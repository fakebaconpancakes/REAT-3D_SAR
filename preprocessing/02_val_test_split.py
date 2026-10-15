"""Deprecated compatibility stub.

The benchmark-aware preprocessing/01_data_splitter.py now creates train, val,
and test directly. This legacy script intentionally performs no file moves.
"""
import warnings

warnings.warn(
    "02_val_test_split.py no longer moves samples. Run 01_data_splitter.py "
    "with --protocol to construct a fresh *_official_v2 dataset.",
    DeprecationWarning,
    stacklevel=2,
)
print("Deprecated: no files were changed. Use preprocessing/01_data_splitter.py.")
