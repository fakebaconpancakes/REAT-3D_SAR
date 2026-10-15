"""Build safe, reproducible NTU RGB+D benchmark splits.

Outputs are written to data/<protocol>_official_v2. Existing outputs are never
overwritten. Raw source files remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

SEED = 42
PROTOCOLS = ("xsub", "xview", "xsub120", "xset120")
# Published sample counts before the release's missing/incomplete-skeleton exclusions.
OFFICIAL_COUNTS = {
    "xsub": (56880, 16560),
    "xview": (56880, 18960),
    "xsub120": (114480, 51454),
    "xset120": (114480, 60012),
}
NTU60_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}
NTU120_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34,
    35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74,
    78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
}
SAMPLE_RE = re.compile(r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R\d{3}A(?P<action>\d{3})$")


def sample_ids(path: Path) -> dict[str, int]:
    match = SAMPLE_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Unexpected NTU skeleton filename: {path.name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def official_partition(protocol: str, ids: dict[str, int]) -> str:
    if protocol == "xsub":
        return "train" if ids["subject"] in NTU60_TRAIN_SUBJECTS else "test"
    if protocol == "xview":
        return "train" if ids["camera"] in {2, 3} else "test"
    if protocol == "xsub120":
        return "train" if ids["subject"] in NTU120_TRAIN_SUBJECTS else "test"
    if protocol == "xset120":
        return "train" if ids["setup"] % 2 == 0 else "test"
    raise ValueError(f"Unsupported protocol: {protocol}")


def validation_group(protocol: str, ids: dict[str, int]) -> int:
    # Keep the official grouping dimension disjoint inside the official train set.
    if protocol in {"xsub", "xsub120", "xview"}:
        return ids["subject"]
    return ids["setup"]


def choose_validation_groups(records: list[dict], protocol: str) -> set[int]:
    groups: dict[int, int] = defaultdict(int)
    for record in records:
        groups[validation_group(protocol, record["ids"])] += 1
    if len(groups) < 2:
        raise ValueError(f"{protocol}: need at least two training groups for group-aware validation")
    target = len(records) * 0.10
    candidates = list(groups)
    random.Random(SEED).shuffle(candidates)
    # Greedy subset selection, preferring a size close to 10% while retaining
    # at least one group for gradient-based training.
    selected: set[int] = set()
    selected_count = 0
    remaining = set(candidates)
    while len(remaining) > 1:
        group = min(remaining, key=lambda value: abs(target - (selected_count + groups[value])))
        if abs(target - selected_count) <= abs(target - (selected_count + groups[group])):
            break
        selected.add(group)
        selected_count += groups[group]
        remaining.remove(group)
    if not selected:
        selected.add(min(candidates, key=lambda value: abs(target - groups[value])))
    return selected


def validate_skeleton(path: Path) -> None:
    """Fail on unreadable/truncated files instead of silently omitting samples."""
    with path.open("r", encoding="utf-8", errors="strict") as source:
        lines = source.readlines()
    if not lines:
        raise ValueError(f"Empty skeleton file: {path}")
    frames = int(lines[0].strip())
    if frames <= 0:
        raise ValueError(f"Invalid frame count in {path}")
    cursor = 1
    for frame in range(frames):
        if cursor >= len(lines):
            raise ValueError(f"Truncated {path} at frame {frame}")
        bodies = int(lines[cursor].strip())
        cursor += 1
        for _ in range(bodies):
            # NTU format: body metadata, joint count, then one line per joint.
            cursor += 1  # skip body metadata
            if cursor >= len(lines):
                raise ValueError(f"Truncated body data in {path}")
            joints = int(lines[cursor].strip())
            if joints != 25:
                raise ValueError(f"Expected 25 joints, found {joints} in {path}")
            cursor += 1
            joint_lines = lines[cursor:cursor + joints]
            if len(joint_lines) != joints:
                raise ValueError(f"Truncated joint data in {path}")
            for joint_line in joint_lines:
                coordinates = joint_line.split()
                if len(coordinates) < 3:
                    raise ValueError(f"Malformed joint coordinates in {path}")
                float(coordinates[0]); float(coordinates[1]); float(coordinates[2])
            cursor += joints
            if cursor > len(lines):
                raise ValueError(f"Truncated joint data in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Directory containing the official raw .skeleton files")
    parser.add_argument("--output-root", type=Path, help="New output root; must not already exist")
    args = parser.parse_args()

    is_120 = args.protocol.endswith("120")
    default_raw = Path("data/nturgbd_skeletons_s018_to_s032" if is_120 else "data/nturgbd_skeletons")
    raw_dir = args.raw_dir or default_raw
    output_root = args.output_root or Path("data") / f"{args.protocol}_official_v2"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw skeleton directory not found: {raw_dir}. No split was created.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}. Choose a new --output-root.")

    missing_file = Path("NTU_RGBD120_samples_with_missing_skeletons.txt" if is_120 else "NTU_RGBD_samples_with_missing_skeletons.txt")
    if not missing_file.is_file():
        raise FileNotFoundError(f"Required official missing-skeleton list not found: {missing_file}")
    missing = {line.strip().split()[0] for line in missing_file.read_text(encoding="utf-8").splitlines()
               if line.strip() and line.strip().startswith("S")}
    expected_total, nominal_test_count = OFFICIAL_COUNTS[args.protocol]
    missing_test_count = 0
    for name in missing:
        missing_test_count += official_partition(args.protocol, sample_ids(Path(name))) == "test"
    expected_test_count = nominal_test_count - missing_test_count

    files = sorted(raw_dir.glob("*.skeleton"))
    if not files:
        raise FileNotFoundError(f"No .skeleton files found in {raw_dir}")
    records = []
    seen = set()
    for path in tqdm(files, desc="Validating raw skeletons", unit="file"):
        ids = sample_ids(path)
        if path.stem in seen:
            raise ValueError(f"Duplicate sample identifier in raw data: {path.stem}")
        seen.add(path.stem)
        if path.stem in missing:
            continue
        validate_skeleton(path)
        official = official_partition(args.protocol, ids)
        records.append({"path": path, "ids": ids, "official": official})

    official_train = [record for record in records if record["official"] == "train"]
    official_test = [record for record in records if record["official"] == "test"]
    expected_usable_total = expected_total - len(missing)
    if len(records) != expected_usable_total:
        raise RuntimeError(
            f"Incomplete or unexpected raw dataset: found {len(records)} usable samples; "
            f"expected {expected_usable_total} ({expected_total} official samples minus {len(missing)} listed exclusions)."
        )
    if len(official_test) != expected_test_count:
        raise RuntimeError(
            f"Incomplete official evaluation partition: found {len(official_test)} usable samples; "
            f"expected {expected_test_count}."
        )
    val_groups = choose_validation_groups(official_train, args.protocol)
    for record in official_train:
        record["split"] = "val" if validation_group(args.protocol, record["ids"]) in val_groups else "train"
    for record in official_test:
        record["split"] = "test"

    split_ids = {name: {record["path"].stem for record in records if record["split"] == name}
                 for name in ("train", "val", "test")}
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Split overlap detected")
    if split_ids["train"] | split_ids["val"] != {r["path"].stem for r in official_train}:
        raise RuntimeError("Training and validation do not exactly partition the official training samples")
    if split_ids["test"] != {r["path"].stem for r in official_test}:
        raise RuntimeError("Test split does not exactly match the official evaluation samples")
    train_groups = {validation_group(args.protocol, r["ids"])
                    for r in records if r["split"] == "train"}
    actual_val_groups = {validation_group(args.protocol, r["ids"]) for r in records if r["split"] == "val"}
    if train_groups & actual_val_groups:
        raise RuntimeError("Group leakage detected between train and validation")

    # Build beside the destination, then atomically publish only after checks.
    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {staging}")
    try:
        for split in ("train", "val", "test"):
            (staging / f"{split}_skeletons").mkdir(parents=True)
        for record in tqdm(records, desc="Copying verified split", unit="file"):
            destination = staging / f"{record['split']}_skeletons" / record["path"].name
            shutil.copy2(record["path"], destination)

        for split in ("train", "val", "test"):
            copied_ids = {path.stem for path in (staging / f"{split}_skeletons").glob("*.skeleton")}
            if copied_ids != split_ids[split]:
                raise RuntimeError(f"Copied {split} files do not match the verified assignment")

        manifest_path = staging / "split_manifest.csv"
        fields = ["filename", "dataset", "benchmark_protocol", "official_partition", "assigned_split",
                  "subject_id", "camera_id", "setup_id"]
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({"filename": record["path"].name, "dataset": "ntu120" if is_120 else "ntu60",
                                 "benchmark_protocol": args.protocol, "official_partition": record["official"],
                                 "assigned_split": record["split"], "subject_id": record["ids"]["subject"],
                                 "camera_id": record["ids"]["camera"], "setup_id": record["ids"]["setup"]})
        total = len(records)
        with (staging / "split_summary.txt").open("w", encoding="utf-8") as summary:
            summary.write(f"Protocol: {args.protocol}\nSeed: {SEED}\n")
            summary.write(f"Official evaluation samples expected after listed exclusions: {expected_test_count}\n")
            summary.write(f"Test samples copied: {len(split_ids['test'])}\n")
            for split in ("train", "val", "test"):
                count = len(split_ids[split])
                summary.write(f"{split}: {count} ({100 * count / total:.3f}%)\n")
            summary.write(f"Total usable samples: {total}\n")
            summary.write(f"Validation target: 10% of official training samples; actual: {100 * len(split_ids['val']) / len(official_train):.3f}%\n")
        staging.rename(output_root)
    except Exception:
        # Keep the staging tree for inspection; never remove possibly useful data.
        raise
    print(f"Created {output_root}; official test samples: {len(split_ids['test'])}; internal validation: {len(split_ids['val'])}/{len(official_train)}")


if __name__ == "__main__":
    main()
