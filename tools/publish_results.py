import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEB_RESULTS_ROOT = REPOSITORY_ROOT / "web_app" / "public" / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a REAT result bundle for the web dashboard."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ensemble", required=True)
    return parser.parse_args()


def copy_file(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(f"Missing result asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def publish_bundle(dataset: str, run_id: str, ensemble: str):
    source_root = REPOSITORY_ROOT / "results" / dataset / run_id / ensemble
    destination_root = WEB_RESULTS_ROOT / dataset / run_id / ensemble
    if not source_root.is_dir():
        raise FileNotFoundError(f"Result bundle does not exist: {source_root}")

    manifest_path = source_root / "explanation_manifest.json"
    fidelity_path = source_root / "fidelity" / "fidelity_results.json"
    semantic_path = source_root / "semantic" / "semantic_results.json"
    for required_file in (manifest_path, fidelity_path, semantic_path):
        if not required_file.is_file():
            raise FileNotFoundError(f"Missing required result file: {required_file}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantic_details_path = source_root / "semantic" / "semantic_details.json"
    semantic_details = {}
    if semantic_details_path.is_file():
        semantic_details = {
            sample["sample_id"]: sample
            for sample in json.loads(
                semantic_details_path.read_text(encoding="utf-8")
            ).get("samples", [])
        }
    fidelity_details = {}
    fidelity_details_path = source_root / "fidelity" / "fidelity_details.csv"
    if fidelity_details_path.is_file():
        with fidelity_details_path.open(newline="", encoding="utf-8") as details_file:
            fidelity_details = {
                row["sample_id"]: row
                for row in csv.DictReader(details_file)
            }
    for sample in manifest.get("samples", []):
        detail = semantic_details.get(sample.get("sample_id"), {})
        fidelity = fidelity_details.get(sample.get("sample_id"), {})
        action_label = detail.get("action_label")
        if action_label:
            sample.setdefault("true_action", action_label)
            sample.setdefault("predicted_action", action_label)
        if "action_idx" in detail:
            sample.setdefault("true_label", detail["action_idx"])
            sample.setdefault("predicted_label", detail["action_idx"])
        if "base_conf" in fidelity:
            sample.setdefault("predicted_confidence", float(fidelity["base_conf"]))
    manifest["heatmap_format"] = "json"
    manifest["heatmap_directory"] = "xai_heatmaps"

    for directory_name in ("xai_gifs", "xai_frames"):
        for source_file in (source_root / directory_name).iterdir():
            if source_file.is_file():
                copy_file(source_file, destination_root / directory_name / source_file.name)

    heatmap_directory = destination_root / "xai_heatmaps"
    for sample in manifest.get("samples", []):
        sample_id = sample["sample_id"]
        source_npy = source_root / sample["heatmap"]
        heatmap = np.load(source_npy)
        output_path = heatmap_directory / f"{sample_id}_fused.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample["heatmap"] = str(output_path.relative_to(destination_root)).replace("\\", "/")
        sample["heatmap_shape"] = list(heatmap.shape)
        sample["heatmap_min"] = float(np.min(heatmap))
        sample["heatmap_max"] = float(np.max(heatmap))
        output_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "shape": list(heatmap.shape),
                    "values": heatmap.astype(float).tolist(),
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "explanation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    copy_file(fidelity_path, destination_root / "fidelity" / fidelity_path.name)
    if fidelity_details_path.is_file():
        fidelity_json = [
            {
                key: (float(value) if key != "sample_id" else value)
                for key, value in row.items()
            }
            for row in fidelity_details.values()
        ]
        (destination_root / "fidelity" / "fidelity_details.json").write_text(
            json.dumps(fidelity_json, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    copy_file(semantic_path, destination_root / "semantic" / semantic_path.name)
    if semantic_details_path.is_file():
        copy_file(
            semantic_details_path,
            destination_root / "semantic" / semantic_details_path.name,
        )

    catalog_path = WEB_RESULTS_ROOT / "catalog.json"
    catalog = {"schema_version": 1, "datasets": []}
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    datasets = {
        item["dataset"]: item
        for item in catalog.get("datasets", [])
    }
    dataset_entry = datasets.setdefault(dataset, {"dataset": dataset, "runs": []})
    run_entry = next(
        (item for item in dataset_entry["runs"] if item["run_id"] == run_id),
        None,
    )
    if run_entry is None:
        run_entry = {"run_id": run_id, "ensembles": []}
        dataset_entry["runs"].append(run_entry)
    if ensemble not in run_entry["ensembles"]:
        run_entry["ensembles"].append(ensemble)

    catalog["datasets"] = sorted(datasets.values(), key=lambda item: item["dataset"])
    for item in catalog["datasets"]:
        item["runs"] = sorted(item["runs"], key=lambda run: run["run_id"])
        for run in item["runs"]:
            run["ensembles"] = sorted(run["ensembles"])
    WEB_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    arguments = parse_args()
    publish_bundle(arguments.dataset, arguments.run_id, arguments.ensemble)
    print(
        f"Published {arguments.dataset}/{arguments.run_id}/{arguments.ensemble} "
        f"to {WEB_RESULTS_ROOT}"
    )
