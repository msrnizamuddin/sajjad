#!/usr/bin/env python3
"""Convert a Kaggle mirror of the official HIBA collection (ISIC collection 251)
into notebook 09's simple (image_id, image_path, label) external-evaluation CSV.

This is NOT the official acquisition path (scripts/acquire_hiba_official_metadata.py
+ scripts/audit_hiba_external_dataset.py), which expects that script's own JSONL
output format and produces the canonical, checksum-verified
data/manifests/hiba_dataset_manifest.csv. Use this converter only when working
from a third-party republication of the same official ISIC collection (e.g. a
Kaggle mirror) that already ships the official per-image metadata.csv schema
(isic_id, copyright_license, diagnosis, ...), as a faster path to a usable
external-evaluation CSV while the official acquisition path is unavailable.

The diagnosis -> final-label mapping is read from
configs/datasets/hiba_external_label_mapping.yaml, the same file the official
audit path uses, so both paths agree on label semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to the mirror's metadata.csv (default: <project-root>/HIBA/metadata.csv)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Path to the mirror's images directory (default: <project-root>/HIBA/images)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to the label mapping YAML (default: configs/datasets/hiba_external_label_mapping.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <project-root>/data/manifests/hiba_for_notebook09.csv)",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    metadata_path = args.metadata or (root / "HIBA/metadata.csv")
    images_dir = args.images_dir or (root / "HIBA/images")
    mapping_path = args.mapping or (root / "configs/datasets/hiba_external_label_mapping.yaml")
    output_path = args.output or (root / "data/manifests/hiba_for_notebook09.csv")

    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    accepted_licenses = set(mapping["licence_policy"]["accepted_values"])
    diagnosis_mapping = mapping["mappings"]

    df = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    total_rows = len(df)

    # License gate, matching licence_policy.unknown_unsupported_conflicting_or_missing_policy.
    license_ok = df["copyright_license"].isin(accepted_licenses)
    excluded_license = int((~license_ok).sum())

    # Diagnosis gate: only diagnoses explicitly present in the mapping and marked
    # include_primary_evaluation=True are kept -- everything else (including
    # actinic keratosis, and any diagnosis string not in the mapping at all) is
    # excluded, matching matching_policy.unknown_value_policy.
    def resolve_label(diagnosis: str) -> str | None:
        entry = diagnosis_mapping.get(diagnosis)
        if entry is None or not entry.get("include_primary_evaluation"):
            return None
        return entry["mapped_final_label"]

    df["label"] = df["diagnosis"].map(resolve_label)
    diagnosis_ok = df["label"].notna()

    included = df[license_ok & diagnosis_ok].copy()
    excluded_diagnosis = int((~diagnosis_ok).sum())

    included["image_path"] = included["isic_id"].map(lambda iid: str(images_dir / f"{iid}.jpg"))
    missing_files = [p for p in included["image_path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} included rows reference missing image files; "
            f"first few: {missing_files[:5]}"
        )

    out = included.rename(columns={"isic_id": "image_id"})[["image_id", "image_path", "label"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"total rows in metadata: {total_rows}")
    print(f"excluded (license not in {sorted(accepted_licenses)}): {excluded_license}")
    print(f"excluded (diagnosis unmapped or include_primary_evaluation=False): {excluded_diagnosis}")
    print(f"included rows written: {len(out)}")
    print()
    print("label distribution:")
    print(out["label"].value_counts().to_string())
    print()
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
