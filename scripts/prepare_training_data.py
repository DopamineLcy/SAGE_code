#!/usr/bin/env python3
"""Build SAGE-guider training metadata from official MIMIC-CXR-JPG tables.

The image-level table deliberately keeps every view in the official train
and validate splits.  The companion chronology file is study-level and sorted
by patient, StudyDate and StudyTime for ontology extraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SPLIT_COLUMNS = {"dicom_id", "subject_id", "study_id", "split"}
REQUIRED_METADATA_COLUMNS = {"dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, help="official MIMIC-CXR split CSV")
    parser.add_argument("--metadata-csv", required=True, help="MIMIC-CXR-JPG metadata CSV")
    parser.add_argument("--training-csv", required=True, help="output all-view train/validate image table")
    parser.add_argument("--chronology-csv", required=True, help="output patient-ordered study chronology")
    return parser.parse_args()


def read_required(path: str, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"dicom_id": "string", "subject_id": "string", "study_id": "string"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def main() -> None:
    args = arguments()
    split = read_required(args.split_csv, REQUIRED_SPLIT_COLUMNS)
    metadata = read_required(args.metadata_csv, REQUIRED_METADATA_COLUMNS)

    split = split.loc[split["split"].isin(["train", "validate"]), list(REQUIRED_SPLIT_COLUMNS)].copy()
    if split.empty:
        raise ValueError("No train/validate rows found in --split-csv")
    if split.duplicated("dicom_id").any():
        raise ValueError("--split-csv has duplicate dicom_id values")
    if metadata.duplicated("dicom_id").any():
        raise ValueError("--metadata-csv has duplicate dicom_id values")

    merged = split.merge(
        metadata[["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime"]],
        on=["dicom_id", "subject_id", "study_id"], how="left", validate="one_to_one",
    )
    if merged[["StudyDate", "StudyTime"]].isna().all(axis=None):
        raise ValueError("No StudyDate/StudyTime values matched metadata; verify MIMIC table versions")
    training_rows = merged[["dicom_id", "subject_id", "study_id", "split"]].sort_values(
        ["split", "subject_id", "study_id", "dicom_id"], kind="mergesort"
    )

    chronology = merged[["subject_id", "study_id", "StudyDate", "StudyTime"]].drop_duplicates(
        ["subject_id", "study_id"]
    ).copy()
    chronology["StudyDate"] = pd.to_numeric(chronology["StudyDate"], errors="coerce").astype("Int64")
    chronology["StudyTime"] = pd.to_numeric(chronology["StudyTime"], errors="coerce").astype("Float64")
    chronology = chronology.sort_values(["subject_id", "StudyDate", "StudyTime", "study_id"], kind="mergesort")

    for output in (Path(args.training_csv), Path(args.chronology_csv)):
        output.parent.mkdir(parents=True, exist_ok=True)
    training_rows.to_csv(args.training_csv, index=False)
    chronology.to_csv(args.chronology_csv, index=False)
    print(f"Wrote {len(training_rows):,} all-view image rows: {args.training_csv}")
    print(f"Wrote {len(chronology):,} chronological study rows: {args.chronology_csv}")


if __name__ == "__main__":
    main()
