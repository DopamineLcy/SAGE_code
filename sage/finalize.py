"""Strictly merge Stage-2 bboxes into the immutable Stage-1 device plan."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import CSV_COLUMNS
from .schema import (
    extract_devices,
    format_prediction,
    normalize_attributes,
    read_prediction_csv,
    valid_bbox,
)


def merge_locked_devices(
    stage1_devices: list[dict[str, Any]],
    stage2_devices: list[dict[str, Any]],
    image_id: str,
) -> list[dict[str, Any]]:
    if len(stage1_devices) != len(stage2_devices):
        raise ValueError(
            f"{image_id}: Stage-2 device count {len(stage2_devices)} does not match "
            f"Stage-1 count {len(stage1_devices)}"
        )
    merged = deepcopy(stage1_devices)
    for index, (fixed, refined) in enumerate(zip(stage1_devices, stage2_devices)):
        fixed_name = fixed.get("name")
        refined_name = refined.get("name")
        if not isinstance(fixed_name, str) or not fixed_name:
            raise ValueError(f"{image_id}: Stage-1 device {index} has no valid name")
        if refined_name != fixed_name:
            raise ValueError(
                f"{image_id}: Stage-2 device {index} name {refined_name!r} does not match "
                f"Stage-1 name {fixed_name!r}"
            )
        fixed_attributes = normalize_attributes(fixed.get("attributes"))
        refined_attributes = normalize_attributes(refined.get("attributes"))
        if refined_attributes != fixed_attributes:
            raise ValueError(f"{image_id}: Stage-2 device {index} changed side/via attributes")
        merged[index]["attributes"] = fixed_attributes
        merged[index]["bbox"] = valid_bbox(refined.get("bbox"), allow_null=True)
    return merged


def finalize(stage1_csv: str, stage2_csv: str, output_csv: str) -> None:
    stage1_rows = read_prediction_csv(stage1_csv)
    stage2_rows = read_prediction_csv(stage2_csv)
    stage1_ids = [row["ImageID"] for row in stage1_rows]
    stage2_lookup = {row["ImageID"]: row for row in stage2_rows}
    if set(stage1_ids) != set(stage2_lookup):
        missing = sorted(set(stage1_ids) - set(stage2_lookup))
        extra = sorted(set(stage2_lookup) - set(stage1_ids))
        raise ValueError(
            "Stage-1 and Stage-2 row sets differ; "
            f"missing Stage-2={missing[:3]}, extra Stage-2={extra[:3]}"
        )

    output_rows: list[dict[str, str]] = []
    for stage1_row in stage1_rows:
        image_id = stage1_row["ImageID"]
        stage2_row = stage2_lookup[image_id]
        if stage1_row["Status"] != "Success":
            raise ValueError(f"{image_id}: Stage-1 status is not Success")
        if stage2_row["Status"] != "Success":
            raise ValueError(f"{image_id}: Stage-2 status is not Success")
        if stage2_row["ImagePath"] != stage1_row["ImagePath"]:
            raise ValueError(f"{image_id}: Stage-2 ImagePath differs from Stage-1")
        devices = merge_locked_devices(
            extract_devices(stage1_row["PredictedAnswer"]),
            extract_devices(stage2_row["PredictedAnswer"]),
            image_id,
        )
        output_rows.append(
            {
                "ImageID": image_id,
                "ImagePath": stage1_row["ImagePath"],
                "Prompt": stage2_row["Prompt"],
                "PredictedAnswer": format_prediction(devices),
                "Status": "Success",
                "Error": "",
            }
        )

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} strictly finalized predictions to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-csv", required=True)
    parser.add_argument("--stage2-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    finalize(arguments.stage1_csv, arguments.stage2_csv, arguments.output_csv)
