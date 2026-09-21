"""Build the single SAGE prompt used by each Lingshu inference stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import extract_devices, load_input_csv, normalize_attributes


def _load_guidance(path: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Guider output must be a JSON list")
    lookup: dict[str, dict[str, Any]] = {}
    for entry in payload:
        image_id = entry.get("image_id") if isinstance(entry, dict) else None
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Every guider entry must have a non-empty image_id")
        if image_id in lookup:
            raise ValueError(f"Duplicate guider image_id: {image_id}")
        lookup[image_id] = entry
    return lookup


def _rounded_guidance(entry: dict[str, Any], precision: int, dense: bool) -> dict[str, Any]:
    devices = []
    for raw in entry.get("structured_devices", []):
        device = {
            "name": raw["name"],
            "presence_score": round(float(raw["presence_score"]), precision),
        }
        for key in ("side_candidates", "via_candidates"):
            if raw.get(key):
                device[key] = [
                    {"value": item["value"], "score": round(float(item["score"]), precision)}
                    for item in raw[key]
                ]
        if dense:
            device["localization_hints"] = [
                {
                    "bbox": [round(float(value), precision) for value in hint["bbox"]],
                    "peak": [round(float(value), precision) for value in hint["peak"]],
                    "score": round(float(hint["score"]), precision),
                }
                for hint in raw.get("localization_hints", [])
            ]
        devices.append(device)
    return {"devices": devices}


STAGE1_TEMPLATES = (
    "Structured specialist guider evidence JSON: <OBJECT_LIST>.\n"
    "presence_score is device evidence; side_candidates and via_candidates are attribute candidates that passed the same threshold.\n"
    "Use this grounded evidence while completing the original task below; do not treat multiple attribute candidates as final answers.\n"
    "If multiple high-score side or via candidates appear for the same device type, carefully verify from the image whether multiple devices of that type with different attributes are truly present before reporting them.\n"
    "<QUERY>",
    "The specialist chest X-ray guider provides structured evidence: <OBJECT_LIST>.\n"
    "Use presence_score for device evidence and use side_candidates/via_candidates only as candidate attributes that passed the same threshold.\n"
    "If multiple high-score side or via candidates appear for the same device type, carefully verify from the image whether multiple devices of that type with different attributes are truly present before reporting them.\n"
    "Rely on this evidence while answering the original task below.\n"
    "<QUERY>",
    "Presence-scored structured device evidence: <OBJECT_LIST>.\n"
    "Higher scores indicate stronger grounded evidence. Attribute candidates are not conflict-resolved.\n"
    "If multiple high-score side or via candidates appear for the same device type, carefully verify from the image whether multiple devices of that type with different attributes are truly present before reporting them.\n"
    "Use this evidence and avoid inventing unsupported devices.\n"
    "<QUERY>",
    "Grounded structured device candidates from the specialist guider: <OBJECT_LIST>.\n"
    "Each listed attribute candidate passed the same threshold as device presence; choose final attributes from image evidence and output null when unsupported.\n"
    "If multiple high-score side or via candidates appear for the same device type, carefully verify from the image whether multiple devices of that type with different attributes are truly present before reporting them.\n"
    "<QUERY>",
)

STAGE2_TEMPLATES = (
    "Second-stage grounding refinement evidence JSON: <OBJECT_LIST>.\n"
    "fixed_devices is the authoritative Stage-1 device plan; preserve exactly the same device count, order, names, side, and via values.\n"
    "localization_hints_by_device contains coarse, noisy CLIP-derived proposals for localization only; none of their bbox values is a final answer.\n"
    "Never copy any single hint bbox verbatim, and never choose the largest or widest hint bbox just because it covers more area.\n"
    "For each device, inspect the image and jointly consider every provided peak point together with its corresponding bbox. Use the agreement among the peak points and proposals to locate the device, then infer a new, accurate bbox from the visible device boundaries.\n"
    "Reconcile overlapping or fragmented hints, remove excess proposal padding, include all visible parts that belong to the same device when required, and exclude unrelated structures or other devices.\n"
    "Use the hints only to fill bbox fields in the fixed plan; do not add, remove, duplicate, rename, or change attributes.\n"
    "<QUERY>",
    "The specialist chest X-ray guider provides second-stage localization evidence: <OBJECT_LIST>.\n"
    "The Stage-1 fixed_devices list is locked. Return exactly the same list with bbox fields inferred from the image.\n"
    "The localization_hints_by_device entries are coarse candidate regions, not final boxes. Do not directly copy any hint bbox, especially not the largest one.\n"
    "For every fixed device, use all supplied peak points and their corresponding bboxes together: peak points indicate likely evidence locations, while bboxes may be padded, incomplete, overlapping, or fragmented. Synthesize one accurate bbox from their combined evidence and the actual visible device boundaries.\n"
    "Choose the tightest box that covers the relevant visible parts of that device, while excluding unrelated anatomy, other devices, and unsupported padding.\n"
    "Do not use localization hints to change presence, side, via, or instance count; several hints for one device are alternative proposals/fragments.\n"
    "<QUERY>",
    "Dense grounding evidence for a locked Stage-1 device plan: <OBJECT_LIST>.\n"
    "fixed_devices defines the only devices that may appear in the answer. The listed peak points and bboxes are coarse, noisy guidance for bbox refinement, not output coordinates.\n"
    "Do not copy a guidance bbox directly or default to the largest guidance bbox. For each device, inspect the image and combine all peak points with all corresponding bboxes to estimate the device center, extent, and true visible boundaries.\n"
    "Resolve overlap and fragmentation across hints, discard irrelevant regions and excess padding, and output a newly reasoned, tight bbox containing the complete relevant device evidence.\n"
    "Preserve all names and attributes exactly; only infer bbox coordinates from the image and the combined guidance.\n"
    "<QUERY>",
    "Grounded localization candidates for fixed CheXDevice predictions: <OBJECT_LIST>.\n"
    "Use fixed_devices as an immutable plan. localization_hints_by_device contains coarse candidate regions and peak points, not final bbox annotations.\n"
    "It is mandatory to reason over all peak points and their corresponding bboxes together. Never reproduce any one proposed bbox exactly, and never use the biggest proposal as a shortcut. Infer the most accurate box by matching the combined guidance to the visible device in the image.\n"
    "Treat hints as possibly padded, overlapping, or fragmented; consolidate them as appropriate, include all relevant parts of the same device, and exclude unrelated content.\n"
    "Output exactly the fixed devices in order, with a newly inferred bbox when localizable; do not create unsupported devices or change attributes.\n"
    "<QUERY>",
)


def _apply_template(template: str, evidence: dict[str, Any], query: str) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=True, separators=(",", ":"))
    return template.replace("<OBJECT_LIST>", evidence_json).replace("<QUERY>", query)


def build_stage1_prompt(
    base_prompt: str,
    evidence: dict[str, Any],
    template_index: int = 0,
) -> str:
    return _apply_template(STAGE1_TEMPLATES[template_index], evidence, base_prompt)


def _fixed_plan(prediction: str) -> list[dict[str, Any]]:
    plan = []
    for index, device in enumerate(extract_devices(prediction)):
        name = device.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Stage-1 device {index} has no valid name")
        plan.append(
            {
                "name": name,
                "attributes": normalize_attributes(device.get("attributes")),
                "bbox": None,
            }
        )
    return plan


def build_stage2_prompt(
    base_prompt: str,
    fixed_devices: list[dict[str, Any]],
    evidence: dict[str, Any],
    template_index: int = 0,
) -> str:
    fixed_names = {device["name"] for device in fixed_devices}
    hints = [device for device in evidence["devices"] if device["name"] in fixed_names]
    dense_payload = {
        "fixed_devices": fixed_devices,
        "localization_hints_by_device": [
            {"name": item["name"], "localization_hints": item.get("localization_hints", [])}
            for item in hints
        ],
    }
    plan_json = json.dumps({"devices": fixed_devices}, ensure_ascii=True, indent=2)
    query = (
        f"{base_prompt}\n\n"
        "Second-stage grounding refinement instruction:\n"
        "The Stage-1 device plan below is fixed and authoritative. Do not add, remove, duplicate, rename, reorder, or change the side/via attributes of any device.\n"
        "Your only task in this second stage is to fill the bbox field for each fixed device using the image and dense localization guidance. If a fixed device cannot be localized reliably, set its bbox to null.\n"
        "Return only one JSON code block. The devices list must have exactly the same length, order, names, and attributes as the Stage-1 plan; only bbox values may differ.\n"
        "Stage-1 fixed device plan:\n"
        f"```json\n{plan_json}\n```"
    )
    return _apply_template(STAGE2_TEMPLATES[template_index], dense_payload, query)


def _load_stage1(path: str) -> dict[str, dict[str, str]]:
    from .schema import read_prediction_csv

    rows = read_prediction_csv(path)
    failures = [row["ImageID"] for row in rows if row["Status"] != "Success"]
    if failures:
        raise ValueError(f"Stage-1 CSV contains failed rows, first: {failures[0]}")
    return {row["ImageID"]: row for row in rows}


def run(args: argparse.Namespace) -> None:
    records = load_input_csv(args.input_csv)
    guidance = _load_guidance(args.guidance_json)
    expected = {row["ImageID"] for row in records}
    if set(guidance) != expected:
        raise ValueError("Input CSV and guider output must contain exactly the same ImageID values")
    base_prompt = Path(args.base_prompt).read_text(encoding="utf-8").strip()
    stage1 = _load_stage1(args.stage1_csv) if args.stage == "stage2" else None
    if stage1 is not None and set(stage1) != expected:
        raise ValueError("Input CSV and Stage-1 CSV must contain exactly the same ImageID values")

    output = []
    template_indices = np.random.RandomState(42).randint(0, 4, len(records))
    for index, row in enumerate(records):
        image_id = row["ImageID"]
        if args.stage == "stage1":
            prompt = build_stage1_prompt(
                base_prompt,
                _rounded_guidance(guidance[image_id], args.score_precision, False),
                int(template_indices[index]),
            )
        else:
            fixed_devices = _fixed_plan(stage1[image_id]["PredictedAnswer"])
            prompt = build_stage2_prompt(
                base_prompt,
                fixed_devices,
                _rounded_guidance(guidance[image_id], args.score_precision, True),
                int(template_indices[index]),
            )
        output.append({"id": image_id, "image_path": row["ImagePath"], "prompt": prompt})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(output)} {args.stage} prompts to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage1", "stage2"))
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--guidance-json", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--stage1-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-precision", type=int, default=3)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.stage == "stage2" and not arguments.stage1_csv:
        raise SystemExit("--stage1-csv is required for stage2")
    run(arguments)
