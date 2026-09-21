"""Shared CheXDevice schema and validation helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from . import CSV_COLUMNS


SIDE_ONLY_DEVICES = {"PICC", "Chest tube"}
SIDE_AND_VIA_DEVICES = {
    "Conventional central venous catheter",
    "Hemodialysis catheter",
    "Port-A-Cath",
    "Pulmonary artery flotation catheter",
}
SIDES = ("Left", "Right")
VIA_ROUTES = (
    "Internal Jugular Vein",
    "Subclavian Vein",
    "Inferior Vena Cava",
)

_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def supports_side(device_name: str) -> bool:
    return device_name in SIDE_ONLY_DEVICES or device_name in SIDE_AND_VIA_DEVICES


def supports_via(device_name: str) -> bool:
    return device_name in SIDE_AND_VIA_DEVICES


def load_device_names(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    names = payload.get("device_list") if isinstance(payload, dict) else payload
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{path} must contain an ordered non-empty device-name list")
    if len(names) != 12 or len(set(names)) != len(names):
        raise ValueError(f"{path} must contain exactly 12 unique device names in checkpoint order")
    return names


def resolve_image_path(image_root: str | Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else Path(image_root) / path


def load_input_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"ImageID", "ImagePath"} - fields
        if missing:
            raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        image_id = row.get("ImageID", "")
        image_path = row.get("ImagePath", "")
        if not image_id or not image_path:
            raise ValueError(f"Input CSV row {number} has an empty ImageID or ImagePath")
        if image_id in seen:
            raise ValueError(f"Duplicate ImageID in input CSV: {image_id}")
        seen.add(image_id)
    return rows


def strip_json_comments(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def extract_json_payload(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Prediction is not text")
    matches = _JSON_BLOCK.findall(text)
    raw = matches[-1] if matches else text.strip()
    for candidate in (raw, strip_json_comments(raw)):
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("Prediction JSON must be an object")
            return payload
        except json.JSONDecodeError:
            pass
    raise ValueError("Prediction does not contain valid JSON")


def extract_devices(text: str) -> list[dict[str, Any]]:
    devices = extract_json_payload(text).get("devices")
    if not isinstance(devices, list):
        raise ValueError("Prediction JSON must contain a devices list")
    if not all(isinstance(device, dict) for device in devices):
        raise ValueError("Every devices entry must be an object")
    return devices


def normalize_attributes(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Device attributes must be an object")
    unknown = set(value) - {"via", "side"}
    if unknown:
        raise ValueError(f"Unsupported device attribute keys: {', '.join(sorted(unknown))}")
    return {"via": value.get("via"), "side": value.get("side")}


def valid_bbox(value: Any, *, allow_null: bool = True) -> list[float] | None:
    if value is None and allow_null:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox must be null or a four-element list")
    try:
        coords = [float(coord) for coord in value]
    except (TypeError, ValueError) as error:
        raise ValueError("bbox coordinates must be numeric") from error
    if any(coord < 0.0 or coord > 1.0 for coord in coords):
        raise ValueError("bbox coordinates must be in [0, 1]")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")
    return coords


def format_prediction(devices: Iterable[dict[str, Any]]) -> str:
    payload = {"devices": list(devices)}
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=4) + "\n```"


def read_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path} must have exactly these columns in order: {CSV_COLUMNS}")
        rows = list(reader)
    seen: set[str] = set()
    for row in rows:
        image_id = row["ImageID"]
        if not image_id:
            raise ValueError(f"{path} contains an empty ImageID")
        if image_id in seen:
            raise ValueError(f"{path} contains duplicate ImageID {image_id}")
        seen.add(image_id)
    return rows
