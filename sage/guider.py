"""Run the SAGE specialist guider on an input CSV."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from .schema import (
    SIDES,
    VIA_ROUTES,
    load_device_names,
    load_input_csv,
    resolve_image_path,
    supports_side,
    supports_via,
)


def build_candidates(device_names: Iterable[str]) -> list[str]:
    """Build checkpoint-ordered presence, side, and via text candidates."""
    candidates: list[str] = []
    for name in device_names:
        candidates.append(name)
        if supports_side(name):
            candidates.extend(f"{side.lower()} {name}" for side in SIDES)
        if supports_via(name):
            candidates.extend(f"{name} via {route}" for route in VIA_ROUTES)
    return candidates


def build_queries(candidate_names: Iterable[str]) -> list[str]:
    return [f"There is {name}" for name in candidate_names]


def _as_float_array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().to(torch.float32).cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=np.float32)


def similarity_vector_to_heatmap(similarity_vector: Any) -> np.ndarray:
    values = _as_float_array(similarity_vector).reshape(-1)
    grid_size = int(round(math.sqrt(values.size)))
    if grid_size * grid_size != values.size:
        raise ValueError(f"Expected a square patch map, got {values.size} patches")
    clipped = np.clip(values, -50.0, 50.0)
    heatmap = (1.0 / (1.0 + np.exp(-clipped))).reshape(grid_size, grid_size)
    low, high = float(heatmap.min()), float(heatmap.max())
    if high - low <= 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - low) / (high - low)).astype(np.float32)


def top_quantile_mask(values: np.ndarray, quantile: float) -> np.ndarray:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    flat = values.reshape(-1)
    top_count = max(1, min(int(math.ceil((1.0 - quantile) * flat.size)), flat.size))
    eligible = np.flatnonzero(flat > float(flat.min()) + 1e-8)
    if eligible.size == 0:
        eligible = np.arange(flat.size)
    top_count = min(top_count, eligible.size)
    indices = eligible[np.argsort(flat[eligible])[-top_count:]]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(values.shape)


def find_local_peaks(
    heatmap: np.ndarray,
    seed_quantile: float,
    minimum_distance: int,
) -> list[tuple[int, int, float]]:
    seed_mask = top_quantile_mask(heatmap, seed_quantile)
    height, width = heatmap.shape
    candidates: list[tuple[int, int, float]] = []
    for row in range(height):
        for column in range(width):
            if not seed_mask[row, column]:
                continue
            score = float(heatmap[row, column])
            neighborhood = heatmap[
                max(0, row - 1) : min(height, row + 2),
                max(0, column - 1) : min(width, column + 2),
            ]
            if score >= float(neighborhood.max()):
                candidates.append((row, column, score))
    candidates.sort(key=lambda item: item[2], reverse=True)

    selected: list[tuple[int, int, float]] = []
    minimum_distance_squared = float(minimum_distance**2)
    for row, column, score in candidates:
        if all(
            (row - other_row) ** 2 + (column - other_column) ** 2
            >= minimum_distance_squared
            for other_row, other_column, _ in selected
        ):
            selected.append((row, column, score))
    return selected


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_row in range(height):
        for start_column in range(width):
            if visited[start_row, start_column] or not mask[start_row, start_column]:
                continue
            visited[start_row, start_column] = True
            stack = [(start_row, start_column)]
            component: list[tuple[int, int]] = []
            while stack:
                row, column = stack.pop()
                component.append((row, column))
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        next_row = row + row_offset
                        next_column = column + column_offset
                        if (
                            (row_offset == 0 and column_offset == 0)
                            or next_row < 0
                            or next_row >= height
                            or next_column < 0
                            or next_column >= width
                            or visited[next_row, next_column]
                            or not mask[next_row, next_column]
                        ):
                            continue
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(component)
    return components


def _split_by_peak(
    component: list[tuple[int, int]],
    peaks: list[tuple[int, int, float]],
) -> list[list[tuple[int, int]]]:
    if len(peaks) == 1:
        return [component]
    peak_coordinates = np.asarray([(row, column) for row, column, _ in peaks], dtype=np.float32)
    regions: list[list[tuple[int, int]]] = [[] for _ in peaks]
    for row, column in component:
        distances = ((peak_coordinates - np.asarray([row, column])) ** 2).sum(axis=1)
        regions[int(np.argmin(distances))].append((row, column))
    return regions


def _region_to_hint(
    region: list[tuple[int, int]],
    peak: tuple[int, int, float],
    grid_height: int,
    grid_width: int,
    expansion: float,
) -> dict[str, Any]:
    rows = [row for row, _ in region]
    columns = [column for _, column in region]
    row, column, score = peak
    return {
        "bbox": [
            round(max(0.0, min(columns) / grid_width - expansion), 6),
            round(max(0.0, min(rows) / grid_height - expansion), 6),
            round(min(1.0, (max(columns) + 1) / grid_width + expansion), 6),
            round(min(1.0, (max(rows) + 1) / grid_height + expansion), 6),
        ],
        "peak": [
            round((column + 0.5) / grid_width, 6),
            round((row + 0.5) / grid_height, 6),
        ],
        "score": round(score, 6),
    }


def build_dense_hints(
    similarity_vector: Any,
    *,
    maximum_hints: int = 2,
    seed_quantile: float = 0.95,
    support_quantile: float = 0.80,
    minimum_component_area: int = 2,
    bbox_expansion: float = 0.03,
    minimum_peak_distance: int = 3,
) -> list[dict[str, Any]]:
    """Convert a square patch similarity map to multi-peak normalized bbox hints."""
    if support_quantile > seed_quantile:
        raise ValueError("support_quantile must be <= seed_quantile")
    if maximum_hints < 1 or minimum_component_area < 1:
        raise ValueError("maximum_hints and minimum_component_area must be positive")
    if bbox_expansion < 0 or minimum_peak_distance < 0:
        raise ValueError("bbox_expansion and minimum_peak_distance must be non-negative")

    heatmap = similarity_vector_to_heatmap(similarity_vector)
    if float(heatmap.max()) <= 0.0:
        return []
    peaks = find_local_peaks(heatmap, seed_quantile, minimum_peak_distance)
    peak_map = {(row, column): (row, column, score) for row, column, score in peaks}
    hints: list[dict[str, Any]] = []
    for component in connected_components(top_quantile_mask(heatmap, support_quantile)):
        component_peaks = [peak_map[coordinate] for coordinate in component if coordinate in peak_map]
        if not component_peaks:
            continue
        component_peaks.sort(key=lambda item: item[2], reverse=True)
        for peak, region in zip(component_peaks, _split_by_peak(component, component_peaks)):
            if len(region) >= minimum_component_area:
                hints.append(
                    _region_to_hint(region, peak, heatmap.shape[0], heatmap.shape[1], bbox_expansion)
                )
    hints.sort(key=lambda item: item["score"], reverse=True)
    return hints[:maximum_hints]


def _patch_attention_backend() -> None:
    from transformers import AutoModel
    from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

    if not getattr(AutoModel.from_pretrained, "_sage_compatible", False):
        original_loader = AutoModel.from_pretrained

        def compatible_loader(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "sdpa"
            return original_loader(*args, **kwargs)

        compatible_loader._sage_compatible = True  # type: ignore[attr-defined]
        AutoModel.from_pretrained = compatible_loader

    if not getattr(Dinov2Config.__init__, "_sage_compatible", False):
        original_init = Dinov2Config.__init__

        def compatible_init(self: Any, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "sdpa"
            original_init(self, *args, **kwargs)

        compatible_init._sage_compatible = True  # type: ignore[attr-defined]
        Dinov2Config.__init__ = compatible_init


def load_guider(
    checkpoint_path: str | Path,
    external_model_root: str | Path,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer

    _patch_attention_backend()
    external_root = Path(external_model_root).resolve()
    if not external_root.is_dir():
        raise FileNotFoundError(f"External model root does not exist: {external_root}")

    try:
        from .guider_model import BaseModel
    except ImportError as error:
        raise ImportError(
            "Could not import the SAGE-guider inference model. Run from the package root."
        ) from error

    arguments = SimpleNamespace(
        world_size=1,
        rank=0,
        use_vision_cls_token=True,
        proj_dim=768,
        num_hidden_layers=2,
        rad_dino_output_layer=-1,
        use_extra_pos_embed=False,
        external_model_root=str(external_root),
    )
    original_directory = Path.cwd()
    try:
        # This preserves compatibility with the training implementation that loads
        # models from ./external while newer copies read external_model_root.
        if external_root.name == "external":
            os.chdir(external_root.parent)
        model = BaseModel(args=arguments)
    finally:
        os.chdir(original_directory)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    clean_state = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if not key.startswith("criterion.nli_model."):
            clean_state[key] = value
    incompatible = model.load_state_dict(clean_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Checkpoint loaded non-strictly; "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )
    model.to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(
        external_root / "rad-dino-maira-2", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        external_root / "BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )
    return model, image_processor, tokenizer


def score_batch(
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    image_paths: list[Path],
    candidate_names: list[str],
    device: Any,
) -> tuple[Any, Any]:
    import torch
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    encoded_queries = tokenizer(
        build_queries(candidate_names),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    precision_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), precision_context:
        output = model.compute_logits(
            pixel_values=pixel_values,
            encoded_key_phrases=[encoded_queries],
        )
    probabilities = output["logits"].float().sigmoid().cpu()
    similarities = output["similarity_scores"].float().cpu()
    return probabilities, similarities


def build_guidance_entry(
    record: dict[str, str],
    probabilities: Any,
    similarities: Any,
    device_names: list[str],
    candidate_names: list[str],
    threshold: float,
    dense_options: dict[str, Any],
) -> dict[str, Any]:
    scores = {
        name: round(float(score), 6)
        for name, score in zip(candidate_names, probabilities.tolist())
    }
    candidate_indices = {name: index for index, name in enumerate(candidate_names)}
    structured_devices: list[dict[str, Any]] = []
    for name in device_names:
        if scores[name] < threshold:
            continue
        device: dict[str, Any] = {
            "name": name,
            "presence_score": scores[name],
            "localization_hints": build_dense_hints(
                similarities[candidate_indices[name]], **dense_options
            ),
        }
        if supports_side(name):
            side_candidates = []
            for side in SIDES:
                score = scores[f"{side.lower()} {name}"]
                if score >= threshold:
                    side_candidates.append({"value": side, "score": score})
            if side_candidates:
                device["side_candidates"] = side_candidates
        if supports_via(name):
            via_candidates = []
            for route in VIA_ROUTES:
                score = scores[f"{name} via {route}"]
                if score >= threshold:
                    via_candidates.append({"value": route, "score": score})
            if via_candidates:
                device["via_candidates"] = via_candidates
        structured_devices.append(device)
    return {
        "image_id": record["ImageID"],
        "image_path": record["ImagePath"],
        "threshold": threshold,
        "structured_devices": structured_devices,
        "dense_method": "multi_peak",
    }


def run(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    records = load_input_csv(args.input_csv)
    device_names = load_device_names(args.labels_json)
    candidate_names = build_candidates(device_names)
    image_paths = [resolve_image_path(args.image_root, row["ImagePath"]) for row in records]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} input images are missing; first entries: {preview}")

    device = torch.device(args.device)
    model, image_processor, tokenizer = load_guider(
        args.checkpoint, args.external_model_root, device
    )
    dense_options = {
        "maximum_hints": args.dense_max_hints,
        "seed_quantile": args.dense_seed_quantile,
        "support_quantile": args.dense_support_quantile,
        "minimum_component_area": args.dense_min_component_area,
        "bbox_expansion": args.dense_bbox_expand,
        "minimum_peak_distance": args.dense_peak_min_distance,
    }
    results: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(records), args.batch_size), desc="SAGE guider"):
        batch_records = records[start : start + args.batch_size]
        batch_paths = image_paths[start : start + args.batch_size]
        probabilities, similarities = score_batch(
            model, image_processor, tokenizer, batch_paths, candidate_names, device
        )
        for record, probability_row, similarity_row in zip(
            batch_records, probabilities, similarities
        ):
            results.append(
                build_guidance_entry(
                    record,
                    probability_row,
                    similarity_row,
                    device_names,
                    candidate_names,
                    args.threshold,
                    dense_options,
                )
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} guider records to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-model-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--dense-max-hints", type=int, default=2)
    parser.add_argument("--dense-seed-quantile", type=float, default=0.95)
    parser.add_argument("--dense-support-quantile", type=float, default=0.80)
    parser.add_argument("--dense-min-component-area", type=int, default=2)
    parser.add_argument("--dense-bbox-expand", type=float, default=0.03)
    parser.add_argument("--dense-peak-min-distance", type=int, default=3)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
