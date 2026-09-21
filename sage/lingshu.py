"""Generate CheXDevice predictions with one SAGE prompt and one Lingshu forward path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from . import CSV_COLUMNS
from .schema import resolve_image_path


def decode_generated_tokens(
    model: Any,
    tokenizer: Any,
    model_inputs: dict[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """Call model.generate once and decode tokens after the padded input length."""
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    input_length = model_inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)


def load_model(model_path: str) -> tuple[Any, Any]:
    import importlib.util
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    attention_backend = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
        device_map="auto",
    )
    return model, AutoProcessor.from_pretrained(model_path)


def _batches(items: list[Any], size: int) -> Any:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _prepare_inputs(processor: Any, prompts: list[str], image_paths: list[Path]) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for prompt, image_path in zip(prompts, image_paths)
    ]
    texts = [
        processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def run(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import set_seed

    questions = json.loads(Path(args.questions_json).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Questions file must be a JSON list")
    ids = [item.get("id") for item in questions]
    if any(not isinstance(image_id, str) or not image_id for image_id in ids):
        raise ValueError("Every question must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("Questions contain duplicate ids")
    resolved = [resolve_image_path(args.image_root, item["image_path"]) for item in questions]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")

    set_seed(args.seed)
    model, processor = load_model(args.model_path)
    tokenizer = processor.tokenizer
    rows: list[dict[str, str]] = []
    indexed = list(zip(questions, resolved))
    for batch in tqdm(list(_batches(indexed, args.batch_size)), desc="SAGE Lingshu inference"):
        batch_questions = [item[0] for item in batch]
        batch_paths = [item[1] for item in batch]
        prompts = [item["prompt"] for item in batch_questions]
        try:
            inputs = _prepare_inputs(processor, prompts, batch_paths).to(model.device)
            with torch.inference_mode():
                outputs = decode_generated_tokens(model, tokenizer, inputs, args.max_new_tokens)
            for item, output in zip(batch_questions, outputs):
                rows.append(
                    {
                        "ImageID": item["id"],
                        "ImagePath": item["image_path"],
                        "Prompt": item["prompt"],
                        "PredictedAnswer": output.strip(),
                        "Status": "Success",
                        "Error": "",
                    }
                )
        except Exception as error:
            if len(batch) == 1:
                item = batch_questions[0]
                rows.append(
                    {
                        "ImageID": item["id"],
                        "ImagePath": item["image_path"],
                        "Prompt": item["prompt"],
                        "PredictedAnswer": "",
                        "Status": "Error",
                        "Error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                raise RuntimeError("Batch generation failed; retry with --batch-size 1") from error

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
