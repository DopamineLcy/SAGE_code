import argparse
import os
import csv
import json
import re
import time
import concurrent.futures
from collections import defaultdict
from typing import Any, List, Dict, Tuple, Optional, Set
from tqdm import tqdm

DEVICE_KEYS = [
    "Endotracheal tube", "Tracheostomy tube", "Hemodialysis catheter",
    "Port-A-Cath", "Pulmonary artery flotation catheter", "PICC",
    "Conventional central venous catheter", "Implantable cardiac pacemaker",
    "Implantable defibrillator", "Temporary pacemaker", "Chest tube",
    "Enteral tube",
]

_KEEP_CASE = {"PICC", "Port-A-Cath"}

COMPACT_INSTRUCTION = (
    "COMPACT OUTPUT: Only include device categories whose presence is "
    '"Yes", "No", or "Uncertain". COMPLETELY OMIT any device category '
    'with presence "Not Mentioned" from the JSON. '
    "Omitted devices will be automatically filled as Not Mentioned in "
    "post-processing.\n"
)


def _not_mentioned_entry(device: str) -> dict:
    name = device if device in _KEEP_CASE else device.lower()
    return {
        "presence": "Not Mentioned",
        "location": "NA",
        "attributes": "NA",
        "evidential_segment": "NA",
        "presence_statement": f"There is no {name}.",
    }


def fill_not_mentioned(study_data: dict) -> dict:
    """Fill device entries omitted by compact output as Not Mentioned."""
    for device in DEVICE_KEYS:
        if device not in study_data:
            study_data[device] = _not_mentioned_entry(device)
    return study_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the 12-device SAGE ontology from chronological MIMIC-CXR reports."
    )
    parser.add_argument("--prompt-file", required=True, type=str,
                        help="SAGE data/ontology/prompt.txt")
    parser.add_argument("--chronology-csv", required=True, type=str,
                        help="chronological CSV with subject_id, study_id, StudyDate and StudyTime")
    parser.add_argument("--report-root", required=True, type=str,
                        help="MIMIC report root containing pXX/p<subject>/s<study>.txt")
    parser.add_argument("--output-dir", required=True, type=str,
                        help="destination root for pXX/p<subject>/s<study>.json")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of concurrent patient-level workers")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip patients whose output JSON files all exist")
    parser.add_argument("--subject-prefix", type=str, default=None,
                        help="optional pXX prefix filter, for resumable extraction")
    parser.add_argument("--api-base", required=True, type=str,
                        help="base URL of an OpenAI-compatible chat-completions service")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key; alternatively set SAGE_ONTOLOGY_API_KEY")
    parser.add_argument("--model", required=True, type=str,
                        help="model name exposed by the compatible service")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="maximum output tokens per API request")
    parser.add_argument("--max-studies-per-batch", type=int, default=20,
                        help="maximum studies per request; longer histories are split into contextual batches")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="maximum retries for a failed API request")
    parser.add_argument("--compact-output", action="store_true", default=True,
                        help="omit Not Mentioned devices to reduce output tokens (default)")
    parser.add_argument("--no-compact-output", action="store_false", dest="compact_output",
                        help="emit all 12 device categories instead of compact output")
    parser.add_argument(
        "--subject-id-csv",
        type=str,
        default=None,
        help="optional CSV containing a subject_id column to extract",
    )
    parser.add_argument(
        "--subject-id-column",
        type=str,
        default="subject_id",
        help="subject-id CSV column name (default: subject_id)",
    )
    parser.add_argument(
        "--force-regenerate-target-subjects",
        action="store_true",
        help="overwrite selected subjects even if their output JSON already exists",
    )
    return parser.parse_args()


def load_patient_studies(csv_path: str) -> Dict[str, List[Dict]]:
    """Read the chronology CSV and group studies by patient."""
    patients: Dict[str, List[Dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients[row["subject_id"]].append({
                "study_id": row["study_id"],
                "date": row["StudyDate"],
                "time": row["StudyTime"],
            })
    return dict(patients)


def load_target_subject_ids(csv_path: str, column: str) -> Set[str]:
    """Read unique subject IDs from the selected CSV column."""
    target_subjects: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            sid = (row.get(column) or "").strip()
            if sid:
                target_subjects.add(sid)
    return target_subjects


def subject_prefix(subject_id: str) -> str:
    return f"p{subject_id}"[:3]


def report_path(input_dir: str, subject_id: str, study_id: str) -> str:
    return os.path.join(
        input_dir, subject_prefix(subject_id),
        f"p{subject_id}", f"s{study_id}.txt",
    )


def output_path(output_dir: str, subject_id: str, study_id: str) -> str:
    return os.path.join(
        output_dir, subject_prefix(subject_id),
        f"p{subject_id}", f"s{study_id}.json",
    )


def strip_think_tags(text: str) -> str:
    """Remove optional <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract a JSON object from a model response."""
    text = strip_think_tags(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    depth = 0
    start = text.find("{")
    if start < 0:
        return None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def is_per_device_json(obj: dict) -> bool:
    """Return whether an object is a single-study device result."""
    return any(k in obj for k in DEVICE_KEYS)


def build_prior_summary(last_study_id: str, last_date: str, last_result: dict) -> str:
    """Summarize the final study in the preceding batch."""
    lines = [
        f"# PRIOR DEVICE STATUS (from the most recent preceding study: "
        f"s{last_study_id}, date={last_date})",
        "Use this summary to resolve references like 'unchanged', 'stable', "
        "'no change in lines/tubes' in the first reports of this batch.",
        "",
    ]
    for device, info in last_result.items():
        if not isinstance(info, dict):
            continue
        presence = info.get("presence", "Not Mentioned")
        if presence in ("Yes", "Uncertain"):
            loc = info.get("location", "NA")
            attr = info.get("attributes", "NA")
            lines.append(f"- {device}: {presence} | location={loc} | attributes={attr}")
        else:
            lines.append(f"- {device}: {presence}")
    return "\n".join(lines)


def call_api(
    client: Any,
    model: str,
    prompt_text: str,
    target_reports: List[Tuple[str, str, str]],
    max_tokens: int,
    max_retries: int,
    prior_summary: Optional[str] = None,
    compact: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ``target_reports`` through the compatible API.

    ``prior_summary`` carries the final device state from the preceding batch.
    ``compact`` omits Not Mentioned devices to reduce output length.
    Each report is ``(study_id, date, report_content)``. Returns
    ``(raw_response, error_message)``.
    """
    n_targets = len(target_reports)
    reports_section = []
    for i, (study_id, date, content) in enumerate(target_reports):
        reports_section.append(
            f"--- Study {i + 1}/{n_targets}: study_id=s{study_id}, date={date} ---\n"
            f"{content}"
        )

    target_id_list = ", ".join(f'"s{r[0]}"' for r in target_reports)
    combined = "\n\n".join(reports_section)

    prior_block = ""
    if prior_summary:
        prior_block = (
            f"{prior_summary}\n\n"
            "NOTE: The above summary describes the device status from this patient's "
            "most recent study BEFORE this batch. Use it to resolve comparative "
            "references (e.g. 'unchanged', 'stable') in the earliest report(s) below. "
            "Do NOT produce output for the prior study.\n\n"
        )

    compact_block = COMPACT_INSTRUCTION if compact else ""

    user_content = (
        f"{prompt_text}\n\n"
        f"{prior_block}"
        f"# REPORTS FROM THE SAME PATIENT (Chronological Order, Total: {n_targets})\n\n"
        f"{combined}\n\n"
        f"# OUTPUT INSTRUCTION\n"
        f"{compact_block}"
        f"Output a JSON object with the following top-level keys: {target_id_list}.\n"
        f"Each key maps to the standard per-device extraction JSON for that study.\n"
        f"START your response with {{ and END with }}."
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced radiologist. Your task is to analyze "
                            "Chest X-ray reports and extract detailed information regarding "
                            "specific medical devices. You will receive multiple reports from "
                            "the SAME patient in chronological order. Your performance is "
                            "crucial for patient care quality."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, f"API failed after {max_retries} attempts: {last_err}"


def process_patient(
    subject_id: str,
    studies: List[Dict],
    input_dir: str,
    output_dir: str,
    prompt_text: str,
    client: Any,
    model: str,
    max_tokens: int,
    max_retries: int,
    max_batch: int,
    skip_existing: bool,
    compact: bool = True,
) -> Tuple[str, str, int]:
    """Process one patient and return ``(subject_id, status, study_count)``."""
    out_map = {
        s["study_id"]: output_path(output_dir, subject_id, s["study_id"])
        for s in studies
    }

    if skip_existing and all(os.path.exists(p) for p in out_map.values()):
        return subject_id, "skipped", len(studies)

    reports: List[Tuple[str, str, str]] = []
    for s in studies:
        rpath = report_path(input_dir, subject_id, s["study_id"])
        try:
            with open(rpath, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            content = "[Report file not found]"
        except Exception as e:
            content = f"[Error reading report: {e}]"
        reports.append((s["study_id"], s["date"], content))

    valid_reports = [r for r in reports if not r[2].startswith("[")]
    if not valid_reports:
        return subject_id, "no valid reports", len(studies)

    all_results: Dict[str, dict] = {}
    batch_errors: List[str] = []

    batches: List[List[Tuple[str, str, str]]] = []
    for i in range(0, len(reports), max_batch):
        batches.append(reports[i : i + max_batch])

    prior_summary: Optional[str] = None

    for batch_idx, target in enumerate(batches):
        raw, err = call_api(
            client, model, prompt_text,
            target, max_tokens, max_retries,
            prior_summary=prior_summary,
            compact=compact,
        )
        if err:
            batch_errors.append(f"batch {batch_idx} {err}")
            prior_summary = None
            continue

        result = extract_json_from_text(raw)
        if result is None:
            batch_errors.append(f"batch {batch_idx} JSON parse failed")
            _save_raw_error(raw, out_map, studies, tag=f"_batch{batch_idx}")
            prior_summary = None
            continue

        if len(target) == 1 and is_per_device_json(result):
            result = {f"s{target[0][0]}": result}

        if compact:
            for key in result:
                if isinstance(result[key], dict):
                    fill_not_mentioned(result[key])

        all_results.update(result)

        last_sid, last_date, _ = target[-1]
        last_key = f"s{last_sid}"
        last_data = result.get(last_key)
        if last_data and len(batches) > 1:
            prior_summary = build_prior_summary(last_sid, last_date, last_data)
        else:
            prior_summary = None

    written = 0
    missing_keys = []
    for s in studies:
        sid = s["study_id"]
        key = f"s{sid}"
        dst = out_map[sid]

        if skip_existing and os.path.exists(dst):
            written += 1
            continue

        study_data = all_results.get(key)
        if study_data is None:
            missing_keys.append(key)
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(study_data, f, indent=2)
        written += 1

    status = f"ok ({written}/{len(studies)})"
    if missing_keys:
        status += f", missing keys: {missing_keys}"
    if batch_errors:
        status += f", batch errors: {batch_errors}"
    return subject_id, status, len(studies)


def _save_raw_error(
    raw: str,
    out_map: Dict[str, str],
    studies: List[Dict],
    tag: str = "",
):
    """Preserve an unparseable raw response for failure diagnosis."""
    first_dst = out_map[studies[0]["study_id"]]
    err_dir = os.path.dirname(first_dst)
    os.makedirs(err_dir, exist_ok=True)
    err_path = os.path.join(err_dir, f"_raw_parse_error{tag}.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(raw)


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    api_key = args.api_key or os.environ.get("SAGE_ONTOLOGY_API_KEY", "")
    if not api_key:
        print("Error: API key not set. Use --api-key or set SAGE_ONTOLOGY_API_KEY.")
        return

    prompt_path = args.prompt_file
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file not found at {prompt_path}")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    csv_path = args.chronology_csv
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    patients = load_patient_studies(csv_path)

    prefix = args.subject_prefix
    filtered_by_prefix = (
        {sid: studies for sid, studies in patients.items() if f"p{sid}".startswith(prefix)}
        if prefix else patients
    )
    if not filtered_by_prefix:
        print(f"No patients matching prefix '{prefix}'. Exiting.")
        return

    target_subject_ids: Optional[Set[str]] = None
    if args.subject_id_csv:
        subject_id_csv_path = (
            args.subject_id_csv
            if os.path.isabs(args.subject_id_csv)
            else os.path.join(script_dir, args.subject_id_csv)
        )
        if not os.path.exists(subject_id_csv_path):
            print(f"Error: subject-id CSV not found at {subject_id_csv_path}")
            return
        try:
            target_subject_ids = load_target_subject_ids(
                subject_id_csv_path, args.subject_id_column
            )
        except Exception as e:
            print(f"Error: failed to load target subject IDs: {e}")
            return

        filtered = {
            sid: studies
            for sid, studies in filtered_by_prefix.items()
            if sid in target_subject_ids
        }
        if not filtered:
            print(
                f"No patients remain after applying subject-id CSV filter "
                f"for prefix '{prefix or 'all'}'. Exiting."
            )
            return
    else:
        filtered = filtered_by_prefix

    total_patients = len(filtered)
    total_studies = sum(len(s) for s in filtered.values())
    max_studies = max(len(s) for s in filtered.values())

    print(f"Prompt:          {prompt_path}")
    print(f"Chronology CSV:  {csv_path}")
    print(f"Report root:     {args.report_root}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Model:           {args.model}")
    print(f"Prefix filter:   {prefix or 'all'}")
    if args.subject_id_csv:
        print(f"Target subject CSV: {args.subject_id_csv}")
        print(f"Target ID column:   {args.subject_id_column}")
        print(
            "Force regenerate:   "
            f"{'ON' if args.force_regenerate_target_subjects else 'OFF'}"
        )
    print(f"Patients:        {total_patients}  ({total_studies} studies)")
    print(f"Max studies/pt:  {max_studies}  (batch limit: {args.max_studies_per_batch})")
    print(f"Workers:         {args.workers}")
    print(f"Max tokens:      {args.max_tokens}")
    print(f"Compact output:  {args.compact_output}")

    n_batched = sum(
        1 for s in filtered.values() if len(s) > args.max_studies_per_batch
    )
    if n_batched:
        print(f"Patients needing batching: {n_batched}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The ontology extractor requires the 'openai' package. Install it with pip install openai.") from exc
    client = OpenAI(base_url=args.api_base, api_key=api_key)

    completed = 0
    skipped = 0
    failures = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_patient,
                    sid, studies,
                    args.report_root, args.output_dir,
                    prompt_text, client, args.model,
                    args.max_tokens, args.max_retries,
                    args.max_studies_per_batch,
                    (
                        False
                        if (
                            target_subject_ids is not None
                            and args.force_regenerate_target_subjects
                            and sid in target_subject_ids
                        )
                        else args.skip_existing
                    ),
                    args.compact_output,
                ): sid
                for sid, studies in filtered.items()
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=total_patients,
                desc="Processing patients",
            ):
                sid = futures[future]
                try:
                    _, status, _ = future.result()
                except Exception as e:
                    status = f"exception: {e}"

                if status == "skipped":
                    skipped += 1
                elif status.startswith("ok"):
                    completed += 1
                else:
                    failures += 1
                    tqdm.write(f"[WARN] p{sid}: {status}")

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved.")

    print(
        f"\nDone. Patients — OK: {completed}, Skipped: {skipped}, "
        f"Failed: {failures}, Total: {total_patients}"
    )
    print(f"Total studies involved: {total_studies}")


if __name__ == "__main__":
    main()
