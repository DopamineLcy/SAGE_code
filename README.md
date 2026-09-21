# SAGE

Code for reviewing the manuscript *Image-grounded guidance narrows reliability gaps in medical vision-language models*.

## Environment

Python 3.10 and CUDA-enabled PyTorch are recommended. Install the required
packages with:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root. The model and image paths below may
be regular files/directories or symbolic links to existing downloads.

## Trained Weights

Download the Lingshu models:

- [Lingshu-7B](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B)
- [Lingshu-32B](https://huggingface.co/lingshu-medical-mllm/Lingshu-32B)

Keep the Lingshu models at the following paths:

```text
LVLM/Lingshu-7B/
LVLM/Lingshu-32B/
```

Download the external weights:

- [Rad-DINO MAIRA-2](https://huggingface.co/microsoft/rad-dino-maira-2)
- [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized)
- [nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small)

Keep the external weights at the following paths:

```text
external/rad-dino-maira-2/
external/BiomedVLP-CXR-BERT-specialized/
external/nli-deberta-v3-small/
```

Finally, download the SAGE-guider [checkpoint](https://drive.google.com/file/d/16cveHZmbLGxdcFprzpDJ7iN80Yt1rN-4/view?usp=drive_link) and save it as:

```text
trained_models/sage_guider_checkpoint.pth
```

## Examples

The full benchmark will be released after manuscript acceptance. This review
repository provides examples to reproduce case studies:

- Lingshu-7B: `12736592_51566590`
- Lingshu-32B: `19454978_57475408`

Obtain the two radiographs from
[MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
after completing its credentialing and data-use requirements, then save them
as:

```text
data/examples/images/12736592_51566590.jpg
data/examples/images/19454978_57475408.jpg
```

The corresponding JSON annotations are included at:

```text
data/chexdevicebench/12736592_51566590.json
data/chexdevicebench/19454978_57475408.json
```

## Reproduce SAGE Directly

### Lingshu-7B

```bash
export CUDA_VISIBLE_DEVICES=0

bash scripts/run_sage_inference.sh \
  --input-csv data/examples/lingshu7b_example.csv \
  --image-root data/examples/images \
  --labels-json data/chexdevicebench/device_list.json \
  --guider-checkpoint trained_models/sage_guider_checkpoint.pth \
  --external-model-root external \
  --lingshu-model LVLM/Lingshu-7B \
  --model-size 7b \
  --output-dir outputs/lingshu7b
```

The final prediction is `outputs/lingshu7b/sage_predictions.csv`.

### Lingshu-32B

```bash
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/run_sage_inference.sh \
  --input-csv data/examples/lingshu32b_example.csv \
  --image-root data/examples/images \
  --labels-json data/chexdevicebench/device_list.json \
  --guider-checkpoint trained_models/sage_guider_checkpoint.pth \
  --external-model-root external \
  --lingshu-model LVLM/Lingshu-32B \
  --model-size 32b \
  --output-dir outputs/lingshu32b
```

The final prediction is `outputs/lingshu32b/sage_predictions.csv`.

## Reproduce SAGE-guider training

Download [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
and its paired [MIMIC-CXR reports](https://physionet.org/content/mimic-cxr/2.1.0/).
Keep all views and build the train/validate table and chronological study list:

```bash
python scripts/prepare_training_data.py \
  --split-csv /path/to/mimic-cxr-2.1.0-split.csv.gz \
  --metadata-csv /path/to/mimic-cxr-2.1.0-metadata.csv.gz \
  --training-csv data/intermediate/mimic_train_validate.csv \
  --chronology-csv data/intermediate/mimic_chronology.csv
```

Extract the 12-device ontology using a locally deployed [Llama-3.3-70B-Instruct-AWQ model](https://huggingface.co/kosbu/Llama-3.3-70B-Instruct-AWQ):

```bash
python scripts/extract_ontology.py \
  --prompt-file data/ontology/prompt.txt \
  --chronology-csv data/intermediate/mimic_chronology.csv \
  --report-root /path/to/mimic-cxr/2.1.0/files \
  --output-dir data/intermediate/ontology_json \
  --api-base http://localhost:8000/v1 \
  --model MODEL_NAME
```

Start training:

```bash
export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train_sage_guider.sh \
  --image-root /path/to/mimic-cxr-jpg/2.1.0/files \
  --metadata-csv data/intermediate/mimic_train_validate.csv \
  --report-root data/intermediate/ontology_json \
  --labels-json data/chexdevicebench/device_list.json \
  --external-model-root external \
  --output-dir outputs/sage_guider
```

## Acknowledgements

Some codes are borrowed from the amazing projects: [RadZero](https://github.com/deepnoid-ai/RadZero), [CARZero](https://github.com/laihaoran/CARZero/), and [MARINE](https://github.com/Linxi-ZHAO/MARINE).
