import json
import os
import random
import torch
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Distributed sampler with weighted sampling to oversample
    images containing positive entities."""

    def __init__(self, weights, total_size, num_replicas=None, rank=None, seed=0):
        import torch.distributed as dist
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.total_size = total_size - (total_size % num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class PretrainDataset(Dataset):
    """MIMIC-CXR-JPG dataset paired with per-study SAGE ontology JSON.

    ``labels_json`` is intentionally read in order.  Reordering labels changes
    the semantic meaning of an existing SAGE-guider checkpoint.
    """

    def __init__(self, is_train: bool = True, args: Any = None) -> None:
        super().__init__()
        self.is_train = is_train
        self.args = args
        self.image_root = args.image_root
        self.report_root = args.report_root

        df = pd.read_csv(args.metadata_csv).astype(
            {'dicom_id': str, 'study_id': str, 'subject_id': str, 'split': str}
        )

        target_split = 'train' if self.is_train else 'validate'
        self.df = df[df['split'] == target_split].reset_index(drop=True)

        with open(args.labels_json, "r", encoding="utf-8") as f:
            self.cols: List[str] = json.load(f)
        if not isinstance(self.cols, list) or len(self.cols) != 12 or not all(isinstance(x, str) for x in self.cols):
            raise ValueError("--labels-json must be an ordered list of exactly 12 device names")

        self.num_entities = len(self.cols)
        
        self.dicom_ids = self.df["dicom_id"].astype(str).values
        self.subject_ids = self.df["subject_id"].astype(int).values
        self.study_ids = self.df["study_id"].astype(int).values

        model_name = os.path.join(args.external_model_root, "rad-dino-maira-2")
        self.rad_dino_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        text_model_path = os.path.join(args.external_model_root, "BiomedVLP-CXR-BERT-specialized")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)

        self.presence_map = {"Yes": 1, "No": 0, "Not Mentioned": 0, "Uncertain": -100}

        print("num_entities: ", self.num_entities)

        self._precompute_positive_counts()

        if self.is_train:
            # Augmentation parameters from args or default
            degrees = getattr(self.args, 'aug_degrees', None)
            scale = getattr(self.args, 'aug_scale', None)

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((518, 518), scale=scale, ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),

                # Scaling is already handled by RandomResizedCrop.
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=degrees, interpolation=InterpolationMode.BICUBIC),
                ], p=self.args.aug_prob),

                transforms.RandomApply([
                    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                ], p=self.args.aug_prob),
            ])
        else:
            self.transform = None

    def _precompute_positive_counts(self):
        """Pre-scan all reports to count positive entities per sample for weighted sampling.
        Results are cached to a .npy file so subsequent runs load instantly."""
        import torch.distributed as dist

        split_tag = "train" if self.is_train else "valid"
        cache_path = os.path.join(self.report_root, f"_positive_counts_{split_tag}.npy")
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            if len(cached) == len(self.df):
                self.num_positives = cached
                print(f"Loaded positive counts from cache: {cache_path}")
                self._print_positive_stats()
                return
            print(f"Cache size mismatch ({len(cached)} vs {len(self.df)}), recomputing...")

        # In DDP, only rank 0 computes cache to avoid duplicated heavy I/O.
        if is_distributed and rank != 0:
            print(f"Rank {rank}: waiting for rank 0 to build positive-count cache...")
            dist.barrier()
            if not os.path.exists(cache_path):
                raise RuntimeError(f"Rank 0 did not produce cache file: {cache_path}")
            cached = np.load(cache_path)
            if len(cached) != len(self.df):
                raise RuntimeError(
                    f"Cache length mismatch after barrier: {len(cached)} vs {len(self.df)}"
                )
            self.num_positives = cached
            print(f"Rank {rank}: loaded positive counts from cache: {cache_path}")
            self._print_positive_stats()
            return

        print("Pre-computing positive entity counts (first run, will be cached)...")
        self.num_positives = np.zeros(len(self.df), dtype=np.int32)
        report_cache: Dict[str, int] = {}

        split_name = "Train" if self.is_train else "Valid"
        for i in tqdm(range(len(self.df)), desc=f"[{split_name}] Scanning reports", disable=(rank != 0)):
            _, report_path = self._build_paths(
                self.dicom_ids[i], int(self.subject_ids[i]), int(self.study_ids[i])
            )
            if report_path in report_cache:
                self.num_positives[i] = report_cache[report_path]
                continue

            report = self._load_report(report_path)
            count = 0
            for entity_name in self.cols:
                info = report.get(entity_name, {})
                presence_str = info.get("presence", "Not Mentioned")
                if self.presence_map.get(presence_str, -100) == 1:
                    count += 1
            self.num_positives[i] = count
            report_cache[report_path] = count

        try:
            np.save(cache_path, self.num_positives)
            print(f"Saved positive counts cache to: {cache_path}")
        except OSError as e:
            print(f"Warning: could not save cache ({e}), will recompute next time")

        if is_distributed:
            dist.barrier()

        self._print_positive_stats()

    def _print_positive_stats(self):
        n_with_pos = int((self.num_positives > 0).sum())
        n_total = len(self.df)
        split_name = "Train" if self.is_train else "Valid"
        print(f"  [{split_name}] Samples with >=1 positive entity: "
              f"{n_with_pos}/{n_total} ({100 * n_with_pos / n_total:.1f}%)")
        print(f"  [{split_name}] Samples with 0 positive entities: "
              f"{n_total - n_with_pos}/{n_total} ({100 * (n_total - n_with_pos) / n_total:.1f}%)")
        print(f"  [{split_name}] Average positives per sample: {self.num_positives.mean():.2f}")

    def get_train_sampler(self, num_replicas=None, rank=None, seed=0, pos_sample_weight=4.0):
        """Create a DistributedWeightedSampler that oversamples positive-containing images.

        Args:
            pos_sample_weight: weight for samples with >=1 positive entity.
                               Samples with 0 positives always have weight 1.
        """
        weights = np.where(self.num_positives > 0, pos_sample_weight, 1.0)
        return DistributedWeightedSampler(
            weights=weights,
            total_size=len(self.df),
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.df)
    
    def _build_paths(self, dicom_id: str, subject_id: int, study_id: int) -> Tuple[str, str]:
        sid_str = str(subject_id)
        stid_str = str(study_id)
        prefix = sid_str[:2]
        rel_dir = os.path.join(f"p{prefix}", f"p{sid_str}", f"s{stid_str}")
        rel_dir_no_study = os.path.join(f"p{prefix}", f"p{sid_str}")
        image_path = os.path.join(self.image_root, rel_dir, f"{dicom_id}.jpg")
        report_path = os.path.join(self.report_root, rel_dir_no_study, f"s{stid_str}.json")
        return image_path, report_path

    def _load_report(self, report_path: str) -> Dict[str, Any]:
        if not os.path.exists(report_path): return {}
        with open(report_path, "r", encoding="utf-8") as f: return json.load(f)

    def _toggle_no_in_statement(self, device: str, val: int) -> str:
        """
        Counterfactual text transform:
        - val=1 (real positive)  -> counterfactual negative: "There is no ..."
        - val=0 (real negative)  -> counterfactual positive: "There is ..."
        """
        device_text = str(device or "").strip()
        if not device_text:
            return device_text

        sampled_device = device_text if random.random() < 0.5 else device_text.lower()

        if val == 1:
            return f"There is no {sampled_device}"
        elif val == 0:
            return f"There is {sampled_device}"
        return sampled_device

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dicom_id = self.dicom_ids[idx]
        subject_id = self.subject_ids[idx]
        study_id = self.study_ids[idx]

        image_path, report_path = self._build_paths(dicom_id, subject_id, study_id)
        
        if os.path.exists(image_path):
            image = pil_loader(image_path)
            if self.is_train and self.transform is not None and self.args.is_augmentation:
                image = self.transform(image)
                
            processed = self.rad_dino_processor(image, return_tensors="pt")
            processed_image = processed['pixel_values'].squeeze(0)
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

        report = self._load_report(report_path)

        return {
            "image": processed_image,
            "report_path": report_path,
            "report_dict": report
        }

    def _get_attribute_str(self, info: Dict[str, Any]) -> str:
        """
        Extract and combine location and attributes into a single string.
        """
        loc = info.get("location", "")
        if loc is None: loc = ""
        if str(loc) == "NA": loc = "uncertain"
        
        attrs = info.get("attributes", [])
        if attrs == "NA":
            attrs = []
        if attrs is None: attrs = []
        if isinstance(attrs, str): attrs = [attrs]
        
        # Filter out None or empty strings from attrs
        attrs = [str(a) for a in attrs if a]
        
        parts = []
        if loc: parts.append(f"location is {loc}.")
        if attrs: parts.append(f"Attributes are {', '.join(attrs)}.")
        
        return " ".join(parts).strip()

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        
        report_paths_batch = []
        
        # We need to collect flattened lists
        flat_report_texts = []
        flat_text_is_pos = []
        flat_sampled_entity_ids = []
        flat_src_indices = []
        flat_text_attributes = [] # [Total_Text_Samples]
        
        presence_values_list = []
        batch_image_attributes_list = [] # [B, 12] strings
        
        batch_images_with_positive_sampled = 0
        
        K = 4

        for b_idx, b in enumerate(batch):
            report_path = b["report_path"]
            report_paths_batch.append(report_path)
            report_dict = b["report_dict"]
            current_presence_values = []
            current_image_attributes = [] # [12]
            pos_indices = []
            neg_indices = []
            
            for idx, entity_name in enumerate(self.cols):
                info = report_dict.get(entity_name, {})
                presence_str = info.get("presence", "error")
                assert presence_str != "error", f"Error in report_dict: {report_dict}"
                # Extract attributes for this entity in this image
                attr_str = self._get_attribute_str(info)
                current_image_attributes.append(attr_str)

                val = self.presence_map.get(presence_str, -100)
                
                current_presence_values.append(val)

                if val == 1:
                    pos_indices.append(idx)
                elif val == 0:
                    neg_indices.append(idx)

            presence_values_list.append(torch.tensor(current_presence_values, dtype=torch.long))
            batch_image_attributes_list.append(current_image_attributes)

            current_sampled_indices = []
            random.shuffle(pos_indices)
            current_sampled_indices.extend(pos_indices[:K])
            
            actual_K_pos = len(current_sampled_indices)
            actual_K_neg =  K - actual_K_pos
            # print("actual_K_pos: ", actual_K_pos, "actual_K_neg: ", actual_K_neg)
            random.shuffle(neg_indices)
            current_sampled_indices.extend(neg_indices[:actual_K_neg])

            assert len(current_sampled_indices) == K, f"len(current_sampled_indices) != K: {len(current_sampled_indices)} != {K}"

            if any(current_presence_values[idx] == 1 for idx in current_sampled_indices):
                batch_images_with_positive_sampled += 1
        
            for entity_idx in current_sampled_indices:
                entity_name = self.cols[entity_idx]
                
                is_presence_yes = current_presence_values[entity_idx]
                
                info = report_dict.get(entity_name, {})

                current_image_entity_attributes = current_image_attributes[entity_idx]
                presence_statement = str(info.get("presence_statement", "") or "").strip()
                entity_text = entity_name if random.random() < 0.5 else entity_name.lower()
                there_is_statement = (
                    f"There is {entity_text}" if is_presence_yes == 1 else f"There is no {entity_text}"
                )

                if random.random() < 0.5:
                    text_content = presence_statement if presence_statement else there_is_statement
                else:
                    text_content = there_is_statement

                flat_report_texts.append(text_content)
                flat_text_is_pos.append(is_presence_yes)
                flat_sampled_entity_ids.append(entity_idx)
                flat_src_indices.append(b_idx)
                
                # Get attribute string for this text sample (derived from the source image's entity info)
                # Note: We use the SAME attributes as the image it came from
                flat_text_attributes.append(current_image_entity_attributes)

                if getattr(self.args, "use_counterfactual", False) and is_presence_yes in (0, 1):
                    cf_is_pos = 1 - is_presence_yes
                    cf_text = self._toggle_no_in_statement(entity_name, val=is_presence_yes)
                    flat_report_texts.append(cf_text)
                    flat_text_is_pos.append(cf_is_pos)
                    flat_sampled_entity_ids.append(entity_idx)
                    flat_src_indices.append(b_idx)
                    flat_text_attributes.append("") # CF text has no real attributes

        presence_values = torch.stack(presence_values_list, dim=0)
        
        # Tokenize (CPU side)
        tokenized = self.tokenizer(
            flat_report_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )

        sampled_image_has_positive_ratio = batch_images_with_positive_sampled / max(len(batch), 1)

        return {
            "images": images,
            "report_paths": report_paths_batch,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "text_is_pos": torch.tensor(flat_text_is_pos, dtype=torch.long),
            "text_entity_ids": torch.tensor(flat_sampled_entity_ids, dtype=torch.long),
            "text_src_indices": torch.tensor(flat_src_indices, dtype=torch.long),
            "text_attributes": flat_text_attributes, # List[str] length N
            "image_attributes": batch_image_attributes_list, # List[List[str]] [B, 12]
            "presence_values": presence_values,
            "sampled_image_has_positive_ratio": sampled_image_has_positive_ratio
        }
