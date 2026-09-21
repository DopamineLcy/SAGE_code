"""Inference-only SAGE-guider network."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder


class SimilarityLogit(nn.Module):
    def forward(
        self,
        queries: torch.Tensor,
        local_tokens: torch.Tensor,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = F.normalize(queries, p=2, dim=-1)
        local_tokens = F.normalize(local_tokens, p=2, dim=-1)
        similarity = torch.einsum("nd,bld->bnl", queries, local_tokens) / temperature
        attention = F.softmax(similarity, dim=-1)
        aggregated = torch.einsum("bnl,bld->bnd", attention, local_tokens)
        logits = (F.normalize(aggregated, p=2, dim=-1) * queries.unsqueeze(0)).sum(dim=-1)
        return logits, similarity


class InferenceCriterion(nn.Module):
    def __init__(self, use_vision_cls_token: bool) -> None:
        super().__init__()
        self.use_vision_cls_token = use_vision_cls_token
        self.loss_temperature = nn.Parameter(torch.tensor([np.log(0.07)], dtype=torch.float32))
        self.attn_temperature = None
        self.similarity_logit = SimilarityLogit()


class VisionEncoder(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        model_path = str(Path(args.external_model_root) / "rad-dino-maira-2")
        backend = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.rad_dino_model = AutoModel.from_pretrained(
            model_path,
            attn_implementation=backend,
            torch_dtype=torch.bfloat16,
        )
        self.rad_dino_model.requires_grad_(False).eval()
        self.rad_dino_output_layer = args.rad_dino_output_layer
        if self.rad_dino_output_layer != -1:
            self.layer_norm = nn.LayerNorm(768)
        self.feature_dim = 768
        self.use_extra_pos_embed = args.use_extra_pos_embed
        if self.use_extra_pos_embed:
            self.extra_pos_embed = nn.Parameter(torch.zeros(1, 1370, self.feature_dim))
        config = Dinov2Config(
            hidden_size=768,
            num_hidden_layers=args.num_hidden_layers,
            use_layer_norm=False,
            attn_implementation=backend,
        )
        self.transformer_blocks = Dinov2Encoder(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.rad_dino_model(
                pixel_values=images.to(dtype=torch.bfloat16),
                output_hidden_states=self.rad_dino_output_layer != -1,
            )
            if self.rad_dino_output_layer == -1:
                features = outputs.last_hidden_state
            else:
                features = self.layer_norm(outputs.hidden_states[self.rad_dino_output_layer])
        if self.use_extra_pos_embed:
            features = features + self.extra_pos_embed
        return self.transformer_blocks(features)["last_hidden_state"]


class TextEncoder(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        model_path = str(Path(args.external_model_root) / "BiomedVLP-CXR-BERT-specialized")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        batch_size, entity_count, sequence_length = input_ids.shape
        outputs = self.model(
            input_ids=input_ids.reshape(-1, sequence_length),
            attention_mask=attention_mask.reshape(-1, sequence_length),
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 0, :].reshape(batch_size, entity_count, -1)


class BaseModel(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.vision_encoder = VisionEncoder(args)
        self.text_encoder = TextEncoder(args)
        self.vision_proj = nn.Linear(768, args.proj_dim, bias=False)
        self.text_proj = nn.Linear(768, args.proj_dim, bias=False)
        self.criterion = InferenceCriterion(args.use_vision_cls_token)

    def compute_logits(
        self,
        pixel_values: torch.Tensor,
        encoded_key_phrases: Any,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        image_tokens = self.vision_proj(self.vision_encoder(pixel_values))
        encoding = encoded_key_phrases[0] if isinstance(encoded_key_phrases, list) else encoded_key_phrases
        text = self.text_encoder(
            encoding["input_ids"].to(pixel_values.device),
            encoding["attention_mask"].to(pixel_values.device),
        )
        text_features = self.text_proj(text).reshape(-1, self.args.proj_dim)
        compute_tokens = image_tokens
        if not self.criterion.use_vision_cls_token:
            compute_tokens = compute_tokens[:, 1:]
        temperature = self.criterion.loss_temperature.exp()
        logits, similarity = self.criterion.similarity_logit(
            text_features,
            compute_tokens,
            temperature,
        )
        if self.criterion.use_vision_cls_token:
            similarity = similarity[:, :, 1:]
        return {"logits": logits / temperature, "similarity_scores": similarity}
