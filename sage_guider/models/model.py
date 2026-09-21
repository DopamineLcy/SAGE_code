import torch
import torch.nn as nn
import importlib.util
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder
from models.loss import Loss as InfoNCELoss

class VisionEncoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        model_name = str(Path(args.external_model_root) / "rad-dino-maira-2")
        attention_backend = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.rad_dino_model = AutoModel.from_pretrained(model_name, attn_implementation=attention_backend, torch_dtype=torch.bfloat16)
        for param in self.rad_dino_model.parameters(): param.requires_grad = False
        self.rad_dino_model.eval()
        self.rad_dino_output_layer = getattr(args, "rad_dino_output_layer", -1)
        if self.rad_dino_output_layer != -1:
            self.layer_norm = nn.LayerNorm(768)
        self.feature_dim = 768

        self.use_extra_pos_embed = getattr(args, "use_extra_pos_embed", False)
        if self.use_extra_pos_embed:
            num_patches = 1369 + 1 
            self.extra_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.feature_dim))
            nn.init.trunc_normal_(self.extra_pos_embed, std=0.02)

        dinov2_config = Dinov2Config(hidden_size=768, num_hidden_layers=args.num_hidden_layers if args else 2, use_layer_norm=False, attn_implementation=attention_backend)
        print(f"Using DINOv2 with {args.num_hidden_layers} hidden layers")
        self.transformer_blocks = Dinov2Encoder(dinov2_config)

    def forward(self, images):
        device = images.device
        with torch.no_grad():
            if images.dtype != torch.bfloat16: images = images.to(torch.bfloat16)
            inputs = {'pixel_values': images.to(device)}
            if self.rad_dino_output_layer == -1:
                outputs = self.rad_dino_model(**inputs)
                patch_features = outputs.last_hidden_state
            else:
                outputs = self.rad_dino_model(**inputs, output_hidden_states=True)
                patch_features = outputs.hidden_states[self.rad_dino_output_layer]
                patch_features = self.layer_norm(patch_features)
        
        if self.use_extra_pos_embed:
            patch_features = patch_features + self.extra_pos_embed
        outputs = self.transformer_blocks(patch_features)
        return outputs["last_hidden_state"]

class TextEncoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        url = str(Path(args.external_model_root) / "BiomedVLP-CXR-BERT-specialized")
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(url, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        for param in self.model.parameters(): param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids.dim() == 3:
            batch_size, num_entities, seq_len = input_ids.shape
        elif input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            num_entities = 1
            input_ids = input_ids.unsqueeze(1)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1)
        else:
             batch_size, num_entities, seq_len = input_ids.shape

        outputs = self.model(input_ids=input_ids.view(-1, seq_len), attention_mask=attention_mask.view(-1, seq_len), return_dict=True)
        return outputs.last_hidden_state[:, 0, :].view(batch_size, num_entities, -1)

class BaseModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.vision_encoder = VisionEncoder(args=args)
        self.text_encoder = TextEncoder(args=args)
        
        proj_dim = args.proj_dim if args else None
        print(f"Using projection dimension: {proj_dim}")
        self.vision_proj = nn.Linear(self.vision_encoder.feature_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(768, proj_dim, bias=False)
        self._init_weights(self.vision_proj); self._init_weights(self.text_proj)
        self.vision_encoder.transformer_blocks.apply(self._init_weights)

        self.criterion = InfoNCELoss(
            attn_temperature=args.attn_temperature if args else None,
            use_vision_cls_token=args.use_vision_cls_token if args else False,
            world_size=self.args.world_size,
            nli_model_path=Path(args.external_model_root) / "nli-deberta-v3-small",
        )

        print("Using contrastive loss")
        print(f"Using vision cls token: {args.use_vision_cls_token if args else False}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
    
    def forward(self, batch, device):
        images = batch['images'].to(device, non_blocking=True)
        # report_paths = batch['report_paths']
        presence_values = batch['presence_values'].to(device, non_blocking=True) 
        
        # Pre-computed flattened inputs from collate_fn
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attn_mask = batch['attention_mask'].to(device, non_blocking=True)
        text_is_pos_tensor = batch['text_is_pos'].to(device, non_blocking=True)
        sampled_entity_ids_tensor = batch['text_entity_ids'].to(device, non_blocking=True)
        src_indices_tensor = batch['text_src_indices'].to(device, non_blocking=True)
        text_attributes = batch.get('text_attributes', None)
        image_attributes = batch.get('image_attributes', None)
        
        # Unwrap if wrapped (to avoid pin_memory overhead)
        if hasattr(text_attributes, 'data'): text_attributes = text_attributes.data
        if hasattr(image_attributes, 'data'): image_attributes = image_attributes.data

        B_total = input_ids.shape[0]
        input_ids = input_ids.view(B_total, 1, -1)
        attn_mask = attn_mask.view(B_total, 1, -1)

        text_feats = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_feats = self.text_proj(text_feats) 
        text_features_flat = text_feats.view(-1, self.args.proj_dim)

        image_tokens = self.vision_encoder(images)
        image_tokens = self.vision_proj(image_tokens)

        loss_output = self.criterion(
            vision_tokens=image_tokens,
            text_features=text_features_flat,
            image_presence_map=presence_values,  # [N_total]
            text_entity_ids=sampled_entity_ids_tensor, # [N_total]
            text_is_pos=text_is_pos_tensor,    # [N_total]
            text_src_indices=src_indices_tensor,
            text_attributes=text_attributes,
            image_attributes=image_attributes
        )

        return loss_output
